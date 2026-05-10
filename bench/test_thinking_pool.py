#!/usr/bin/env python3
"""
Thinking pool round-trip test.

Verifies that after generating a thinking response, the pool checkpoint
is canonicalized so the next request gets a pool hit (small suffix) instead
of a full re-prefill.

Usage:
    ./bench/test_thinking_pool.py [--port 8211]
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

SERVER_BIN = "./ds4-server"
MODEL = "./ds4flash.gguf"


def send_request(port, session_id, messages, model="deepseek-v4-flash", max_tokens=200):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "X-Session-Id": session_id,
    })

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
        elapsed = time.time() - t0
        usage = body.get("usage", {})
        choices = body.get("choices", [])
        msg = choices[0].get("message", {}) if choices else {}
        return {
            "elapsed": elapsed,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "content": msg.get("content", ""),
            "reasoning_content": msg.get("reasoning_content", ""),
            "finish_reason": choices[0].get("finish_reason", "") if choices else "",
        }


def send_request_with_tools(port, session_id, messages, tools,
                           model="deepseek-v4-flash", max_tokens=200):
    """Send a chat completion with tools."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "tools": tools,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "X-Session-Id": session_id,
    })

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
        elapsed = time.time() - t0
        usage = body.get("usage", {})
        choices = body.get("choices", [])
        msg = choices[0].get("message", {}) if choices else {}
        return {
            "elapsed": elapsed,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "content": msg.get("content", ""),
            "reasoning_content": msg.get("reasoning_content", ""),
            "tool_calls": msg.get("tool_calls", []),
            "finish_reason": choices[0].get("finish_reason", "") if choices else "",
        }


def start_server(port):
    log_path = f"/tmp/ds4-think-pool-test-{port}.log"
    log_file = open(log_path, "w")
    cmd = [
        SERVER_BIN, "--model", MODEL,
        "--ctx", "32000",
        "--port", str(port),
    ]
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
    print(f"  Server pid={proc.pid}, log={log_path}")
    for i in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            print(f"  Ready in {i+1}s")
            return proc, log_path
        except:
            time.sleep(1)
            if proc.poll() is not None:
                print(f"  ERROR: server exited with code {proc.returncode}")
                sys.exit(1)
    proc.kill()
    print("  ERROR: timeout")
    sys.exit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8211)
    args = parser.parse_args()

    if not os.path.exists(SERVER_BIN) or not os.path.exists(MODEL):
        print("ERROR: need ds4-server and model file")
        sys.exit(1)

    print("=" * 60)
    print("  THINKING POOL ROUND-TRIP TEST")
    print("=" * 60)

    proc, log_path = start_server(args.port)
    try:
        # --- Build a context of ~5K tokens to make prefill cost visible ---
        system = (
            "You are a helpful coding assistant. " * 50 +
            "Always think step by step before answering."
        )
        session_id = "think-test-1"

        # --- Turn 1: Generate with thinking ---
        print("\n  Turn 1: Initial thinking request...")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "What is the time complexity of merge sort? Be concise."},
        ]
        r1 = send_request(args.port, session_id, messages, model="deepseek-v4-flash")
        print(f"    elapsed={r1['elapsed']:.2f}s  prompt={r1['prompt_tokens']}  gen={r1['completion_tokens']}")
        print(f"    reasoning: {r1['reasoning_content'][:80]}...")
        print(f"    content: {r1['content'][:80]}")
        print(f"    finish: {r1['finish_reason']}")

        if not r1["content"]:
            print("    ERROR: no content generated")
            return

        # --- Turn 2: Follow-up (includes assistant response with reasoning) ---
        print("\n  Turn 2: Follow-up request (should get pool hit)...")
        messages.append({
            "role": "assistant",
            "content": r1["content"],
            "reasoning_content": r1["reasoning_content"],
        })
        messages.append({
            "role": "user",
            "content": "And what about quicksort?",
        })
        r2 = send_request(args.port, session_id, messages, model="deepseek-v4-flash")
        print(f"    elapsed={r2['elapsed']:.2f}s  prompt={r2['prompt_tokens']}  gen={r2['completion_tokens']}")
        print(f"    reasoning: {r2['reasoning_content'][:80]}...")
        print(f"    content: {r2['content'][:80]}")

        # --- Turn 3: Another follow-up ---
        print("\n  Turn 3: Another follow-up (should also get pool hit)...")
        messages.append({
            "role": "assistant",
            "content": r2["content"],
            "reasoning_content": r2["reasoning_content"],
        })
        messages.append({
            "role": "user",
            "content": "Compare them.",
        })
        r3 = send_request(args.port, session_id, messages, model="deepseek-v4-flash")
        print(f"    elapsed={r3['elapsed']:.2f}s  prompt={r3['prompt_tokens']}  gen={r3['completion_tokens']}")
        print(f"    content: {r3['content'][:80]}")

        # --- Also test non-thinking for comparison ---
        print("\n  Comparison: Non-thinking mode (deepseek-chat)...")
        session_id2 = "no-think-test-1"
        msgs2 = [
            {"role": "system", "content": system},
            {"role": "user", "content": "What is the time complexity of merge sort? Be concise."},
        ]
        r4 = send_request(args.port, session_id2, msgs2, model="deepseek-chat")
        msgs2.append({"role": "assistant", "content": r4["content"]})
        msgs2.append({"role": "user", "content": "And what about quicksort?"})
        r5 = send_request(args.port, session_id2, msgs2, model="deepseek-chat")
        print(f"    Turn 1: {r4['elapsed']:.2f}s (prompt={r4['prompt_tokens']})")
        print(f"    Turn 2: {r5['elapsed']:.2f}s (prompt={r5['prompt_tokens']})")

        # --- Golden path: thinking + tools (no tool call in response) ---
        print("\n  Golden path: Thinking + tools (model answers without calling tools)...")
        session_id3 = "think-tools-test"
        tools = [{
            "type": "function",
            "function": {
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}}
                }
            }
        }]
        msgs3 = [
            {"role": "system", "content": system},
            {"role": "user", "content": "What is the time complexity of merge sort? Be concise."},
        ]
        r6 = send_request_with_tools(args.port, session_id3, msgs3, tools,
                                      model="deepseek-v4-flash")
        print(f"    Turn 1: {r6['elapsed']:.2f}s (prompt={r6['prompt_tokens']}, gen={r6['completion_tokens']})")
        if r6["content"]:
            msgs3.append({
                "role": "assistant",
                "content": r6["content"],
                "reasoning_content": r6["reasoning_content"],
            })
            msgs3.append({"role": "user", "content": "And quicksort?"})
            r7 = send_request_with_tools(args.port, session_id3, msgs3, tools,
                                          model="deepseek-v4-flash")
            print(f"    Turn 2: {r7['elapsed']:.2f}s (prompt={r7['prompt_tokens']}, gen={r7['completion_tokens']})")
        else:
            print("    Turn 1 failed or called a tool — skipping")
            r7 = {"elapsed": 999, "prompt_tokens": 0, "completion_tokens": 0}

        # --- Check server log for canonicalization events ---
        print("\n  Server log (canonicalization events):")
        with open(log_path) as f:
            for line in f:
                if "canonical" in line or "common=" in line.lower():
                    print(f"    {line.strip()}")

        # --- Summary ---
        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        print(f"  Thinking mode:")
        print(f"    Turn 1 (cold):    {r1['elapsed']:.2f}s")
        print(f"    Turn 2 (pool?):   {r2['elapsed']:.2f}s")
        print(f"    Turn 3 (pool?):   {r3['elapsed']:.2f}s")
        print(f"  Non-thinking mode:")
        print(f"    Turn 1 (cold):    {r4['elapsed']:.2f}s")
        print(f"    Turn 2 (pool):    {r5['elapsed']:.2f}s")
        print()

        # The key metric: Turn 2 thinking prefill should be fast (pool hit)
        # We can tell from the prompt_tokens vs elapsed time.
        # Pool hit: most time is generation. Pool miss: significant prefill time.
        think_gen_time = r2["completion_tokens"] * 0.05  # thinking is slower (~20 t/s)
        think_prefill = max(0, r2["elapsed"] - think_gen_time)
        nothink_gen_time = r5["completion_tokens"] * 0.04
        nothink_prefill = max(0, r5["elapsed"] - nothink_gen_time)

        print(f"  Estimated prefill times:")
        print(f"    Thinking Turn 2:  ~{think_prefill:.1f}s (total {r2['elapsed']:.1f}s - gen {think_gen_time:.1f}s)")
        print(f"    Non-think Turn 2: ~{nothink_prefill:.1f}s (total {r5['elapsed']:.1f}s - gen {nothink_gen_time:.1f}s)")

        # Success criterion: thinking prefill should be < 3s (pool hit)
        # A full re-prefill of 400 tokens would take ~2-3s on this model
        if think_prefill < 3.0:
            print(f"  ✓ PASS: Thinking Turn 2 prefill ~{think_prefill:.1f}s (< 3s = pool hit)")
        else:
            print(f"  ✗ FAIL: Thinking Turn 2 prefill ~{think_prefill:.1f}s (>= 3s = likely pool miss)")
            print(f"          Check log for 'canonicalized' messages")

        # Golden path check
        if r7.get("prompt_tokens"):
            tools_gen_time = r7["completion_tokens"] * 0.05
            tools_prefill = max(0, r7["elapsed"] - tools_gen_time)
            print(f"\n  Golden path (thinking + tools, no tool call):")
            print(f"    Turn 1: {r6['elapsed']:.2f}s")
            print(f"    Turn 2: {r7['elapsed']:.2f}s (~{tools_prefill:.1f}s prefill)")
            if tools_prefill < 3.0:
                print(f"  \u2713 PASS: Golden path Turn 2 prefill ~{tools_prefill:.1f}s (< 3s = cache hit)")
            else:
                print(f"  \u2717 FAIL: Golden path Turn 2 prefill ~{tools_prefill:.1f}s (>= 3s = cache miss)")
                print(f"          BPE round-trip may be lossy with tools+thinking")

        print(f"\n  Full log: {log_path}")
        print(f"{'='*60}\n")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except:
            proc.kill()


if __name__ == "__main__":
    main()
