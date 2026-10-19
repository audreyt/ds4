#!/usr/bin/env python3
"""Vision conversion inventory and malformed-source rejection (no model needed)."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-tools"))
from deepseek41_vision import build_plan, sidecar_metadata, source_shapes
from glm53_quantize import QTYPE_BF16, align, kv_string


class Source:
    def __init__(self):
        self.tensors = {name: {"dtype": "BF16", "shape": list(shape),
                               "nbytes": math.prod(shape) * 2}
                        for name, shape in source_shapes().items()}

    def info(self, name):
        return self.tensors[name]


class VisionConversion(unittest.TestCase):
    def test_lossless_plan(self):
        source = Source()
        source.tensors["layers.0.ffn.gate.bias_vl"] = {"dtype": "F32"}
        plan = build_plan(source)
        self.assertEqual(len(plan), 266)
        self.assertEqual(sum(item.nbytes for item in plan), 970536960)
        end = 0
        for item in plan:
            self.assertEqual(item.qtype, QTYPE_BF16)
            self.assertTrue(item.raw_copy)
            self.assertEqual(item.source, item.name)
            self.assertEqual(item.shape, tuple(reversed(source.info(item.name)["shape"])))
            self.assertEqual(item.offset, end)
            end += align(item.nbytes)

    def test_reject_missing_extra_and_wrong_tensors(self):
        source = Source()
        del source.tensors["image_start"]
        with self.assertRaisesRegex(ValueError, "inventory"):
            build_plan(source)
        source = Source()
        source.tensors["image_pad"] = copy.deepcopy(source.tensors["image_end"])
        with self.assertRaisesRegex(ValueError, "inventory"):
            build_plan(source)
        for field, value in (("dtype", "F32"), ("shape", [4096]), ("nbytes", 0)):
            source = Source()
            source.tensors["image_end"][field] = value
            with self.assertRaises(ValueError):
                build_plan(source)

    def test_sidecar_records_source_revision(self):
        # ds4 pairs the sidecar with the language GGUF by this revision.
        config = {
            "model_type": "deepseek_v41", "image_token_id": 129264,
            "text_config": {"hidden_size": 5120, "num_hidden_layers": 40, "n_routed_experts": 384},
            "vision_config": {"num_hidden_layers": 32, "hidden_size": 1024,
                "num_attention_heads": 16, "intermediate_size": 2816, "patch_size": 14,
                "rope_theta": 10000, "downsample_ratio": 3, "max_image_tokens": 1024,
                "min_pixels": 295936, "max_wh_ratio": None},
        }
        revision = "1bae0e1088993b9b82de08f59de5be5a34e42170"
        with tempfile.TemporaryDirectory() as hf:
            Path(hf, "config.json").write_text(json.dumps(config))
            records = sidecar_metadata(hf, revision)
        self.assertIn(kv_string("general.source.revision", revision), records)


if __name__ == "__main__":
    unittest.main()
