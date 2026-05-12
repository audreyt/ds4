CC ?= cc
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
QUALITY_LDLIBS := -lm -pthread -framework Foundation -framework Metal
else
NATIVE_CPU_FLAG ?= -march=native
QUALITY_LDLIBS := -lm -pthread
endif

CFLAGS ?= -O3 -Wall -Wextra -std=c11 $(NATIVE_CPU_FLAG)
CPPFLAGS ?= -D_GNU_SOURCE

.PHONY: all clean quality-score

all: deepseek4-quantize

deepseek4-quantize: deepseek4-quantize.c quants.c quants.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ deepseek4-quantize.c quants.c -lm -pthread

quality-score:
	$(MAKE) -C ..
	$(CC) $(CFLAGS) -I.. -o quality-testing/score_official quality-testing/score_official.c ../ds4.o ../ds4_metal.o $(QUALITY_LDLIBS)

clean:
	rm -f deepseek4-quantize quality-testing/score_official
