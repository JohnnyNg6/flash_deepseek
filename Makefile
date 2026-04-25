CC      := clang
CFLAGS  := -O3 -Wall -Wextra -fobjc-arc -DACCELERATE_NEW_LAPACK -std=gnu11
OBJCFLAGS := -O3 -Wall -Wextra -fobjc-arc -DACCELERATE_NEW_LAPACK
FRAMEWORKS := -framework Metal -framework Foundation -framework Accelerate
LDFLAGS := -lpthread

MODEL_DIR ?= $(HOME)/models/deepseek-r1-q4km/DeepSeek-R1-Q4_K_M

all: repack_experts infer

gguf.o: gguf.c gguf.h
	$(CC) $(CFLAGS) -c gguf.c -o $@

quant.o: quant.c quant.h
	$(CC) $(CFLAGS) -c quant.c -o $@

tokenizer.o: tokenizer.c tokenizer.h
	$(CC) $(CFLAGS) -c tokenizer.c -o $@

repack_experts: repack_experts.c gguf.o quant.o
	$(CC) $(CFLAGS) repack_experts.c gguf.o quant.o -o $@ $(LDFLAGS)

infer: infer.m gguf.o quant.o tokenizer.o shaders.metal
	$(CC) $(OBJCFLAGS) $(FRAMEWORKS) infer.m gguf.o quant.o tokenizer.o -o $@ $(LDFLAGS)

clean:
	rm -f *.o repack_experts infer

.PHONY: all clean repack run
