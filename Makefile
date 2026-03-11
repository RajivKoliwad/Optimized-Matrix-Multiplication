# Makefile — build
NVCC ?= nvcc
ARCH ?= native
CXXSTD ?= c++17
OPTFLAGS ?= -O2
LIBS      := -lcublas

all: tmm

tmm: tiled-mat-mul.cu
	$(NVCC) $(OPTFLAGS) -std=$(CXXSTD) -arch=$(ARCH) -o $@ $< $(LIBS)
clean:
	rm -f tmm
