# Top-level build.
#   make            # icx (Intel):   seq + omp
#   make CC=gcc     # gcc:           seq + omp
#   make mpi        # mpiicx/mpicc:  mpi (collective + p2p)
#   CUDA is built on Colab:  nvcc -O3 -arch=sm_75 cuda/matmul_cuda.cu -o matmul_cuda -lcublas
CC  ?= icx
STD ?= -std=c11
ifeq ($(CC),gcc)
  XHOST ?= -march=native
  OMP   ?= -fopenmp
else
  XHOST ?= -xHost
  OMP   ?= -qopenmp
endif
CFLAGS ?= -O3 $(XHOST) $(STD)

# MPI wrapper: mpiicx if available, else mpicc
MPICC ?= $(shell command -v mpiicx 2>/dev/null || echo mpicc)
ifeq ($(notdir $(MPICC)),mpicc)
  MPIXHOST ?= -march=native
else
  MPIXHOST ?= -xHost
endif

all: seq/matmul_seq omp/matmul_omp

seq/matmul_seq: seq/matmul_seq.c
	$(CC) $(CFLAGS) $< -o $@

omp/matmul_omp: omp/matmul_omp.c
	$(CC) $(CFLAGS) $(OMP) $< -o $@

mpi: mpi/matmul_mpi mpi/matmul_mpi_p2p

mpi/matmul_mpi: mpi/matmul_mpi.c
	$(MPICC) -O3 $(MPIXHOST) $(STD) $< -o $@

mpi/matmul_mpi_p2p: mpi/matmul_mpi.c
	$(MPICC) -O3 $(MPIXHOST) $(STD) -DP2P $< -o $@

clean:
	rm -f seq/matmul_seq seq/matmul_naive omp/matmul_omp mpi/matmul_mpi mpi/matmul_mpi_p2p cuda/matmul_cuda

.PHONY: all mpi clean
