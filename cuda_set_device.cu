#include "cuda_runtime.h"
#include "mpi.h"

#include "cuda_set_device.h"

extern "C" void cuda_set_device();
extern "C" void cuda_my_memset(void* ptr, int value, size_t count);

#include <stdio.h>

void cuda_set_device(){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    cudaSetDevice(rank/32);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, rank/32);
    printf("Rank %d: Using device %d: %s\n", rank, rank/32, prop.name);
}

void cuda_my_memset(void* ptr, int value, size_t count){
    cudaMemset(ptr, value, count);
}
