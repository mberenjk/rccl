#include <iostream>
#include <cuda_runtime.h>

__global__ void rcclWaitForAllRanksBarrier(int* data) {
	 int barrier_signal = 1;
    int* d_signal;
    cudaMalloc(&d_signal, sizeof(int));
    cudaMemcpy(d_signal, &barrier_signal, sizeof(int), cudaMemcpyHostToDevice);

    // Intra-node barrier using NCCL
    NCCLCHECK(ncclAllReduce((const void*)d_signal, (void*)d_signal, 1, ncclInt, ncclSum, nccl_comm, 0));
    cudaDeviceSynchronize();

    // Inter-node barrier using MPI
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    cudaFree(d_signal);
}
