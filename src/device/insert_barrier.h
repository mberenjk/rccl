#include <iostream>
#include <cuda_runtime.h>

__global__ void rcclWaitForAllRanksBarrier(int* data, int rank, int numRanks, int* tempBuffer) {
    int prevRank = (rank - 1 + numRanks) % numRanks;
    int nextRank = (rank + 1) % numRanks;

    for (int i = 0; i < numRanks - 1; ++i) {
        *data += tempBuffer[i];
    }

    *tempBuffer = *data;
}


