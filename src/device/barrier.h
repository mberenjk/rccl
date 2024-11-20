/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"


namespace {
#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__)
  __device__ void runRing(ncclWorkElem *args) {
#else
  __device__ __attribute__((noinline)) void runRing(ncclWorkElem *args, uint8_t* sendBuffer, uint8_t* recvBuffer, int rank, int nranks, ncclDevComm* devComm) {
#endif
    const int tid = threadIdx.x;
    const int nthreads = args->nWarps * WARP_SIZE;
    //ncclRing *ring = &ncclShmem.channel.ring;
    // const int nranks = 4;//ncclShmem.comm.nRanks;
    // const int rank = ncclShmem.comm.rank;
    ncclRing* ring = &devComm->channels[0].ring;

    //const int prevRank = ring->userRanks[nranks-1];
    // const int root = args->root;
    // const size_t chunkCount = args->chunkCount;
    // const size_t channelCount = args->workCount;
    // const size_t gridOffset = args->workOffset;
    size_t offset;
    int nelem;
    // if(ring->prev > nranks - 1) ring->prev = nranks - 1;
    // if(ring->next > nranks - 1) ring->next = nranks - 1;
    // if(ring->prev < 0) ring->prev = 0;
    // if(ring->next < 0) ring->next = 0;
    // args->count = 1;
    // args->sendbuff = sendBuffer;
    // args->recvbuff = recvBuffer;
    
    // printf(" nthreads = %d \n", nthreads);
    printf(" nthreads = %d  ring->prev = %d ring->next = %d  \n", nthreads, ring->prev, ring->next);
    // //if(ring->prev >= 0 && ring->prev < nranks - 1 && ring->next >= 0 && ring->next < nranks - 1)
    // if(tid == 0)
    Primitives<uint8_t, FuncSum<uint8_t>, FanSymmetric<1>, 0, ProtoLL, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff, 0, 0, 0, 0);

    // if (prevRank == root) {
    //   for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
    //     offset = gridOffset + elemOffset;
    //     nelem = min(chunkCount, channelCount - elemOffset);
    //     prims.send(offset, nelem);
    //   }
    // }
    // else if (rank == root) {
    //   for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
    //     offset = gridOffset + elemOffset;
    //     nelem = min(chunkCount, channelCount - elemOffset);
    //     prims.recvReduceCopy(offset, offset, nelem, /*postOp=*/true);
    //   }
    // }
    // else {
    //   for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
    //     offset = gridOffset + elemOffset;
    //     nelem = min(chunkCount, channelCount - elemOffset);
    //     prims.recvReduceSend(offset, nelem);
    //   }
    // }
  }
}

template<typename T>
 __device__ __attribute__((noinline)) void runRing2(ncclWorkElem *args) {
 }

__global__ __forceinline__ void rcclWaitForAllRanksBarrier(ncclWorkElem *args, uint8_t* sendBuffer, uint8_t* recvBuffer, int rank, int nranks, ncclDevComm* devComm)
{
  //runRing2<uint8_t>(args);
  runRing(args, sendBuffer, recvBuffer, rank, nranks, devComm);
}