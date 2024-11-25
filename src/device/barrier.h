/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

 __device__ __attribute__((noinline)) void  copyShmemData(struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclWork* workHead){
  const int tid = threadIdx.x;
  int x = tid;
  int total = 0, y;
  int num = MAXCHANNELS/64 > 0 ? MAXCHANNELS/64 : 1;

  switch (tid/WARP_SIZE) {
  case 0:
	//ncclShmem.channelId = blockIdx.x;
    for (int i = 0; i < num; i++) {
      if (channelMask.masks[i] & (1ull<<x)) {
        y = __popcll(channelMask.masks[i] & ((1ull<<x)-1));
        y = total + y;
        if (blockIdx.x == y) {
          ncclShmem.channelId = x + total;
	  break;
        }
      }
      if (WARP_SIZE < 64) {
        x = WARP_SIZE + tid;
        if (channelMask.masks[i] & (1ull<<x)) {
	  y = __popcll(channelMask.masks[i] & ((1ull<<x)-1));
	  y = y + total;
          if (blockIdx.x == y) {
	    ncclShmem.channelId = x + total;
	    break;
	  }
        }
      }
      total = total + __popcll(channelMask.masks[i]);
    }
    break;
  case 1:
    if (tid < WARP_SIZE + NCCL_MAX_GROUPS)
      ncclShmem.groups[tid-WARP_SIZE].barrier = 0;
    break;
  case 2:
    if (tid < 2*WARP_SIZE + NCCL_MAX_GROUPS*NCCL_MAX_GROUPS)
      ncclShmem.groups[(tid-2*WARP_SIZE)/NCCL_MAX_GROUPS].barrier_next[(tid-2*WARP_SIZE)%NCCL_MAX_GROUPS] = 0;
    break;
  case 3:
    /* set abort flag to 0 */
    if (tid == 3*WARP_SIZE) ncclShmem.aborted = 0;
    break;
  default:
    break;
  }
  __synclds(); // publish ncclShmem.channelId
  // To map blockId to channelId, we need the n'th set bit of channelMask which
  // is the inverse of counting the number of set bits among the the first n.
  int channelId = ncclShmem.channelId;

  if (true) {
    void *dst, *src;
    int bytes;
    // Use first 3 warps to load comm, channel, and work into shmem
    switch (tid/WARP_SIZE) {
    case 0:
      dst = &ncclShmem.comm;
      src = comm;
      bytes = sizeof(ncclDevComm);
      static_assert(sizeof(ncclDevComm) <= 16*WARP_SIZE, "ncclDevComm cannot be loaded by a single warp in one insn.");
      break;
    case 1:
      // Get address of channel without incurring indirect load from ncclDevComm::channels
      dst = &ncclShmem.channel;
      src = &((ncclDevCommAndChannels*)comm)->channels[channelId];
      bytes = sizeof(ncclDevChannel);
      static_assert(sizeof(ncclDevChannel) <= 16*WARP_SIZE, "ncclDevChannel cannot be loaded by a single warp in one insn.");
      break;
    case 2:
      dst = &ncclShmem.work;
      src = workHead + blockIdx.x;
      bytes = sizeof(ncclWork);
      static_assert(sizeof(ncclWork) <= 16*WARP_SIZE, "ncclWork cannot be loaded by a single warp in one insn.");
      break;
    default:
      bytes = 0;
      break;
    }
    if (bytes) copyToShmem16(tid%WARP_SIZE, dst, src, bytes);
  }
#ifdef ENABLE_COLLTRACE
  if (tid == 0) {
    ncclShmem.collTrace = comm->collTrace + COLLTRACE_NUM_ITEMS*ncclShmem.channelId;
    ncclShmem.collTraceTail = comm->collTraceTail + ncclShmem.channelId;
  }
#endif
  __synclds(); // publish shmem
}

namespace {
#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__)
  __device__ void runRing(ncclWorkElem *args) {
#else
  __device__ __attribute__((noinline)) void runRing(ncclWorkElem *args, uint8_t* sendBuffer, uint8_t* recvBuffer, ncclDevComm* devComm) {
#endif
    const int tid = threadIdx.x;
    const int nthreads = args->nWarps * WARP_SIZE;
    ncclRing *ring = &ncclShmem.channel.ring;
    const int nranks = ncclShmem.comm.nRanks;
    const int rank = ncclShmem.comm.rank;

    const int prevRank = ring->userRanks[nranks-1];
    const int root = args->root;
    const size_t chunkCount = args->chunkCount;
    const size_t channelCount = args->workCount;
    const size_t gridOffset = args->workOffset;
    size_t offset;
    int nelem;
    Primitives<uint8_t, FuncSum<uint8_t>, FanSymmetric<1>, 0, ProtoLL, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff, 0, 0, 0, 0);

    if (prevRank == root) {
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.send(offset, nelem);
      }
    }
    else if (rank == root) {
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.recvReduceCopy(offset, offset, nelem, /*postOp=*/true);
      }
    }
    else {
      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        prims.recvReduceSend(offset, nelem);
      }
    }
  }
}

template<typename T>
 __device__ __attribute__((noinline)) void runRing2(ncclWorkElem *args) {
 }

__global__ __forceinline__ void rcclWaitForAllRanksBarrier(struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclWork* workHead, ncclWorkElem *args, uint8_t* sendBuffer, uint8_t* recvBuffer, ncclDevComm* devComm)
{
  copyShmemData(comm, channelMask, workHead);
  runRing(args, sendBuffer, recvBuffer, devComm);
}