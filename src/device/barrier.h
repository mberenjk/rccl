/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

 __device__ __attribute__((noinline)) void  copyShmemData(struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclDevKernelArgs const* args){//struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclWork* workHead){
  const int tid = threadIdx.x;
  int tn = blockDim.x;
  int x = tid;
  int total = 0, y;
  int num = MAXCHANNELS/64 > 0 ? MAXCHANNELS/64 : 1;
  if (tid < sizeof(ncclDevKernelArgs)/sizeof(uint32_t)) {
    ((uint32_t*)&ncclShmem.args)[tid] = ((uint32_t*)args)[tid];
  }
  
  switch (tid/WARP_SIZE) {
  case 0:
    ncclShmem.channelId = blockIdx.x;
    for (int i = 0; i < num; i++) {
      if (args->channelMask.masks[i] & (1ull<<x)) {
        y = __popcll(args->channelMask.masks[i] & ((1ull<<x)-1));
        y = total + y;
        if (blockIdx.x == y) {
          ncclShmem.channelId = x + total;
          break;
        }
      }
      if (WARP_SIZE < 64) {
        x = WARP_SIZE + tid;
        if (args->channelMask.masks[i] & (1ull<<x)) {
          y = __popcll(args->channelMask.masks[i] & ((1ull<<x)-1));
          y = y + total;
          if (blockIdx.x == y) {
            ncclShmem.channelId = x + total;
            break;
          }
        }
      }
      total = total + __popcll(args->channelMask.masks[i]);

    }
    break;
  case 1:
    if (tid < WARP_SIZE + NCCL_MAX_GROUPS)
      ncclShmem.groups[tid-WARP_SIZE].barrier = 0;

    break;
  case 2:
#ifdef ENABLE_FAULT_INJECTION
    /* load faults injection before first sync threads */
    if (tid == 2*WARP_SIZE) ncclShmem.faults = comm->faults;
#endif
    break;
  case 3:
    /* set abort flag to 0 */
    if (tid == 3*WARP_SIZE) ncclShmem.aborted = 0;
    break;
  default:
    break;
  }
  __syncthreads(); // publish ncclShmem.{args, channelId}
  
  // Use first 2 warps to load comm and channel, and reamaining load work batch.
  switch (tid/WARP_SIZE) {
  case 0:
    {
      void* dst = &ncclShmem.comm;
      void* src = comm;
      int bytes = sizeof(ncclDevComm);
      static_assert(sizeof(ncclDevComm) <= 16*WARP_SIZE, "ncclDevComm cannot be loaded by a single warp in one insn.");
      copyToShmem16(tid, dst, src, bytes);
    } break;
  case 1:
    { // Get address of channel without incurring indirect load from ncclDevComm::channels
      void* dst = &ncclShmem.channel;
      void* src = &((ncclDevCommAndChannels*)comm)->channels[ncclShmem.channelId];
      int bytes = sizeof(ncclDevChannel);
      static_assert(sizeof(ncclDevChannel) <= 16*WARP_SIZE, "ncclDevChannel cannot be loaded by a single warp in one insn.");
      copyToShmem16(tid-WARP_SIZE, dst, src, bytes);
    } break;
  default:
    {
      int subtid = tid - 2*WARP_SIZE;
      int subtn = tn - 2*WARP_SIZE;
      // Coverity reports a possible thread divergence due to not all threads participating in the collective.
      // However, the code ensures that the participation is on a per-warp basis.
      // coverity[device_thread_diverged:FALSE]
      loadWorkBatchToShmem(subtid, subtn, args, /*batchIx=*/blockIdx.x);
    } break;
  }
  __syncthreads(); // publish shmem
#ifdef ENABLE_PROFILING
  if (tid == 0) {
    ncclShmem.prof.count = 0;
    ncclShmem.prof.seq = ncclShmem.comm.devProf[blockIdx.x].seq;
  }
#endif
}

namespace {
#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__)
  __device__ void runRing(ncclDevWorkColl *args) {
#else
  __device__ __attribute__((noinline)) void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
#endif
    const int bid = ncclShmem.channelId - work->channelLo;
    ncclRing *ring = &ncclShmem.channel.ring;
    const int rank = ring->userRanks[0];
    const int nextRank = ring->userRanks[1];
    const int root = work->root;
    ssize_t size;
    ssize_t chunkCount;
    ssize_t channelCount;
    ssize_t gridOffset;
    using Proto = ProtoSimple<1, 1, 2, 0, 1>; //hard-coded COLL_UNROLL to 2
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, 1, &size, &gridOffset, &channelCount, &chunkCount);
    size =1;
    channelCount = 1;
    chunkCount = 1;
    gridOffset = 0;
    size_t offset;
    int nelem;
    int workNthreads;
    bool isNetOffload = work->isOneRPN && work->netRegUsed;
    char *inputBuf = (char*)work->sendbuff;
    char *outputBuf = (char*)work->recvbuff;
    workNthreads = isNetOffload ? WARP_SIZE : nthreads;

    #if defined(ENABLE_NPKIT)
    int npKitCtxIdx = bid;
    #endif

    #if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, NPKIT_GET_CPU_TIMESTAMP_FROM_BLOCK,
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
    #endif

    #if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
    #endif

    #if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_BROADCAST_RING_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_BROADCAST_RING_ENTRY, size*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
    #endif
    if (tid < workNthreads) {
      // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
      // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
    Primitives<uint8_t, FuncSum<uint8_t>, FanSymmetric<1>, 0, Proto, 0>
      prims(tid, workNthreads, &ring->prev, &ring->next, inputBuf, outputBuf, 0, 0, 0, 0, work);

    #if defined(ENABLE_NPKIT)
      if (tid == 0) {
        prims.npKitCtxIdx = npKitCtxIdx;
      }
    #endif      
    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);
        //printf("nelem = %d \n", nelem);

        if (rank == root) {
          if (inputBuf == outputBuf || isNetOffload) {
            prims.directSend(offset, offset, nelem);
          } else {
            prims.directCopySend(offset, offset, nelem);
          }
        } else if (nextRank == root) {
          prims.directRecv(offset, offset, nelem);
        } else {
          prims.directRecvCopyDirectSend(offset, offset, nelem);
        }
      }
    } else if (inputBuf != outputBuf && rank == root) {
      inputBuf = inputBuf + gridOffset;
      outputBuf = outputBuf + gridOffset;
      reduceCopy<2, FuncSum<uint8_t>, uint8_t, 0, 1, 1, 0, 1, 1, /*PreOpSrcs=*/0>
        (tid - workNthreads, nthreads - workNthreads, work->redOpArg, &work->redOpArg, false, 1, (void**)&inputBuf, 1, (void**)&outputBuf, channelCount);
    }
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_BROADCAST_RING_EXIT)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_BROADCAST_RING_EXIT, size*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
    if (isNetOffload) barrier_sync(14, nThreads);
#endif
  }
}



__global__ __forceinline__ void rcclWaitForAllRanksBarrier(struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclDevWorkColl* work,  struct ncclDevKernelArgs const* args)//struct ncclDevComm* comm, struct channelMasks channelMask, struct ncclWork* workHead, ncclDevWorkColl *args, ncclDevComm* devComm)
{
  copyShmemData(comm, channelMask, args);
  int tid = threadIdx.x;
  int tn = blockDim.x;
  int w = 0;
  //struct ncclDevWorkColl* work = (struct ncclDevWorkColl*)(ncclShmem.workStorage + w*ncclShmem.workSize);
  int subtn = work->nWarps*WARP_SIZE;
  runRing(tid, subtn, work);
}