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
  template<typename T>//, typename RedOp>
#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx942__) && !defined(__gfx950__)
  __device__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
#else
  __device__ __attribute__((noinline)) void runRing_bc(int tid, int nthreads, struct ncclDevWorkColl* work) {
#endif
#if defined(ENABLE_NPKIT)
    const int bid = ncclShmem.channelId - work->channelLo;
    int npKitCtxIdx = bid; // unused variable - compiler warning
#endif
    ncclRing *ring = &ncclShmem.channel.ring;
    const int rank = ring->userRanks[0];
    const int nextRank = ring->userRanks[1];
    const int root = work->root;
    ssize_t size;
    ssize_t chunkCount;
    ssize_t channelCount;
    ssize_t gridOffset;
    using Proto = ProtoSimple<1, 1,0,1>;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &size, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    int nelem;
    int workNthreads;
    bool isNetOffload = work->isOneRPN && work->netRegUsed;


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

    T *inputBuf = (T*)work->sendbuff;
    T *outputBuf = (T*)work->recvbuff;
    workNthreads = isNetOffload ? WARP_SIZE : nthreads;

    if (tid < workNthreads) {
      // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
      // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, FuncSum<uint8_t>, FanSymmetric<1>, 0, Proto, 0>
        prims(tid, workNthreads, &ring->prev, &ring->next, inputBuf, outputBuf, work->redOpArg, 0, work->connIndex, work->connIndex, work);

#if defined(ENABLE_NPKIT)
      if (tid == 0) {
        prims.npKitCtxIdx = npKitCtxIdx;
      }
#endif

      for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
        offset = gridOffset + elemOffset;
        nelem = min(chunkCount, channelCount - elemOffset);

        if (rank == root) {
          if (inputBuf == outputBuf || isNetOffload) {
            prims.directSend(offset, offset, nelem);
          } else {
            prims.directCopySend(offset, offset, nelem);
          }
        } else if (nextRank == root) {
          prims.directRecv(offset, nelem);
        } else {
          prims.directRecvCopyDirectSend(offset, offset, nelem);
        }
      }
    } else if (inputBuf != outputBuf && rank == root) {
      inputBuf = inputBuf + gridOffset;
      outputBuf = outputBuf + gridOffset;
      reduceCopy<1, 0, FuncSum<uint8_t>, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs=*/0>
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


namespace {
  template<typename T, bool isNetOffload = false>
#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx942__) && !defined(__gfx950__)
  __device__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
#else
  __device__ __attribute__((noinline)) void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
#endif
#if defined(ENABLE_NPKIT)
    const int bid = ncclShmem.channelId - work->channelLo;
    int npKitCtxIdx = bid; // unused variable - compiler warning
#endif
    ncclRing *ring = &ncclShmem.channel.ring;
    const int *ringRanks = ring->userRanks;
    const int nranks = ncclShmem.comm.nRanks;
    ssize_t count, partOffset, partCount, chunkCount;
    using Proto = ProtoSimple<1, 1,0,1>;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &partOffset, &partCount, &chunkCount);
    ssize_t offset;
    ssize_t dataOffset;
    int nelem;
    int rankDest;


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

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_ENTRY)
    if (tid == 0) {
      NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_ENTRY, count*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
          ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
    }
#endif
    int workNthreads;
    T *inputBuf = (T*)work->sendbuff;
    T *outputBuf = (T*)work->recvbuff;

    // If isNetOffload == true, we only use 1 warp to drive Ring algo/network communication
    // and the rest of warps proceed to copy src data into dst buffer in parallel when AG
    // is not in-place.
    if (isNetOffload) {
      workNthreads = WARP_SIZE;
      chunkCount = NCCL_MAX_NET_SIZE;
    } else {
      workNthreads = nthreads;
    }

    if (tid < workNthreads) {
      // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
      // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
      // coverity[callee_ptr_arith:FALSE]
      Primitives<T, FuncSum<uint8_t>, FanSymmetric<1>, 0, Proto, 0, isNetOffload> prims
        (tid, workNthreads, &ring->prev, &ring->next, inputBuf, outputBuf, work->redOpArg, 0, work->connIndex, work->connIndex, work, NULL, isNetOffload ? NCCL_MAX_NET_SIZE : 0);

#if defined(ENABLE_NPKIT)
      if (tid == 0) {
        prims.npKitCtxIdx = npKitCtxIdx;
      }
#endif
      for (size_t elemOffset = 0; elemOffset < partCount; elemOffset += chunkCount) {
        /////////////// begin AllGather steps ///////////////
        nelem = min(chunkCount, partCount - elemOffset);
        dataOffset = partOffset + elemOffset;

        // step 0: push data to next GPU
        rankDest = ringRanks[0];
        offset = dataOffset + rankDest * count;

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_SEND_ENTRY)
        if (tid == 0) {
          NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_SEND_ENTRY, nelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
              ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
          prims.npKitDataProcessTotalTime = 0;
        }
#endif

        if ((inputBuf + dataOffset == outputBuf + offset) || isNetOffload) { // In place or onePPN
          prims.directSend(dataOffset, offset, nelem);
        } else {
          prims.directCopySend(dataOffset, offset, nelem);
        }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_SEND_EXIT)
        if (tid == 0) {
          NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_SEND_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, NPKIT_GET_GPU_TIMESTAMP(),
              ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        }
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_ENTRY)
        if (tid == 0 && nranks > 2) {
          NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_ENTRY, nelem*(nranks-2)*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
              ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
          prims.npKitDataProcessTotalTime = 0;
        }
#endif

        // k-2 steps: copy to next GPU
        for (int j = 1; j < nranks - 1; ++j) {
          rankDest = ringRanks[nranks - j];
          offset = dataOffset + rankDest * count;
          prims.directRecvCopyDirectSend(offset, offset, nelem);
        }

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_EXIT)
        if (tid == 0 && nranks > 2) {
          NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_RECV_COPY_SEND_EXIT, nelem*(nranks-2)*sizeof(T), prims.npKitDataProcessTotalTime, NPKIT_GET_GPU_TIMESTAMP(),
              ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        }
#endif

        // Make final copy from buffer to dest.
        rankDest = ringRanks[1];
        offset = dataOffset + rankDest * count;

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_ENTRY)
        if (tid == 0) {
          NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_ENTRY, nelem*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
              ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
          prims.npKitDataProcessTotalTime = 0;
        }
#endif
        // Final wait/copy.
        prims.directRecv(offset, nelem);
  
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_EXIT)
        if (tid == 0) {
          NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_DIRECT_RECV_EXIT, nelem*sizeof(T), prims.npKitDataProcessTotalTime, NPKIT_GET_GPU_TIMESTAMP(),
              ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
        }
#endif



      }
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_ALL_GATHER_RING_EXIT)
      if (tid == 0) {
        NpKit::CollectGpuEvent(NPKIT_EVENT_ALL_GATHER_RING_EXIT, count*sizeof(T), 0, NPKIT_GET_GPU_TIMESTAMP(),
            ncclShmem.comm.npKitEventCollectContexts + npKitCtxIdx);
      }
#endif
    } else if (inputBuf != outputBuf + ringRanks[0] * count) {
      inputBuf = inputBuf + partOffset;
      outputBuf = outputBuf + partOffset + ringRanks[0] * count;
      reduceCopy<1, 0, FuncSum<uint8_t>, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs=*/0>
        (tid - workNthreads, nthreads - workNthreads, work->redOpArg, &work->redOpArg, false, 1, (void**)&inputBuf, 1, (void**)&outputBuf, partCount);
    }
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
    // we have to wait for all warps before we can proceed to the next work;
    // otherwise, we can have contention if next work will use the outputBuf
    // in this work. We use bar 14 to avoid conflicts with prims barrier and
    // __syncthread().
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
  int nthreads = work->nWarps*WARP_SIZE;
  runRing<uint8_t>(tid, nthreads, work);
  //runRing<uint8_t, FuncSum<uint8_t>>(tid, nthreads, work);
}
