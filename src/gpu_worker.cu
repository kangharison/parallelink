/*
 * parallelink: GPU-side persistent kernel
 *
 * Each GPU thread autonomously:
 *   1. Builds an NVMe command
 *   2. Submits to SQ via sq_enqueue()
 *   3. Polls CQ for completion via cq_poll()
 *   4. Immediately submits the next I/O (no CPU round-trip)
 *
 * CPU only sets workload params and polls done_count.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

#include "plink_engine.h"

/* libnvm headers (from extern/bam) */
#include "nvm_parallel_queue.h"
#include "nvm_cmd.h"
#include "ctrl.h"
#include "queue.h"
#include "buffer.h"

/* ------------------------------------------------------------------ */
/*  LBA generation helpers                                            */
/* ------------------------------------------------------------------ */

__device__ static uint64_t next_lba_random(uint64_t lba_range,
					   uint64_t seed)
{
	/* Simple xorshift64 PRNG */
	seed ^= seed << 13;
	seed ^= seed >> 7;
	seed ^= seed << 17;
	return seed % lba_range;
}

__device__ static uint64_t next_lba_sequential(uint64_t lba_range,
					       uint32_t n_blocks,
					       uint64_t ios_done)
{
	return (ios_done * n_blocks) % lba_range;
}

/* ------------------------------------------------------------------ */
/*  Persistent I/O kernel                                             */
/* ------------------------------------------------------------------ */

__global__ void plink_io_worker(struct plink_shared_state *state,
				QueuePair *qps, int n_queues)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= state->total_threads)
		return;

	int qp_idx = tid % n_queues;
	QueuePair *my_qp = &qps[qp_idx];

	uint64_t ios_done = 0;
	uint64_t seed = (uint64_t)tid * 6364136223846793005ULL + 1;

	while (!state->shutdown && ios_done < state->ios_per_thread) {

		/* Determine target LBA */
		uint64_t lba;
		if (state->random) {
			seed = seed * 6364136223846793005ULL + 1;
			lba = next_lba_random(state->lba_range, seed);
		} else {
			lba = next_lba_sequential(state->lba_range,
						  state->n_blocks,
						  tid * state->ios_per_thread + ios_done);
		}

		/* Build NVMe command */
		nvm_cmd_t cmd;
		uint16_t cid = get_cid(&my_qp->sq);
		nvm_cmd_header(&cmd, cid, state->opcode, my_qp->nvmNamespace);

		/* TODO: set PRP addresses from GPU data buffer */
		/* nvm_cmd_data_ptr(&cmd, prp1, prp2); */
		nvm_cmd_rw_blks(&cmd, lba, state->n_blocks);

		/* Submit to SQ */
		uint64_t t_start = clock64();
		sq_enqueue(&my_qp->sq, &cmd);

		/* Poll CQ for completion */
		uint32_t head, head_;
		uint32_t cq_pos = cq_poll(&my_qp->cq, cid, &head, &head_);
		cq_dequeue(&my_qp->cq, cq_pos, &my_qp->sq, head, head_);
		put_cid(&my_qp->sq, cid);

		uint64_t t_end = clock64();

		/* Record latency if enabled */
		if (state->record_lat && state->latencies)
			state->latencies[tid] = t_end - t_start;

		/* Update global completion counter */
		atomicAdd((unsigned long long *)&state->done_count, 1);

		ios_done++;
	}
}

/* ------------------------------------------------------------------ */
/*  Host-side interface (called from gpu-engine.c)                    */
/* ------------------------------------------------------------------ */

extern "C" int plink_gpu_init(struct plink_shared_state **state,
			      int gpu_id, const char *nvme_dev,
			      int n_queues, int queue_depth)
{
	cudaError_t err = cudaSetDevice(gpu_id);
	if (err != cudaSuccess) {
		fprintf(stderr, "parallelink: cudaSetDevice(%d) failed: %s\n",
			gpu_id, cudaGetErrorString(err));
		return -1;
	}

	/* Allocate shared state in managed memory (CPU+GPU accessible) */
	err = cudaMallocManaged(state, sizeof(struct plink_shared_state));
	if (err != cudaSuccess) {
		fprintf(stderr, "parallelink: cudaMallocManaged failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}
	memset(*state, 0, sizeof(struct plink_shared_state));

	/*
	 * TODO: Full initialization sequence:
	 *   1. open(nvme_dev) → fd
	 *   2. nvm_ctrl_init(&ctrl, fd) → BAR0 mmap
	 *   3. Create admin queue, identify controller/namespace
	 *   4. cudaMalloc GPU data buffers
	 *   5. nvm_dma_map_device() → GPU physical addresses for PRP
	 *   6. Create I/O QueuePairs (SQ/CQ in GPU VRAM)
	 *   7. Store QueuePair pointers for kernel launch
	 */

	return 0;
}

extern "C" int plink_gpu_launch(struct plink_shared_state *state,
				int gpu_warps, int n_queues)
{
	int threads_per_block = 128;
	int total_threads = gpu_warps * 32;
	int n_blocks = (total_threads + threads_per_block - 1) / threads_per_block;

	/*
	 * TODO: pass actual QueuePair device pointer
	 * plink_io_worker<<<n_blocks, threads_per_block>>>(state, d_qps, n_queues);
	 */

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "parallelink: kernel launch failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}

	return 0;
}

extern "C" void plink_gpu_shutdown(struct plink_shared_state *state)
{
	if (!state)
		return;

	state->shutdown = 1;
	cudaDeviceSynchronize();

	if (state->latencies)
		cudaFree(state->latencies);
	cudaFree(state);
}
