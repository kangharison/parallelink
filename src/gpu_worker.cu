/*
 * parallelink: GPU-side persistent kernel + host bring-up
 *
 * Initializes libnvm (BaM) controller, allocates a GPU-resident page cache
 * (data buffers + PRP lists) and launches a persistent CUDA kernel that
 * submits/completes NVMe I/O directly from the GPU. CPU only polls
 * done_count for fio statistics.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <stdexcept>

#include "gpu_engine.h"

/* libnvm (BaM) C++ wrappers */
#include <ctrl.h>
#include <buffer.h>
#include <queue.h>
#include <page_cache.h>
#include <nvm_parallel_queue.h>
#include <nvm_cmd.h>
#include <nvm_io.h>

/* ------------------------------------------------------------------ */
/*  Host-side context kept alive across init → launch → shutdown      */
/* ------------------------------------------------------------------ */
struct plink_gpu_ctx {
	Controller      *ctrl;
	page_cache_t    *pc;
	page_cache_d_t  *d_pc;
	Controller     **d_ctrls;
	cudaStream_t     stream;
	int              n_queues;
	int              gpu_warps;
	int              total_threads;
};

/* One fio job → one engine instance → one context. fio runs a single
 * init()/cleanup() per job-thread, so keeping a file-scope context is
 * safe for the common single-job setup. */
static plink_gpu_ctx g_ctx = {};

/* ------------------------------------------------------------------ */
/*  Persistent I/O kernel                                             */
/* ------------------------------------------------------------------ */
/*
 * Each thread picks a queue-pair and a page-cache slot and issues
 * ios_per_thread NVMe commands via BaM's read_data/write_data device
 * helpers. Those helpers internally build the command, set PRPs from
 * pc->prp1/prp2, do sq_enqueue + cq_poll + cq_dequeue, and release cid.
 */
__global__ void plink_io_worker(struct plink_shared_state *state,
				Controller **ctrls,
				page_cache_d_t *pc,
				int n_queues)
{
	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= state->total_threads)
		return;

	int q_idx = (tid / 32) % n_queues;
	QueuePair *qp = &(ctrls[0]->d_qps[q_idx]);

	/* Each thread owns one distinct page-cache slot for its in-flight I/O. */
	uint64_t pc_entry = (uint64_t)tid % pc->n_pages;

	/* I/O granularity in LBAs. n_blocks is in 512B LBAs from the host side
	 * but BaM read_data/write_data expect it in device block units.  */
	uint32_t lba_shift = qp->block_size_log;
	uint64_t n_blocks_dev = ((uint64_t)state->n_blocks * state->block_size) >> lba_shift;
	if (n_blocks_dev == 0)
		n_blocks_dev = 1;

	uint64_t lba_max = state->lba_range; /* in 512B LBAs */

	uint64_t ios_done = 0;
	uint64_t seed = tid * 6364136223846793005ULL + 1;

	while (!state->shutdown) {

		/* Pick LBA in device block units */
		uint64_t start_block;
		if (state->random) {
			seed ^= seed << 13;
			seed ^= seed >> 7;
			seed ^= seed << 17;
			start_block = seed % lba_max;
		} else {
			start_block = (tid * state->ios_per_thread + ios_done)
				      * state->n_blocks;
			if (lba_max)
				start_block %= lba_max;
		}

		uint64_t t_start = clock64();

		if (state->opcode == PLINK_OP_READ)
			read_data(pc, qp, start_block, n_blocks_dev, pc_entry);
		else
			write_data(pc, qp, start_block, n_blocks_dev, pc_entry);

		uint64_t t_end = clock64();

		if (state->record_lat && state->latencies)
			state->latencies[tid] = t_end - t_start;

		atomicAdd((unsigned long long *)&state->done_count, 1ULL);
		ios_done++;
	}
}

/* ------------------------------------------------------------------ */
/*  Host-side API exposed to gpu_engine.c                             */
/* ------------------------------------------------------------------ */

extern "C" int plink_gpu_init(struct plink_shared_state **state_out,
			      int gpu_id, const char *nvme_dev,
			      int n_queues, int queue_depth)
{
	cudaError_t err;

	err = cudaSetDevice(gpu_id);
	if (err != cudaSuccess) {
		fprintf(stderr, "parallelink: cudaSetDevice(%d) failed: %s\n",
			gpu_id, cudaGetErrorString(err));
		return -1;
	}

	/* Shared state lives in managed memory for cheap CPU polling. */
	err = cudaMallocManaged(state_out, sizeof(struct plink_shared_state));
	if (err != cudaSuccess) {
		fprintf(stderr, "parallelink: cudaMallocManaged failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}
	memset(*state_out, 0, sizeof(struct plink_shared_state));

	try {
		/* namespace 1 is the default for most NVMe drives. */
		g_ctx.ctrl = new Controller(nvme_dev, 1, gpu_id,
					    (uint64_t)queue_depth,
					    (uint64_t)n_queues);
	} catch (const std::exception &e) {
		fprintf(stderr, "parallelink: Controller init failed: %s\n"
			"  (check that libnvm.ko is loaded and %s is bound to libnvm)\n",
			e.what(), nvme_dev);
		cudaFree(*state_out);
		*state_out = nullptr;
		return -1;
	}

	/* Page cache: one 4K page per GPU thread's max concurrency.
	 * Sized generously so tid % n_pages maps each thread to a distinct
	 * slot even for large warp counts. */
	const uint64_t page_size = 4096ULL;
	const uint64_t n_pages   = 8192ULL;

	std::vector<Controller *> ctrls_vec;
	ctrls_vec.push_back(g_ctx.ctrl);

	try {
		g_ctx.pc = new page_cache_t(page_size, n_pages, gpu_id,
					    *g_ctx.ctrl, (uint64_t)64,
					    ctrls_vec);
	} catch (const std::exception &e) {
		fprintf(stderr, "parallelink: page_cache_t init failed: %s\n",
			e.what());
		delete g_ctx.ctrl;
		g_ctx.ctrl = nullptr;
		cudaFree(*state_out);
		*state_out = nullptr;
		return -1;
	}

	g_ctx.d_pc     = (page_cache_d_t *)g_ctx.pc->d_pc_ptr;
	g_ctx.d_ctrls  = g_ctx.pc->pdt.d_ctrls;
	g_ctx.n_queues = n_queues;

	err = cudaStreamCreateWithFlags(&g_ctx.stream, cudaStreamNonBlocking);
	if (err != cudaSuccess) {
		fprintf(stderr, "parallelink: cudaStreamCreate failed: %s\n",
			cudaGetErrorString(err));
		delete g_ctx.pc;   g_ctx.pc   = nullptr;
		delete g_ctx.ctrl; g_ctx.ctrl = nullptr;
		cudaFree(*state_out);
		*state_out = nullptr;
		return -1;
	}

	return 0;
}

extern "C" int plink_gpu_launch(struct plink_shared_state *state,
				int gpu_warps, int n_queues)
{
	g_ctx.gpu_warps     = gpu_warps;
	g_ctx.total_threads = gpu_warps * 32;

	/* Allocate per-thread latency buffer if requested. */
	if (state->record_lat && !state->latencies) {
		cudaError_t err = cudaMallocManaged(
			&state->latencies,
			sizeof(uint64_t) * g_ctx.total_threads);
		if (err != cudaSuccess) {
			fprintf(stderr,
				"parallelink: latency buffer alloc failed: %s\n",
				cudaGetErrorString(err));
			return -1;
		}
	}

	int threads_per_block = 128;
	int blocks = (g_ctx.total_threads + threads_per_block - 1)
		     / threads_per_block;

	plink_io_worker<<<blocks, threads_per_block, 0, g_ctx.stream>>>(
		state, g_ctx.d_ctrls, g_ctx.d_pc, n_queues);

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
	__sync_synchronize();

	if (g_ctx.stream) {
		cudaStreamSynchronize(g_ctx.stream);
		cudaStreamDestroy(g_ctx.stream);
		g_ctx.stream = 0;
	}

	if (g_ctx.pc) {
		delete g_ctx.pc;
		g_ctx.pc = nullptr;
	}
	if (g_ctx.ctrl) {
		delete g_ctx.ctrl;
		g_ctx.ctrl = nullptr;
	}

	if (state->latencies) {
		cudaFree(state->latencies);
		state->latencies = nullptr;
	}
	cudaFree(state);
}
