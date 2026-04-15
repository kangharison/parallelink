/*
 * parallelink: GPU-side persistent kernel + host bring-up
 *
 * Initializes libnvm (BaM) controller, allocates a GPU-resident page cache
 * (data buffers + PRP lists) and launches a persistent CUDA kernel that
 * submits/completes NVMe I/O directly from the GPU. CPU only polls a
 * mirrored done counter for fio statistics.
 *
 * Memory layout (post-P0 refactor — no managed memory on the hot path):
 *
 *   - Workload parameters (`struct plink_workload`) are passed BY VALUE as
 *     a kernel launch argument → parameter buffer → registers. Every field
 *     the I/O loop reads (opcode, ios_per_thread, n_blocks, lba_range,
 *     random, record_lat, latencies*) is in-register; zero memory traffic
 *     for workload config during the loop.
 *
 *   - The done counter (`d_done_count`) is pure device memory
 *     (`cudaMalloc`). The GPU atomicAdds to it at full device-memory
 *     speed. The CPU never touches this pointer directly; it pulls a
 *     mirror via `cudaMemcpyAsync` on a side stream (plink_gpu_poll_done).
 *
 *   - The CPU→GPU shutdown signal lives in a pinned+mapped host
 *     allocation (`cudaHostAlloc(cudaHostAllocMapped)`). The CPU writes
 *     `h_ctrl->shutdown = 1` as a plain host store; the GPU reads
 *     `d_ctrl->shutdown` which maps to the same physical page via PCIe —
 *     no migration, no page-fault storm.
 *
 * This replaces the previous cudaMallocManaged `plink_shared_state` which
 * placed hot workload fields and the done counter on the same unified
 * page as the shutdown flag, causing a page-migration ping-pong between
 * the CPU poller and the GPU kernel that slowed throughput to a crawl
 * and could hang the kernel entirely on non-ATS systems.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <pthread.h>
#include <unistd.h>

#include "gpu_engine.h"

/* libnvm (BaM) C++ wrappers */
#include <ctrl.h>
#include <buffer.h>
#include <queue.h>
#include <page_cache.h>
#include <nvm_parallel_queue.h>
#include <nvm_cmd.h>
#include <nvm_io.h>
#include <nvm_rpc.h>
#include <nvm_dma.h>
#include <nvm_error.h>

/* ------------------------------------------------------------------ */
/*  Host-side context kept alive across init → launch → shutdown      */
/* ------------------------------------------------------------------ */
struct plink_gpu_ctx {
	Controller      *ctrl;
	page_cache_t    *pc;
	page_cache_d_t  *d_pc;
	Controller     **d_ctrls;

	cudaStream_t     compute_stream;  /* runs the persistent kernel */
	cudaStream_t     copy_stream;     /* D2H done-counter mirror */

	struct plink_ctrl_block *h_ctrl;  /* pinned mapped, host view */
	struct plink_ctrl_block *d_ctrl;  /* device ptr into the same pinned page */

	uint64_t        *d_done_count;    /* pure device memory */

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
 *
 * Launch bounds target the same SM-level occupancy as BaM's block
 * benchmark (2048 resident threads/SM) but with a 128-thread block
 * shape: 128 × 16 = 2048. The absolute occupancy target matters more
 * than block shape here because BaM's queue primitives spin on
 * device-wide atomics with __nanosleep backoff — the more warps
 * resident per SM, the better the SM can hide that latency. 128 also
 * lines up cleanly with the (tid/32) % n_queues warp-to-queue mapping.
 */
__global__ __launch_bounds__(128, 16)
void plink_io_worker(struct plink_ctrl_block *ctrl,
		     uint64_t *d_done_count,
		     struct plink_workload wl,
		     Controller **ctrls,
		     page_cache_d_t *pc,
		     int n_queues)
{
	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= (uint64_t)wl.total_threads)
		return;

	int q_idx = (tid / 32) % n_queues;
	QueuePair *qp = &(ctrls[0]->d_qps[q_idx]);

	/* Each thread owns one distinct page-cache slot for its in-flight I/O. */
	uint64_t pc_entry = (uint64_t)tid % pc->n_pages;

	/* I/O granularity in LBAs. wl.n_blocks is in 512B LBAs from the host side
	 * but BaM read_data/write_data expect it in device block units. */
	uint32_t lba_shift = qp->block_size_log;
	uint64_t n_blocks_dev = ((uint64_t)wl.n_blocks * 512ULL) >> lba_shift;
	if (n_blocks_dev == 0)
		n_blocks_dev = 1;

	uint64_t lba_max = wl.lba_range;

	uint64_t ios_done = 0;
	uint64_t seed = tid * 6364136223846793005ULL + 1;

	/*
	 * The kernel runs until the CPU signals shutdown. fio's keep_running()
	 * decides when the job ends and then plink_gpu_shutdown() writes the
	 * pinned ctrl flag. Each shutdown read is a PCIe round-trip to host
	 * memory, so we amortize by polling once every 64 I/Os — at BaM I/O
	 * rates the added latency is well under 10ns/iter.
	 */
	while (true) {
		if ((ios_done & 63) == 0) {
			if (ctrl->shutdown)
				break;
		}

		uint64_t lba_512;
		if (wl.random) {
			seed ^= seed << 13;
			seed ^= seed >> 7;
			seed ^= seed << 17;
			lba_512 = seed % lba_max;
		} else {
			lba_512 = (tid * wl.ios_per_thread + ios_done)
				  * wl.n_blocks;
			if (lba_max)
				lba_512 %= lba_max;
		}
		uint64_t start_block = (lba_512 * 512ULL) >> lba_shift;

		uint64_t t_start = clock64();

		if (wl.opcode == PLINK_OP_READ)
			read_data(pc, qp, start_block, n_blocks_dev, pc_entry);
		else
			write_data(pc, qp, start_block, n_blocks_dev, pc_entry);

		uint64_t t_end = clock64();

		/*
		 * Explicit warp reconverge. Empirically required to avoid a hang
		 * on Volta+; kept defensively until P0 is verified to render it
		 * unnecessary on its own.
		 */
		__syncwarp();

		if (wl.record_lat && wl.latencies)
			wl.latencies[tid] = t_end - t_start;

		atomicAdd((unsigned long long *)d_done_count, 1ULL);
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

	struct plink_shared_state *state =
		(struct plink_shared_state *)calloc(1, sizeof(*state));
	if (!state) {
		fprintf(stderr, "parallelink: shared-state alloc failed\n");
		return -1;
	}

	/* Pinned+mapped ctrl block for CPU→GPU shutdown signalling. */
	err = cudaHostAlloc(&g_ctx.h_ctrl, sizeof(struct plink_ctrl_block),
			    cudaHostAllocMapped | cudaHostAllocWriteCombined);
	if (err != cudaSuccess) {
		fprintf(stderr, "parallelink: cudaHostAlloc(ctrl) failed: %s\n",
			cudaGetErrorString(err));
		free(state);
		return -1;
	}
	g_ctx.h_ctrl->shutdown = 0;

	err = cudaHostGetDevicePointer((void **)&g_ctx.d_ctrl, g_ctx.h_ctrl, 0);
	if (err != cudaSuccess) {
		fprintf(stderr, "parallelink: cudaHostGetDevicePointer failed: %s\n",
			cudaGetErrorString(err));
		cudaFreeHost(g_ctx.h_ctrl);
		g_ctx.h_ctrl = nullptr;
		free(state);
		return -1;
	}

	/* Pure device memory for the done counter — GPU atomicAdds at
	 * device-memory speed; CPU never touches this pointer directly. */
	err = cudaMalloc(&g_ctx.d_done_count, sizeof(uint64_t));
	if (err != cudaSuccess) {
		fprintf(stderr, "parallelink: cudaMalloc(d_done_count) failed: %s\n",
			cudaGetErrorString(err));
		cudaFreeHost(g_ctx.h_ctrl);
		g_ctx.h_ctrl = nullptr;
		free(state);
		return -1;
	}
	cudaMemset(g_ctx.d_done_count, 0, sizeof(uint64_t));

	try {
		/* namespace 1 is the default for most NVMe drives. */
		g_ctx.ctrl = new Controller(nvme_dev, 1, gpu_id,
					    (uint64_t)queue_depth,
					    (uint64_t)n_queues);
	} catch (const std::exception &e) {
		fprintf(stderr, "parallelink: Controller init failed: %s\n"
			"  (check that libnvm.ko is loaded and %s is bound to libnvm)\n",
			e.what(), nvme_dev);
		cudaFree(g_ctx.d_done_count); g_ctx.d_done_count = nullptr;
		cudaFreeHost(g_ctx.h_ctrl);   g_ctx.h_ctrl        = nullptr;
		free(state);
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
		delete g_ctx.ctrl;            g_ctx.ctrl          = nullptr;
		cudaFree(g_ctx.d_done_count); g_ctx.d_done_count  = nullptr;
		cudaFreeHost(g_ctx.h_ctrl);   g_ctx.h_ctrl        = nullptr;
		free(state);
		return -1;
	}

	g_ctx.d_pc     = (page_cache_d_t *)g_ctx.pc->d_pc_ptr;
	g_ctx.d_ctrls  = g_ctx.pc->pdt.d_ctrls;
	g_ctx.n_queues = n_queues;

	err = cudaStreamCreateWithFlags(&g_ctx.compute_stream,
					cudaStreamNonBlocking);
	if (err != cudaSuccess) {
		fprintf(stderr, "parallelink: cudaStreamCreate(compute) failed: %s\n",
			cudaGetErrorString(err));
		goto fail_streams;
	}
	err = cudaStreamCreateWithFlags(&g_ctx.copy_stream,
					cudaStreamNonBlocking);
	if (err != cudaSuccess) {
		fprintf(stderr, "parallelink: cudaStreamCreate(copy) failed: %s\n",
			cudaGetErrorString(err));
		cudaStreamDestroy(g_ctx.compute_stream);
		g_ctx.compute_stream = 0;
		goto fail_streams;
	}

	state->h_ctrl      = g_ctx.h_ctrl;
	state->done_mirror = 0;
	*state_out         = state;
	return 0;

fail_streams:
	delete g_ctx.pc;              g_ctx.pc            = nullptr;
	delete g_ctx.ctrl;            g_ctx.ctrl          = nullptr;
	cudaFree(g_ctx.d_done_count); g_ctx.d_done_count  = nullptr;
	cudaFreeHost(g_ctx.h_ctrl);   g_ctx.h_ctrl        = nullptr;
	free(state);
	return -1;
}

extern "C" int plink_gpu_launch(struct plink_shared_state *state,
				const struct plink_workload *wl_in,
				int gpu_warps, int n_queues)
{
	(void)state;

	g_ctx.gpu_warps     = gpu_warps;
	g_ctx.total_threads = gpu_warps * 32;

	/* Make a local, mutable copy so we can fix up latencies* if needed. */
	struct plink_workload wl = *wl_in;
	wl.total_threads = g_ctx.total_threads;

	/* Allocate per-thread latency buffer in pure device memory if
	 * requested. Previously this was cudaMallocManaged, which reintroduced
	 * the same unified-memory thrash we just eliminated for shared state. */
	if (wl.record_lat && !wl.latencies) {
		uint64_t *d_lat = nullptr;
		cudaError_t err = cudaMalloc(&d_lat,
			sizeof(uint64_t) * g_ctx.total_threads);
		if (err != cudaSuccess) {
			fprintf(stderr,
				"parallelink: latency buffer alloc failed: %s\n",
				cudaGetErrorString(err));
			return -1;
		}
		cudaMemset(d_lat, 0,
			sizeof(uint64_t) * g_ctx.total_threads);
		wl.latencies = d_lat;
	}

	const int threads_per_block = 128;
	int blocks = (g_ctx.total_threads + threads_per_block - 1)
		     / threads_per_block;

	plink_io_worker<<<blocks, threads_per_block, 0, g_ctx.compute_stream>>>(
		g_ctx.d_ctrl, g_ctx.d_done_count, wl,
		g_ctx.d_ctrls, g_ctx.d_pc, n_queues);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "parallelink: kernel launch failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}

	return 0;
}

extern "C" int plink_gpu_poll_done(struct plink_shared_state *state)
{
	if (!state || !g_ctx.d_done_count)
		return -1;

	uint64_t mirror = 0;
	cudaError_t err = cudaMemcpyAsync(&mirror, g_ctx.d_done_count,
					  sizeof(uint64_t),
					  cudaMemcpyDeviceToHost,
					  g_ctx.copy_stream);
	if (err != cudaSuccess)
		return -1;

	err = cudaStreamSynchronize(g_ctx.copy_stream);
	if (err != cudaSuccess)
		return -1;

	state->done_mirror = mirror;
	return 0;
}

/* ------------------------------------------------------------------ */
/*  Admin command injection bridge                                    */
/*                                                                    */
/*  A single dedicated host DMA page mapped into the controller. The  */
/*  admin helper thread in gpu_engine.c forwards socket requests here */
/*  and we serialize with an internal mutex on top of BaM's own lock  */
/*  inside nvm_raw_rpc(), since it costs nothing and keeps the admin  */
/*  path independent of any future callers.                           */
/* ------------------------------------------------------------------ */
static struct {
	nvm_dma_t      *dma;
	void           *buf;
	pthread_mutex_t lock;
	bool            initialized;
} g_admin = { nullptr, nullptr, PTHREAD_MUTEX_INITIALIZER, false };

extern "C" int plink_admin_init(void)
{
	if (g_admin.initialized)
		return 0;
	if (!g_ctx.ctrl || !g_ctx.ctrl->ctrl || !g_ctx.ctrl->aq_ref) {
		fprintf(stderr,
			"parallelink: admin_init before controller ready\n");
		return -1;
	}

	long ps = sysconf(_SC_PAGESIZE);
	if (ps <= 0)
		ps = 4096;

	void *buf = nullptr;
	if (posix_memalign(&buf, (size_t)ps, PLINK_ADMIN_MAX_DATA) != 0)
		return -1;
	memset(buf, 0, PLINK_ADMIN_MAX_DATA);

	int status = nvm_dma_map_host(&g_admin.dma, g_ctx.ctrl->ctrl,
				      buf, PLINK_ADMIN_MAX_DATA);
	if (!nvm_ok(status)) {
		fprintf(stderr,
			"parallelink: nvm_dma_map_host(admin) failed: %s\n",
			nvm_strerror(status));
		free(buf);
		return -1;
	}

	g_admin.buf         = buf;
	g_admin.initialized = true;
	return 0;
}

extern "C" int plink_admin_rpc(const void *cmd_in, void *cpl_out,
			       void *data, uint32_t data_len, int direction)
{
	if (!g_admin.initialized)
		return EINVAL;
	if (data_len > PLINK_ADMIN_MAX_DATA)
		return E2BIG;
	if (!cmd_in || !cpl_out)
		return EINVAL;

	pthread_mutex_lock(&g_admin.lock);

	nvm_cmd_t cmd;
	nvm_cpl_t cpl;
	memcpy(&cmd, cmd_in, sizeof(cmd));
	memset(&cpl, 0, sizeof(cpl));

	/*
	 * Force PRP1 to our admin buffer regardless of what the client
	 * filled in. Clients have no idea what bus address our dma buf
	 * lives at; keeping this authoritative on the server side is
	 * both simpler and safer.
	 */
	nvm_cmd_data_ptr(&cmd,
			 g_admin.dma ? g_admin.dma->ioaddrs[0] : 0,
			 0);

	if (direction == 1 && data && data_len)
		memcpy(g_admin.buf, data, data_len);
	else
		memset(g_admin.buf, 0, PLINK_ADMIN_MAX_DATA);

	int rc = nvm_raw_rpc(g_ctx.ctrl->aq_ref, &cmd, &cpl);

	memcpy(cpl_out, &cpl, sizeof(cpl));

	if (rc == 0 && direction == 2 && data && data_len)
		memcpy(data, g_admin.buf, data_len);

	pthread_mutex_unlock(&g_admin.lock);
	return rc;
}

extern "C" void plink_admin_teardown(void)
{
	if (!g_admin.initialized)
		return;
	if (g_admin.dma) {
		nvm_dma_unmap(g_admin.dma);
		g_admin.dma = nullptr;
	}
	if (g_admin.buf) {
		free(g_admin.buf);
		g_admin.buf = nullptr;
	}
	g_admin.initialized = false;
}

extern "C" void plink_gpu_shutdown(struct plink_shared_state *state)
{
	if (!state)
		return;

	/* Signal the kernel to stop. Single host store to pinned memory;
	 * the GPU picks it up on its next periodic check. */
	if (g_ctx.h_ctrl)
		g_ctx.h_ctrl->shutdown = 1;

	if (g_ctx.compute_stream) {
		cudaStreamSynchronize(g_ctx.compute_stream);
		cudaStreamDestroy(g_ctx.compute_stream);
		g_ctx.compute_stream = 0;
	}
	if (g_ctx.copy_stream) {
		cudaStreamDestroy(g_ctx.copy_stream);
		g_ctx.copy_stream = 0;
	}

	if (g_ctx.pc) {
		delete g_ctx.pc;
		g_ctx.pc = nullptr;
	}
	if (g_ctx.ctrl) {
		delete g_ctx.ctrl;
		g_ctx.ctrl = nullptr;
	}

	if (g_ctx.d_done_count) {
		cudaFree(g_ctx.d_done_count);
		g_ctx.d_done_count = nullptr;
	}
	if (g_ctx.h_ctrl) {
		cudaFreeHost(g_ctx.h_ctrl);
		g_ctx.h_ctrl = nullptr;
		g_ctx.d_ctrl = nullptr;
	}

	free(state);
}
