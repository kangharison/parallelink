#ifndef GPU_ENGINE_H
#define GPU_ENGINE_H

#include <stdint.h>

#define PLINK_MAX_QUEUE_DEPTH 1024
#define PLINK_MAX_QPS         64
#define PLINK_STATE_MAGIC     0xBA3D5A7E
#define PLINK_STATE_PATH      "/tmp/plink_fio_state"

/* I/O direction (NVMe opcodes) */
#define PLINK_OP_READ         0x02
#define PLINK_OP_WRITE        0x01

/*
 * CPU→GPU control block. Lives in pinned+mapped host memory so that
 * CPU writes are plain stores to host RAM and GPU reads go straight
 * over PCIe without ever being cached on device — i.e. there is no
 * page migration and no unified-memory thrashing.
 *
 * This block is used ONLY for low-frequency control signalling
 * (currently just `shutdown`). Hot workload parameters are passed as
 * kernel arguments; the done counter lives in pure device memory.
 */
struct plink_ctrl_block {
	volatile uint32_t shutdown;
};

/*
 * Workload parameters. Passed by value as a kernel launch argument,
 * so the whole struct lands in the kernel's parameter buffer and
 * each field is fetched into a register on first use. No managed
 * memory, no PCIe round-trip per access inside the I/O loop.
 */
struct plink_workload {
	uint8_t  opcode;          /* PLINK_OP_READ or PLINK_OP_WRITE */
	int      random;          /* 1=random, 0=sequential */
	uint32_t block_size;      /* bytes per I/O */
	uint32_t n_blocks;        /* 512B blocks per I/O */
	uint64_t lba_range;       /* total 512B LBAs addressable */
	uint64_t ios_per_thread;  /* I/Os each GPU thread should issue */
	int      total_threads;   /* total GPU threads launched */
	int      record_lat;      /* whether to record per-I/O latency */
	uint64_t *latencies;      /* device pointer, may be NULL */
};

/*
 * Opaque shared state returned to the engine caller. Holds the host-
 * visible pieces: the pinned ctrl block (for shutdown signalling) and
 * a CPU-side mirror of the device done counter (updated by
 * plink_gpu_poll_done).
 */
struct plink_shared_state {
	struct plink_ctrl_block *h_ctrl;      /* pinned mapped, host ptr */
	uint64_t                 done_mirror; /* last D2H copy of d_done_count */
};

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize GPU resources and NVMe controller via libnvm */
int plink_gpu_init(struct plink_shared_state **state,
		   int gpu_id, const char *nvme_dev,
		   int n_queues, int queue_depth);

/* Launch the persistent GPU kernel with the given workload params */
int plink_gpu_launch(struct plink_shared_state *state,
		     const struct plink_workload *wl,
		     int gpu_warps, int n_queues);

/*
 * Pull the current device-side done counter into state->done_mirror
 * via a side stream. Non-blocking on the compute stream, returns 0 on
 * success. Called by the fio engine from getevents().
 */
int plink_gpu_poll_done(struct plink_shared_state *state);

/* Stop the persistent kernel and release GPU resources */
void plink_gpu_shutdown(struct plink_shared_state *state);

#ifdef __cplusplus
}
#endif

#endif /* GPU_ENGINE_H */
