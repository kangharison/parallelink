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

/* GPU→CPU completion entry */
struct plink_completion {
	uint32_t tag;
	int32_t  status;
};

/* CPU-GPU shared state (resides in managed/mapped memory) */
struct plink_shared_state {
	/* GPU kernel control */
	volatile int      shutdown;
	volatile uint64_t done_count;

	/* Workload parameters (set by CPU once at init) */
	uint8_t  opcode;          /* PLINK_OP_READ or PLINK_OP_WRITE */
	int      random;          /* 1=random, 0=sequential */
	uint32_t block_size;      /* bytes per I/O */
	uint32_t n_blocks;        /* blocks per I/O command */
	uint64_t lba_range;       /* total LBAs addressable */
	uint64_t ios_per_thread;  /* I/Os each GPU thread should issue */
	int      total_threads;   /* total GPU threads launched */
	int      record_lat;      /* whether to record per-I/O latency */

	/* Per-thread latency samples (GPU writes, CPU reads) */
	uint64_t *latencies;
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * GPU-side functions (implemented in gpu_worker.cu)
 * Called from the fio engine (gpu-engine.c) on the CPU side.
 */

/* Initialize GPU resources and NVMe controller via libnvm */
int plink_gpu_init(struct plink_shared_state **state,
		   int gpu_id, const char *nvme_dev,
		   int n_queues, int queue_depth);

/* Launch the persistent GPU kernel */
int plink_gpu_launch(struct plink_shared_state *state,
		     int gpu_warps, int n_queues);

/* Stop the persistent kernel and release GPU resources */
void plink_gpu_shutdown(struct plink_shared_state *state);

#ifdef __cplusplus
}
#endif

#endif /* GPU_ENGINE_H */
