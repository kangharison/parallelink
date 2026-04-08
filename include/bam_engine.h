#ifndef BAM_ENGINE_H
#define BAM_ENGINE_H

#include <stdint.h>

#define BAM_MAX_QUEUE_DEPTH 1024
#define BAM_MAX_QPS         64
#define BAM_STATE_MAGIC     0xBA3D5A7E
#define BAM_STATE_PATH      "/tmp/bam_fio_state"

/* I/O direction */
#define BAM_OP_READ         0x02
#define BAM_OP_WRITE        0x01

/* GPU→CPU completion entry */
struct bam_completion {
	uint32_t tag;
	int32_t  status;
};

/* CPU-GPU shared state (resides in managed/mapped memory) */
struct bam_shared_state {
	/* GPU kernel control */
	volatile int      shutdown;
	volatile uint64_t done_count;

	/* Workload parameters (set by CPU once at init) */
	uint8_t  opcode;          /* BAM_OP_READ or BAM_OP_WRITE */
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
 * Called from the fio engine (engine.c) on the CPU side.
 */

/* Initialize GPU resources and NVMe controller via libnvm */
int bam_gpu_init(struct bam_shared_state **state,
		 int gpu_id, const char *nvme_dev,
		 int n_queues, int queue_depth);

/* Launch the persistent GPU kernel */
int bam_gpu_launch(struct bam_shared_state *state,
		   int gpu_warps, int n_queues);

/* Stop the persistent kernel and release GPU resources */
void bam_gpu_shutdown(struct bam_shared_state *state);

#ifdef __cplusplus
}
#endif

#endif /* BAM_ENGINE_H */
