/*
 * plink-holder: resident process that keeps GPU memory and NVMe state
 * alive across fio test runs.
 *
 * Without this, every fio run would require a full NVMe controller
 * reset + queue recreation + GPU DMA remapping (~200-1000ms).
 *
 * Usage:
 *   $ plink-holder --nvme=/dev/libnvm0 --gpu=0 --queues=32
 *   (runs in foreground, Ctrl-C to release resources)
 *
 * fio connects via CUDA IPC handles stored in PLINK_STATE_PATH.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <unistd.h>

#include "gpu_engine.h"

static volatile int running = 1;

static void sighandler(int sig)
{
	(void)sig;
	running = 0;
}

static void usage(const char *prog)
{
	fprintf(stderr, "Usage: %s [options]\n"
		"  --nvme=PATH    libnvm device (default: /dev/libnvm0)\n"
		"  --gpu=ID       CUDA device ID (default: 0)\n"
		"  --queues=N     number of NVMe queue pairs (default: 16)\n",
		prog);
}

int main(int argc, char **argv)
{
	const char *nvme_dev = "/dev/libnvm0";
	int gpu_id = 0;
	int n_queues = 16;

	for (int i = 1; i < argc; i++) {
		if (!strncmp(argv[i], "--nvme=", 7))
			nvme_dev = argv[i] + 7;
		else if (!strncmp(argv[i], "--gpu=", 6))
			gpu_id = atoi(argv[i] + 6);
		else if (!strncmp(argv[i], "--queues=", 9))
			n_queues = atoi(argv[i] + 9);
		else {
			usage(argv[0]);
			return 1;
		}
	}

	signal(SIGINT, sighandler);
	signal(SIGTERM, sighandler);

	cudaError_t err = cudaSetDevice(gpu_id);
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice(%d) failed: %s\n",
			gpu_id, cudaGetErrorString(err));
		return 1;
	}

	/*
	 * TODO: Full initialization:
	 *   1. open(nvme_dev) → nvm_ctrl_init()
	 *   2. cudaMalloc SQ/CQ/tickets/cid/head_mark per QP
	 *   3. nvm_dma_map_device() for each GPU allocation
	 *   4. nvm_admin_sq_create / cq_create
	 *   5. cudaIpcGetMemHandle() for each allocation
	 *   6. Save handles to PLINK_STATE_PATH
	 */

	fprintf(stderr, "plink-holder: ready (gpu=%d, nvme=%s, queues=%d)\n",
		gpu_id, nvme_dev, n_queues);
	fprintf(stderr, "plink-holder: holding GPU memory. Ctrl-C to release.\n");

	while (running)
		sleep(1);

	/*
	 * TODO: Cleanup:
	 *   1. nvm_admin_sq_delete / cq_delete
	 *   2. nvm_dma_unmap
	 *   3. cudaFree all allocations
	 *   4. nvm_ctrl_free
	 *   5. Remove PLINK_STATE_PATH
	 */

	fprintf(stderr, "plink-holder: shutting down\n");
	return 0;
}
