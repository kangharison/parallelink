/*
 * parallelink: fio I/O engine for GPU-direct NVMe access
 *
 * This engine launches a persistent CUDA kernel that autonomously
 * submits and completes NVMe I/O commands via PCIe P2P, bypassing
 * the CPU from the I/O data path.
 *
 * The CPU side (this file) only:
 *   - Configures workload parameters at init
 *   - Launches the GPU kernel once
 *   - Polls done_count for statistics collection
 */

#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "config-host.h"
#include "fio.h"
#include "optgroup.h"

#include "gpu_engine.h"

struct plink_options {
	struct thread_data *td;
	unsigned int gpu_warps;
	unsigned int gpu_id;
	unsigned int n_queues;
	char        *nvme_dev;
};

struct plink_data {
	struct plink_shared_state *state;
	struct plink_workload      wl;
	uint64_t last_seen;   /* last polled done counter */

	/*
	 * io_u ring. The GPU engine doesn't actually use io_u's for I/O —
	 * the GPU autonomously issues commands. io_u's are kept here purely
	 * as accounting tokens so that fio's completion path
	 * (io_completed → td->bytes_done) runs and keep_running() doesn't
	 * terminate the job for "no progress".
	 */
	struct io_u **queued;
	unsigned int  ring_size;
	unsigned int  head;
	unsigned int  tail;
	unsigned int  nr;
};

/* ------------------------------------------------------------------ */
/*  Engine-specific fio options                                       */
/* ------------------------------------------------------------------ */

static struct fio_option options[] = {
	{
		.name     = "gpu_warps",
		.lname    = "Number of GPU warps",
		.type     = FIO_OPT_INT,
		.off1     = offsetof(struct plink_options, gpu_warps),
		.def      = "32",
		.help     = "Number of GPU warps for I/O submission",
		.category = FIO_OPT_C_ENGINE,
		.group    = FIO_OPT_G_INVALID,
	},
	{
		.name     = "gpu_id",
		.lname    = "GPU device ID",
		.type     = FIO_OPT_INT,
		.off1     = offsetof(struct plink_options, gpu_id),
		.def      = "0",
		.help     = "CUDA GPU device ID",
		.category = FIO_OPT_C_ENGINE,
		.group    = FIO_OPT_G_INVALID,
	},
	{
		.name     = "n_queues",
		.lname    = "Number of NVMe queue pairs",
		.type     = FIO_OPT_INT,
		.off1     = offsetof(struct plink_options, n_queues),
		.def      = "16",
		.help     = "Number of NVMe submission/completion queue pairs",
		.category = FIO_OPT_C_ENGINE,
		.group    = FIO_OPT_G_INVALID,
	},
	{
		.name     = "nvme_dev",
		.lname    = "libnvm device path",
		.type     = FIO_OPT_STR_STORE,
		.off1     = offsetof(struct plink_options, nvme_dev),
		.def      = "/dev/libnvm0",
		.help     = "libnvm character device path (requires libnvm module)",
		.category = FIO_OPT_C_ENGINE,
		.group    = FIO_OPT_G_INVALID,
	},
	{
		.name = NULL,
	},
};

/* ------------------------------------------------------------------ */
/*  Engine callbacks                                                  */
/* ------------------------------------------------------------------ */

static int fio_plink_init(struct thread_data *td)
{
	struct plink_options *o = td->eo;
	struct plink_data *pd;
	int ret;

	log_info("parallelink: build=%s\n", PLINK_BUILD_TYPE);

	/*
	 * Diskless engine: register a dummy file so fio's get_next_file()
	 * has something to hand back in io_u->file. Without this the I/O
	 * loop hits NULL and bails out of get_io_u() immediately.
	 */
	if (!td->files_index) {
		add_file(td, td->o.filename ? td->o.filename : "parallelink",
			 0, 1);
		if (!td->o.nr_files)
			td->o.nr_files = 1;
	}

	pd = calloc(1, sizeof(*pd));
	if (!pd)
		return -ENOMEM;

	pd->ring_size = td->o.iodepth ? td->o.iodepth : 256;
	pd->queued = calloc(pd->ring_size, sizeof(*pd->queued));
	if (!pd->queued) {
		free(pd);
		return -ENOMEM;
	}

	ret = plink_gpu_init(&pd->state, o->gpu_id, o->nvme_dev,
			     o->n_queues, 256);
	if (ret) {
		log_err("parallelink: GPU init failed (ret=%d). "
			"Is libnvm loaded? (insmod libnvm.ko)\n", ret);
		free(pd->queued);
		free(pd);
		return ret;
	}

	/*
	 * Workload parameters are now passed as kernel launch arguments
	 * (→ register/constant memory on the GPU) rather than dropped
	 * into a cudaMallocManaged struct the CPU keeps touching. Filling
	 * pd->wl here is purely a host-side setup; it is copied by value
	 * into the kernel call in plink_gpu_launch().
	 */
	pd->wl.block_size   = td->o.bs[DDIR_READ];
	pd->wl.n_blocks     = td->o.bs[DDIR_READ] / 512;
	pd->wl.opcode       = td_read(td) ? PLINK_OP_READ : PLINK_OP_WRITE;
	pd->wl.random       = td_random(td);
	pd->wl.lba_range    = td->o.size / 512;
	pd->wl.record_lat   = 0;
	pd->wl.latencies    = NULL;
	pd->last_seen       = 0;

	int total_threads       = o->gpu_warps * 32;
	uint64_t total_ios      = td->o.size / td->o.bs[DDIR_READ];

	pd->wl.total_threads  = total_threads;
	pd->wl.ios_per_thread = total_threads ? (total_ios / total_threads) : 0;

	/* Launch persistent GPU kernel */
	ret = plink_gpu_launch(pd->state, &pd->wl, o->gpu_warps, o->n_queues);
	if (ret) {
		log_err("parallelink: GPU kernel launch failed\n");
		plink_gpu_shutdown(pd->state);
		free(pd->queued);
		free(pd);
		return ret;
	}

	td->io_ops_data = pd;
	return 0;
}

/*
 * queue(): the GPU kernel autonomously generates and submits I/O, so
 * the io_u carries no real payload. We stash it in a ring so that
 * event() can hand it back to fio later for completion accounting.
 */
static enum fio_q_status fio_plink_queue(struct thread_data *td,
					 struct io_u *io_u)
{
	struct plink_data *pd = td->io_ops_data;

	if (pd->nr >= pd->ring_size)
		return FIO_Q_BUSY;

	pd->queued[pd->head] = io_u;
	pd->head = (pd->head + 1) % pd->ring_size;
	pd->nr++;

	return FIO_Q_QUEUED;
}

/* commit() is a no-op: GPU submits directly to NVMe SQ */
static int fio_plink_commit(struct thread_data *td)
{
	return 0;
}

/*
 * getevents(): poll the GPU's done_count to harvest completions.
 * This is how fio collects BW/IOPS/latency statistics from the
 * GPU-driven I/O loop.
 */
static int fio_plink_getevents(struct thread_data *td, unsigned int min,
			       unsigned int max, const struct timespec *t)
{
	struct plink_data *pd = td->io_ops_data;
	unsigned int events = 0;

	/*
	 * Clamp by ring occupancy: we can only "complete" as many io_u's
	 * back to fio as we currently have queued, regardless of how many
	 * I/Os the GPU has actually finished.
	 */
	if (max > pd->nr)
		max = pd->nr;
	if (min > max)
		min = max;

	while (events < min) {
		/*
		 * Pull a fresh mirror of the device-side done counter via a
		 * side stream. The kernel itself runs on a different stream
		 * and is unaffected. This replaces the previous managed-
		 * memory poll, which thrashed unified pages with the GPU.
		 */
		if (plink_gpu_poll_done(pd->state))
			break;

		uint64_t gpu_done = pd->state->done_mirror;
		uint64_t new_events = gpu_done - pd->last_seen;

		if (new_events > 0) {
			if (new_events > max)
				new_events = max;
			pd->last_seen += new_events;
			events = (unsigned int)new_events;
			break;
		}
		/* 100 µs backoff — fio stats resolution is ms, so this is
		 * plenty, and it keeps CPU-side pressure off the side stream. */
		usleep(100);
	}

	return (int)events;
}

static struct io_u *fio_plink_event(struct thread_data *td, int event)
{
	struct plink_data *pd = td->io_ops_data;
	struct io_u *io_u;

	(void)event;

	if (!pd->nr)
		return NULL;

	io_u = pd->queued[pd->tail];
	pd->queued[pd->tail] = NULL;
	pd->tail = (pd->tail + 1) % pd->ring_size;
	pd->nr--;

	io_u->error = 0;
	io_u->resid = 0;
	return io_u;
}

static void fio_plink_cleanup(struct thread_data *td)
{
	struct plink_data *pd = td->io_ops_data;

	if (pd) {
		plink_gpu_shutdown(pd->state);
		free(pd->queued);
		free(pd);
		td->io_ops_data = NULL;
	}
}

static int fio_plink_open_file(struct thread_data *td, struct fio_file *f)
{
	return 0;
}

static int fio_plink_close_file(struct thread_data *td, struct fio_file *f)
{
	return 0;
}

/* ------------------------------------------------------------------ */
/*  Engine registration                                               */
/* ------------------------------------------------------------------ */

/*
 * External engine: ioengine_ops must be non-static so that fio can
 * find it via dlsym() when loading the .so at runtime.
 *
 * fio tries these symbols in order (ioengines.c:dlopen_ioengine):
 *   1. dlsym(dlhandle, "<engine_name>")   → "parallelink"
 *   2. dlsym(dlhandle, "ioengine")
 *   3. dlsym(dlhandle, "get_ioengine")    → function call
 *
 * By naming the struct "ioengine" (non-static), method 2 will match.
 */
struct ioengine_ops ioengine = {
	.name               = "parallelink",
	.version            = FIO_IOOPS_VERSION,
	.flags              = FIO_NOEXTEND | FIO_NODISKUTIL | FIO_DISKLESSIO,
	.init               = fio_plink_init,
	.queue              = fio_plink_queue,
	.commit             = fio_plink_commit,
	.getevents          = fio_plink_getevents,
	.event              = fio_plink_event,
	.cleanup            = fio_plink_cleanup,
	.open_file          = fio_plink_open_file,
	.close_file         = fio_plink_close_file,
	.options            = options,
	.option_struct_size = sizeof(struct plink_options),
};
