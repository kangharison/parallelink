/*
 * fio I/O engine for BAM (GPU-direct NVMe access)
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

#include "bam_engine.h"

struct bam_options {
	struct thread_data *td;
	unsigned int gpu_warps;
	unsigned int gpu_id;
	unsigned int n_queues;
	char        *nvme_dev;
};

struct bam_data {
	struct bam_shared_state *state;
	uint64_t last_seen;   /* last polled done_count */
};

/* ------------------------------------------------------------------ */
/*  Engine-specific fio options                                       */
/* ------------------------------------------------------------------ */

static struct fio_option options[] = {
	{
		.name     = "gpu_warps",
		.lname    = "Number of GPU warps",
		.type     = FIO_OPT_INT,
		.off1     = offsetof(struct bam_options, gpu_warps),
		.def      = "32",
		.help     = "Number of GPU warps for I/O submission",
		.category = FIO_OPT_C_ENGINE,
		.group    = FIO_OPT_G_INVALID,
	},
	{
		.name     = "gpu_id",
		.lname    = "GPU device ID",
		.type     = FIO_OPT_INT,
		.off1     = offsetof(struct bam_options, gpu_id),
		.def      = "0",
		.help     = "CUDA GPU device ID",
		.category = FIO_OPT_C_ENGINE,
		.group    = FIO_OPT_G_INVALID,
	},
	{
		.name     = "n_queues",
		.lname    = "Number of NVMe queue pairs",
		.type     = FIO_OPT_INT,
		.off1     = offsetof(struct bam_options, n_queues),
		.def      = "16",
		.help     = "Number of NVMe submission/completion queue pairs",
		.category = FIO_OPT_C_ENGINE,
		.group    = FIO_OPT_G_INVALID,
	},
	{
		.name     = "nvme_dev",
		.lname    = "libnvm device path",
		.type     = FIO_OPT_STR_STORE,
		.off1     = offsetof(struct bam_options, nvme_dev),
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

static int fio_bam_init(struct thread_data *td)
{
	struct bam_options *o = td->eo;
	struct bam_data *bd;
	int ret;

	bd = calloc(1, sizeof(*bd));
	if (!bd)
		return -ENOMEM;

	ret = bam_gpu_init(&bd->state, o->gpu_id, o->nvme_dev,
			   o->n_queues, td->o.iodepth);
	if (ret) {
		log_err("bam: GPU init failed (ret=%d). "
			"Is libnvm loaded? (insmod libnvm.ko)\n", ret);
		free(bd);
		return ret;
	}

	/* Configure workload parameters for GPU kernel */
	bd->state->block_size   = td->o.bs[DDIR_READ];
	bd->state->n_blocks     = td->o.bs[DDIR_READ] / 512;
	bd->state->opcode       = td_read(td) ? BAM_OP_READ : BAM_OP_WRITE;
	bd->state->random       = td_random(td);
	bd->state->lba_range    = td->o.size / 512;
	bd->state->done_count   = 0;
	bd->state->shutdown     = 0;
	bd->last_seen           = 0;

	/* Calculate per-thread I/O count */
	int total_threads       = o->gpu_warps * 32;
	uint64_t total_ios      = td->o.size / td->o.bs[DDIR_READ];

	bd->state->total_threads  = total_threads;
	bd->state->ios_per_thread = total_ios / total_threads;

	/* Launch persistent GPU kernel */
	ret = bam_gpu_launch(bd->state, o->gpu_warps, o->n_queues);
	if (ret) {
		log_err("bam: GPU kernel launch failed\n");
		bam_gpu_shutdown(bd->state);
		free(bd);
		return ret;
	}

	td->io_ops_data = bd;
	return 0;
}

/*
 * queue() is a no-op: the GPU kernel autonomously generates and
 * submits I/O. fio calls this per io_u, but the real work happens
 * on the GPU side.
 */
static enum fio_q_status fio_bam_queue(struct thread_data *td,
				       struct io_u *io_u)
{
	return FIO_Q_QUEUED;
}

/* commit() is a no-op: GPU submits directly to NVMe SQ */
static int fio_bam_commit(struct thread_data *td)
{
	return 0;
}

/*
 * getevents(): poll the GPU's done_count to harvest completions.
 * This is how fio collects BW/IOPS/latency statistics from the
 * GPU-driven I/O loop.
 */
static int fio_bam_getevents(struct thread_data *td, unsigned int min,
			     unsigned int max, const struct timespec *t)
{
	struct bam_data *bd = td->io_ops_data;
	int events = 0;

	while (events < (int)min) {
		uint64_t gpu_done = __atomic_load_n(
			&bd->state->done_count, __ATOMIC_ACQUIRE);
		uint64_t new_events = gpu_done - bd->last_seen;

		if (new_events > 0) {
			events = (new_events > max) ? max : (int)new_events;
			bd->last_seen += events;
			break;
		}
		usleep(1);
	}

	return events;
}

static struct io_u *fio_bam_event(struct thread_data *td, int event)
{
	/* Return a generic io_u — real I/O was done on GPU */
	return NULL;
}

static void fio_bam_cleanup(struct thread_data *td)
{
	struct bam_data *bd = td->io_ops_data;

	if (bd) {
		bam_gpu_shutdown(bd->state);
		free(bd);
		td->io_ops_data = NULL;
	}
}

static int fio_bam_open_file(struct thread_data *td, struct fio_file *f)
{
	return 0;
}

static int fio_bam_close_file(struct thread_data *td, struct fio_file *f)
{
	return 0;
}

/* ------------------------------------------------------------------ */
/*  Engine registration                                               */
/* ------------------------------------------------------------------ */

static struct ioengine_ops ioengine = {
	.name               = "bam",
	.version            = FIO_IOOPS_VERSION,
	.flags              = FIO_NOEXTEND | FIO_NODISKUTIL |
			      FIO_ASYNCIO_SETS_ISSUE_TIME,
	.init               = fio_bam_init,
	.queue              = fio_bam_queue,
	.commit             = fio_bam_commit,
	.getevents          = fio_bam_getevents,
	.event              = fio_bam_event,
	.cleanup            = fio_bam_cleanup,
	.open_file          = fio_bam_open_file,
	.close_file         = fio_bam_close_file,
	.options            = options,
	.option_struct_size = sizeof(struct bam_options),
};

static void fio_init fio_bam_register(void)
{
	register_ioengine(&ioengine);
}

static void fio_exit fio_bam_unregister(void)
{
	unregister_ioengine(&ioengine);
}
