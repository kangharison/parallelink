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
#include <pthread.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/types.h>
#include <signal.h>
#include <fcntl.h>
#include <stdint.h>

#include "config-host.h"
#include "fio.h"
#include "optgroup.h"

#include "gpu_engine.h"
#include "plink_admin_wire.h"

/* Wire protocol: see docs/design or the bam-admin-cli source. */
#define PLINK_ADMIN_CMD_LEN 64
#define PLINK_ADMIN_CPL_LEN 16

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

	/* Admin command injection helper thread. */
	pthread_t     admin_thr;
	int           admin_listen_fd;
	volatile int  admin_run;
	int           admin_enabled;
};

/* ------------------------------------------------------------------ */
/*  Build a raw NVMe SQE from a passthru cmd                          */
/* ------------------------------------------------------------------ */
/*
 * plink_admin_rpc() consumes a 64-byte nvm_cmd_t. PRP1/PRP2 are
 * overridden by the server from its pinned DMA buffer, so we only
 * need to fill the fields that actually drive the command:
 *   dw0  = cid(16) | psdt(2) | rsvd(4) | fuse(2) | opcode(8)
 *   dw1  = nsid
 *   dw2  = cdw2 (typically reserved)
 *   dw3  = cdw3
 *   dw10..dw15 = cdw10..cdw15
 * Metadata/PRP fields are zeroed.
 */
static void plink_build_sqe(uint8_t sqe[PLINK_ADMIN_CMD_LEN],
			    const struct plink_nvme_passthru_cmd *c)
{
	memset(sqe, 0, PLINK_ADMIN_CMD_LEN);
	uint32_t dw0 = ((uint32_t)1 << 16)          /* cid=1 */
		     | (((uint32_t)c->flags & 0xff) << 8)
		     | ((uint32_t)c->opcode & 0xff);
	memcpy(sqe +  0, &dw0,      4);
	memcpy(sqe +  4, &c->nsid,  4);
	memcpy(sqe +  8, &c->cdw2,  4);
	memcpy(sqe + 12, &c->cdw3,  4);
	memcpy(sqe + 40, &c->cdw10, 4);
	memcpy(sqe + 44, &c->cdw11, 4);
	memcpy(sqe + 48, &c->cdw12, 4);
	memcpy(sqe + 52, &c->cdw13, 4);
	memcpy(sqe + 56, &c->cdw14, 4);
	memcpy(sqe + 60, &c->cdw15, 4);
}

/* ------------------------------------------------------------------ */
/*  Admin helper: I/O utilities                                       */
/* ------------------------------------------------------------------ */
static int read_full(int fd, void *buf, size_t n)
{
	uint8_t *p = buf;
	while (n) {
		ssize_t r = read(fd, p, n);
		if (r == 0)
			return -1;
		if (r < 0) {
			if (errno == EINTR)
				continue;
			return -1;
		}
		p += r;
		n -= (size_t)r;
	}
	return 0;
}

static int write_full(int fd, const void *buf, size_t n)
{
	const uint8_t *p = buf;
	while (n) {
		ssize_t r = write(fd, p, n);
		if (r < 0) {
			if (errno == EINTR)
				continue;
			return -1;
		}
		p += r;
		n -= (size_t)r;
	}
	return 0;
}

static void handle_admin_client(int cfd)
{
	struct plink_nvme_passthru_cmd pcmd;
	uint8_t  sqe[PLINK_ADMIN_CMD_LEN];
	uint8_t  cpl[PLINK_ADMIN_CPL_LEN];
	uint8_t  data[PLINK_ADMIN_MAX_DATA];
	int32_t  rc = 0;
	uint32_t result = 0;
	int      direction = PLINK_DIR_NONE;
	uint32_t data_len = 0;

	memset(cpl, 0, sizeof(cpl));

	if (read_full(cfd, &pcmd, sizeof(pcmd)) < 0)
		return;

	data_len  = pcmd.data_len;
	direction = plink_admin_opcode_direction(pcmd.opcode);

	if (data_len > PLINK_ADMIN_MAX_DATA) {
		rc = E2BIG;
		goto reply;
	}

	/* Bidirectional admin commands aren't supported by our RPC. */
	if (direction == PLINK_DIR_BIDI) {
		rc = ENOTSUP;
		goto reply;
	}

	if (direction == PLINK_DIR_H2D && data_len) {
		if (read_full(cfd, data, data_len) < 0)
			return;
	}

	plink_build_sqe(sqe, &pcmd);
	rc = plink_admin_rpc(sqe, cpl, data, data_len, direction);

	/* CQE dword 0 = command-specific result */
	if (rc == 0)
		memcpy(&result, cpl, sizeof(result));

reply:
	if (write_full(cfd, &rc, sizeof(rc)) < 0)
		return;
	if (write_full(cfd, &result, sizeof(result)) < 0)
		return;
	if (rc == 0 && direction == PLINK_DIR_D2H && data_len)
		write_full(cfd, data, data_len);
}

static void *plink_admin_thread(void *arg)
{
	struct plink_data *pd = arg;

	while (__atomic_load_n(&pd->admin_run, __ATOMIC_ACQUIRE)) {
		int cfd = accept(pd->admin_listen_fd, NULL, NULL);
		if (cfd < 0) {
			if (errno == EINTR)
				continue;
			break;
		}
		handle_admin_client(cfd);
		close(cfd);
	}
	return NULL;
}

static int plink_admin_start(struct plink_data *pd)
{
	struct sockaddr_un addr;
	int fd;

	if (plink_admin_init() != 0) {
		log_err("parallelink: admin init failed, admin socket disabled\n");
		return -1;
	}

	unlink(PLINK_ADMIN_SOCKET_PATH);

	fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd < 0) {
		log_err("parallelink: admin socket() failed: %s\n",
			strerror(errno));
		plink_admin_teardown();
		return -1;
	}

	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, PLINK_ADMIN_SOCKET_PATH,
		sizeof(addr.sun_path) - 1);

	if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		log_err("parallelink: admin bind(%s) failed: %s\n",
			PLINK_ADMIN_SOCKET_PATH, strerror(errno));
		close(fd);
		plink_admin_teardown();
		return -1;
	}
	if (listen(fd, 4) < 0) {
		log_err("parallelink: admin listen failed: %s\n",
			strerror(errno));
		close(fd);
		unlink(PLINK_ADMIN_SOCKET_PATH);
		plink_admin_teardown();
		return -1;
	}

	pd->admin_listen_fd = fd;
	pd->admin_run       = 1;

	if (pthread_create(&pd->admin_thr, NULL,
			   plink_admin_thread, pd) != 0) {
		log_err("parallelink: pthread_create(admin) failed\n");
		close(fd);
		unlink(PLINK_ADMIN_SOCKET_PATH);
		plink_admin_teardown();
		return -1;
	}

	pd->admin_enabled = 1;
	log_info("parallelink: admin socket ready at %s\n",
		 PLINK_ADMIN_SOCKET_PATH);
	return 0;
}

static void plink_admin_stop(struct plink_data *pd)
{
	if (!pd->admin_enabled)
		return;

	__atomic_store_n(&pd->admin_run, 0, __ATOMIC_RELEASE);

	/*
	 * Unblock accept() by shutting down the listen socket. The helper
	 * thread sees run==0 next iteration and bails.
	 */
	if (pd->admin_listen_fd >= 0) {
		shutdown(pd->admin_listen_fd, SHUT_RDWR);
		close(pd->admin_listen_fd);
		pd->admin_listen_fd = -1;
	}
	pthread_join(pd->admin_thr, NULL);
	unlink(PLINK_ADMIN_SOCKET_PATH);
	plink_admin_teardown();
	pd->admin_enabled = 0;
}

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
	 * The engine owns a single CUDA context, one /dev/libnvm0 fd and a
	 * pile of DMA mappings. Forking (thread=0, numjobs>1) copies the
	 * parent's FDs but not the CUDA context, which leaves the child
	 * holding half-valid state. Refuse those configurations outright
	 * so the user gets a clear error instead of a mysterious crash.
	 */
	if (td->o.numjobs != 1) {
		log_err("parallelink: numjobs must be 1 (got %u)\n",
			td->o.numjobs);
		return -EINVAL;
	}
	if (!td->o.use_thread) {
		log_err("parallelink: thread=1 required "
			"(fork mode would break CUDA/DMA state)\n");
		return -EINVAL;
	}

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
	pd->admin_listen_fd = -1;
	pd->admin_run       = 0;
	pd->admin_enabled   = 0;

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

	/*
	 * Bring up the admin injection helper. Non-fatal if it fails —
	 * the main I/O engine keeps working, only out-of-band admin
	 * commands are unavailable.
	 */
	(void)plink_admin_start(pd);

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
 * This is how fio collects BW/IOPS statistics from the GPU-driven
 * I/O loop.
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
		plink_admin_stop(pd);
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
