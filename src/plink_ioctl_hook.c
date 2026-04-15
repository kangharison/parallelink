/*
 * plink_ioctl_hook: intercept libnvme's admin passthru ioctls and
 * redirect them to the parallelink out-of-band admin socket so an
 * external fio engine can execute them on behalf of this process.
 *
 * Only NVME_IOCTL_ADMIN_CMD is hooked. Everything else (including
 * NVME_IOCTL_IO_CMD and the 64-bit variants) falls through to the
 * real ioctl(2).
 *
 * This object is linked into libnvme when it is built with -DPLINK
 * (see patches/libnvme-plink-ioctl-hook.patch). nvme-cli then picks
 * up the hook transparently by virtue of linking that libnvme.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <linux/nvme_ioctl.h>

#include "plink_admin_wire.h"

/* ------------------------------------------------------------------ */
/*  I/O helpers                                                       */
/* ------------------------------------------------------------------ */
static int read_full(int fd, void *buf, size_t n)
{
	uint8_t *p = buf;
	while (n) {
		ssize_t r = read(fd, p, n);
		if (r == 0) {
			errno = EIO;
			return -1;
		}
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

static int connect_admin_sock(void)
{
	int fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd < 0)
		return -1;

	struct sockaddr_un addr;
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, PLINK_ADMIN_SOCKET_PATH,
		sizeof(addr.sun_path) - 1);

	if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
		int e = errno;
		close(fd);
		errno = e;
		return -1;
	}
	return fd;
}

/* ------------------------------------------------------------------ */
/*  Admin passthru forwarding                                         */
/* ------------------------------------------------------------------ */
/*
 * Marshal a kernel nvme_passthru_cmd onto the wire, exchange with
 * the engine, and fill cmd->result back for the caller. Returns the
 * same convention as ioctl(NVME_IOCTL_ADMIN_CMD):
 *   0              success
 *   -1 with errno  host-side failure (socket, ENOTSUP, etc.)
 *
 * NVMe device errors reported by the engine as a non-zero rc are
 * currently surfaced as -1 / EIO, since the engine's reply only
 * carries rc and CQE dword 0. If we ever need the real status code
 * we can widen the reply; for now this matches what libnvme callers
 * already handle (they check for err != 0 and treat it as failure).
 */
static int forward_admin(struct nvme_passthru_cmd *cmd)
{
	struct plink_nvme_passthru_cmd w;
	int sfd = -1;
	int direction;
	int32_t rc;
	uint32_t result = 0;
	int saved_errno;

	if (cmd->data_len > PLINK_ADMIN_WIRE_MAX_DATA) {
		errno = E2BIG;
		return -1;
	}

	memset(&w, 0, sizeof(w));
	w.opcode       = cmd->opcode;
	w.flags        = cmd->flags;
	w.rsvd1        = cmd->rsvd1;
	w.nsid         = cmd->nsid;
	w.cdw2         = cmd->cdw2;
	w.cdw3         = cmd->cdw3;
	w.metadata     = cmd->metadata;
	w.addr         = cmd->addr;
	w.metadata_len = cmd->metadata_len;
	w.data_len     = cmd->data_len;
	w.cdw10        = cmd->cdw10;
	w.cdw11        = cmd->cdw11;
	w.cdw12        = cmd->cdw12;
	w.cdw13        = cmd->cdw13;
	w.cdw14        = cmd->cdw14;
	w.cdw15        = cmd->cdw15;
	w.timeout_ms   = cmd->timeout_ms;
	w.result       = 0;

	direction = plink_admin_opcode_direction(w.opcode);
	if (direction == PLINK_DIR_BIDI) {
		errno = ENOTSUP;
		return -1;
	}

	sfd = connect_admin_sock();
	if (sfd < 0)
		return -1;

	if (write_full(sfd, &w, sizeof(w)) < 0)
		goto io_err;
	if (direction == PLINK_DIR_H2D && w.data_len) {
		if (write_full(sfd, (const void *)(uintptr_t)cmd->addr,
			       w.data_len) < 0)
			goto io_err;
	}

	if (read_full(sfd, &rc, sizeof(rc)) < 0)
		goto io_err;
	if (read_full(sfd, &result, sizeof(result)) < 0)
		goto io_err;

	if (rc == 0 && direction == PLINK_DIR_D2H && w.data_len) {
		if (read_full(sfd, (void *)(uintptr_t)cmd->addr,
			      w.data_len) < 0)
			goto io_err;
	}

	close(sfd);

	cmd->result = result;

	if (rc == 0)
		return 0;
	errno = (rc > 0) ? rc : EIO;
	return -1;

io_err:
	saved_errno = errno ? errno : EIO;
	if (sfd >= 0)
		close(sfd);
	errno = saved_errno;
	return -1;
}

/* ------------------------------------------------------------------ */
/*  Public entry point                                                */
/* ------------------------------------------------------------------ */
int plink_ioctl_hook(int fd, unsigned long ioctl_cmd, void *arg)
{
	if (ioctl_cmd == NVME_IOCTL_ADMIN_CMD)
		return forward_admin((struct nvme_passthru_cmd *)arg);

	/* Everything else — I/O passthru, 64-bit variants, reset,
	 * namespace management, etc. — is not our business. */
	return ioctl(fd, ioctl_cmd, arg);
}
