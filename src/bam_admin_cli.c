/*
 * bam-admin-cli: out-of-band NVMe admin command injector for the
 * parallelink fio engine.
 *
 * Connects to PLINK_ADMIN_SOCKET_PATH (created by the engine's admin
 * helper thread) and forwards a single NVMe admin command per
 * invocation. Supports a handful of convenience subcommands
 * (id-ctrl, id-ns, smart-log, get-log) plus a raw escape hatch for
 * anything else.
 *
 * Wire protocol: see include/plink_admin_wire.h.
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "plink_admin_wire.h"

#define ADMIN_CMD_LEN   64
#define ADMIN_CPL_LEN   16
#define ADMIN_MAX_DATA  PLINK_ADMIN_WIRE_MAX_DATA

static void die(const char *msg)
{
	fprintf(stderr, "bam-admin-cli: %s\n", msg);
	exit(1);
}

static void diev(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "bam-admin-cli: ");
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, "\n");
	va_end(ap);
	exit(1);
}

/* ------------------------------------------------------------------ */
/*  Socket helpers                                                    */
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

static int connect_sock(const char *path)
{
	int fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd < 0)
		diev("socket: %s", strerror(errno));

	struct sockaddr_un addr;
	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

	if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0)
		diev("connect %s: %s", path, strerror(errno));

	return fd;
}


/* ------------------------------------------------------------------ */
/*  One-shot transaction                                              */
/* ------------------------------------------------------------------ */
struct txn_result {
	int32_t  rc;
	uint32_t result;          /* CQE dword0 */
	uint8_t  data[ADMIN_MAX_DATA];
	uint32_t data_len;
};

static void do_txn(const char *sock_path,
		   const struct plink_nvme_passthru_cmd *pcmd,
		   const void *h2d_data,
		   struct txn_result *out)
{
	if (pcmd->data_len > ADMIN_MAX_DATA)
		diev("data_len %u exceeds %u", pcmd->data_len, ADMIN_MAX_DATA);

	int direction = plink_admin_opcode_direction(pcmd->opcode);

	int fd = connect_sock(sock_path);

	if (write_full(fd, pcmd, sizeof(*pcmd)) < 0)
		die("write cmd");
	if (direction == PLINK_DIR_H2D && pcmd->data_len) {
		if (write_full(fd, h2d_data, pcmd->data_len) < 0)
			die("write data");
	}

	if (read_full(fd, &out->rc, sizeof(out->rc)) < 0)
		die("read rc");
	if (read_full(fd, &out->result, sizeof(out->result)) < 0)
		die("read result");

	out->data_len = 0;
	if (out->rc == 0 && direction == PLINK_DIR_D2H && pcmd->data_len) {
		if (read_full(fd, out->data, pcmd->data_len) < 0)
			die("read data");
		out->data_len = pcmd->data_len;
	}

	close(fd);
}

/* ------------------------------------------------------------------ */
/*  Output                                                            */
/* ------------------------------------------------------------------ */
static void hex_dump(const uint8_t *buf, size_t n)
{
	for (size_t i = 0; i < n; i += 16) {
		printf("%08zx  ", i);
		for (size_t j = 0; j < 16; j++) {
			if (i + j < n)
				printf("%02x ", buf[i + j]);
			else
				printf("   ");
			if (j == 7)
				printf(" ");
		}
		printf(" |");
		for (size_t j = 0; j < 16 && i + j < n; j++) {
			unsigned char c = buf[i + j];
			putchar(isprint(c) ? c : '.');
		}
		printf("|\n");
	}
}

static void report_rc(const struct txn_result *r)
{
	if (r->rc == 0) {
		fprintf(stderr, "rc=0 (success)\n");
		return;
	}
	if (r->rc > 0)
		fprintf(stderr, "rc=%d (errno: %s)\n", r->rc, strerror(r->rc));
	else
		fprintf(stderr, "rc=%d (NVM error)\n", r->rc);
}

/* ------------------------------------------------------------------ */
/*  Subcommands                                                       */
/* ------------------------------------------------------------------ */
static void passthru_init(struct plink_nvme_passthru_cmd *c,
			  uint8_t opcode, uint32_t nsid)
{
	memset(c, 0, sizeof(*c));
	c->opcode = opcode;
	c->nsid   = nsid;
}

static int cmd_id_ctrl(const char *sock)
{
	struct plink_nvme_passthru_cmd c;
	passthru_init(&c, 0x06, 0);           /* IDENTIFY */
	c.cdw10    = 0x01;                     /* CNS=1: Identify Controller */
	c.data_len = 4096;

	struct txn_result r;
	do_txn(sock, &c, NULL, &r);
	report_rc(&r);
	if (r.rc == 0)
		hex_dump(r.data, r.data_len);
	return r.rc ? 1 : 0;
}

static int cmd_id_ns(const char *sock, uint32_t nsid)
{
	struct plink_nvme_passthru_cmd c;
	passthru_init(&c, 0x06, nsid);
	c.cdw10    = 0x00;                     /* CNS=0: Identify Namespace */
	c.data_len = 4096;

	struct txn_result r;
	do_txn(sock, &c, NULL, &r);
	report_rc(&r);
	if (r.rc == 0)
		hex_dump(r.data, r.data_len);
	return r.rc ? 1 : 0;
}

static int cmd_get_log(const char *sock, uint8_t lid,
		       uint32_t size, uint32_t nsid)
{
	/* NUMD (0-based) = (size / 4) - 1. NVMe 1.3+: dword[10] =
	 * NUMDL<<16 | LID. */
	if (size == 0 || (size & 3))
		diev("get-log size %u must be nonzero multiple of 4", size);
	uint32_t numd = (size / 4) - 1;
	if (numd > 0xffff)
		diev("get-log size %u exceeds 256KB (NUMDL limit)", size);

	struct plink_nvme_passthru_cmd c;
	passthru_init(&c, 0x02, nsid);
	c.cdw10    = ((uint32_t)(numd & 0xffff) << 16) | lid;
	c.data_len = size;

	struct txn_result r;
	do_txn(sock, &c, NULL, &r);
	report_rc(&r);
	if (r.rc == 0)
		hex_dump(r.data, r.data_len);
	return r.rc ? 1 : 0;
}

static int cmd_get_features(const char *sock, uint8_t fid, uint8_t sel,
			    uint32_t nsid, uint32_t data_len)
{
	struct plink_nvme_passthru_cmd c;
	passthru_init(&c, 0x0a, nsid);
	/* dword[10]: SEL[10:8] | FID[7:0]
	 *   SEL 000 current, 001 default, 010 saved, 011 supported caps */
	c.cdw10    = ((uint32_t)(sel & 0x7) << 8) | fid;
	c.data_len = data_len;

	struct txn_result r;
	do_txn(sock, &c, NULL, &r);
	report_rc(&r);
	if (r.rc == 0) {
		printf("result (cqe dw0) = 0x%08x\n", r.result);
		if (r.data_len)
			hex_dump(r.data, r.data_len);
	}
	return r.rc ? 1 : 0;
}

static int parse_hex_cmd(const char *hex, uint8_t out[ADMIN_CMD_LEN])
{
	size_t n = 0;
	memset(out, 0, ADMIN_CMD_LEN);
	while (*hex && n < ADMIN_CMD_LEN) {
		while (*hex && !isxdigit((unsigned char)*hex))
			hex++;
		if (!*hex || !isxdigit((unsigned char)hex[1]))
			return -1;
		char buf[3] = { hex[0], hex[1], 0 };
		out[n++] = (uint8_t)strtoul(buf, NULL, 16);
		hex += 2;
	}
	return (int)n;
}

static int cmd_raw(const char *sock, const char *hex, uint32_t data_len)
{
	uint8_t sqe[ADMIN_CMD_LEN];
	int n = parse_hex_cmd(hex, sqe);
	if (n < 0)
		die("raw: malformed hex input");
	if (n != ADMIN_CMD_LEN)
		diev("raw: expected 64 bytes of hex, got %d", n);

	/* Project the 64-byte SQE onto a passthru command. The server
	 * rebuilds the SQE from these fields; PRP1/PRP2 are server-side. */
	uint32_t dw0;
	memcpy(&dw0, sqe + 0, 4);

	struct plink_nvme_passthru_cmd c;
	memset(&c, 0, sizeof(c));
	c.opcode = (uint8_t)(dw0 & 0xff);
	c.flags  = (uint8_t)((dw0 >> 8) & 0xff);
	memcpy(&c.nsid,  sqe +  4, 4);
	memcpy(&c.cdw2,  sqe +  8, 4);
	memcpy(&c.cdw3,  sqe + 12, 4);
	memcpy(&c.cdw10, sqe + 40, 4);
	memcpy(&c.cdw11, sqe + 44, 4);
	memcpy(&c.cdw12, sqe + 48, 4);
	memcpy(&c.cdw13, sqe + 52, 4);
	memcpy(&c.cdw14, sqe + 56, 4);
	memcpy(&c.cdw15, sqe + 60, 4);
	c.data_len = data_len;

	struct txn_result r;
	do_txn(sock, &c, NULL, &r);
	report_rc(&r);
	if (r.rc == 0) {
		printf("result (cqe dw0) = 0x%08x\n", r.result);
		if (r.data_len)
			hex_dump(r.data, r.data_len);
	}
	return r.rc ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/*  main                                                              */
/* ------------------------------------------------------------------ */
static void usage(void)
{
	fprintf(stderr,
"Usage: bam-admin-cli <subcommand> [args...]\n"
"\n"
"Subcommands:\n"
"  id-ctrl                           Identify Controller (4 KB)\n"
"  id-ns <nsid>                      Identify Namespace (4 KB)\n"
"  smart-log [nsid]                  Get Log Page LID=0x02 (512 B)\n"
"  get-log <lid> <size> [nsid]       Get arbitrary log page\n"
"  get-features <fid> [sel] [nsid] [data_len]\n"
"                                    Get Features (sel: 0=cur 1=def 2=saved 3=caps)\n"
"  raw <64-byte hex> [data_len]      Raw SQE, optional dev->host payload\n"
"\n"
"Connects to " PLINK_ADMIN_SOCKET_PATH ".\n");
	exit(2);
}

int main(int argc, char **argv)
{
	int i = 1;

	while (i < argc && strncmp(argv[i], "--", 2) == 0) {
		if (!strcmp(argv[i], "--help")) {
			usage();
		} else {
			diev("unknown flag %s", argv[i]);
		}
	}
	if (i >= argc)
		usage();

	const char *sock = PLINK_ADMIN_SOCKET_PATH;
	const char *sub = argv[i++];

	if (!strcmp(sub, "id-ctrl")) {
		return cmd_id_ctrl(sock);
	} else if (!strcmp(sub, "id-ns")) {
		if (i >= argc)
			die("id-ns: missing nsid");
		return cmd_id_ns(sock, (uint32_t)strtoul(argv[i], NULL, 0));
	} else if (!strcmp(sub, "smart-log")) {
		uint32_t nsid = 0xffffffff;
		if (i < argc)
			nsid = (uint32_t)strtoul(argv[i], NULL, 0);
		return cmd_get_log(sock, 0x02, 512, nsid);
	} else if (!strcmp(sub, "get-log")) {
		if (i + 1 >= argc)
			die("get-log: usage <lid> <size> [nsid]");
		uint8_t  lid  = (uint8_t)strtoul(argv[i++], NULL, 0);
		uint32_t size = (uint32_t)strtoul(argv[i++], NULL, 0);
		uint32_t nsid = 0xffffffff;
		if (i < argc)
			nsid = (uint32_t)strtoul(argv[i], NULL, 0);
		return cmd_get_log(sock, lid, size, nsid);
	} else if (!strcmp(sub, "get-features")) {
		if (i >= argc)
			die("get-features: missing fid");
		uint8_t  fid      = (uint8_t)strtoul(argv[i++], NULL, 0);
		uint8_t  sel      = 0;
		uint32_t nsid     = 0;
		uint32_t data_len = 0;
		if (i < argc)
			sel      = (uint8_t)strtoul(argv[i++], NULL, 0);
		if (i < argc)
			nsid     = (uint32_t)strtoul(argv[i++], NULL, 0);
		if (i < argc)
			data_len = (uint32_t)strtoul(argv[i++], NULL, 0);
		return cmd_get_features(sock, fid, sel, nsid, data_len);
	} else if (!strcmp(sub, "raw")) {
		if (i >= argc)
			die("raw: missing hex");
		const char *hex = argv[i++];
		uint32_t dlen = 0;
		if (i < argc)
			dlen = (uint32_t)strtoul(argv[i], NULL, 0);
		return cmd_raw(sock, hex, dlen);
	}

	diev("unknown subcommand %s", sub);
	return 2;
}
