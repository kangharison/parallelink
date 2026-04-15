/*
 * bam-admin-cli: out-of-band NVMe admin command injector for the
 * parallelink fio engine.
 *
 * Connects to /tmp/bam-admin-<pid>.sock (created by the engine's
 * admin helper thread) and forwards a single NVMe admin command
 * per invocation. Supports a handful of convenience subcommands
 * (id-ctrl, id-ns, smart-log, get-log) plus a raw escape hatch for
 * anything else.
 *
 * Wire protocol (matches gpu_engine.c):
 *   request:
 *     [ 64 B ] nvm_cmd_t (raw SQE, opcode in byte[0] bits [7:0])
 *     [  4 B ] uint32_t data_len
 *     [  4 B ] uint32_t direction (0=none, 1=h2d, 2=d2h)
 *     [  data_len B ] payload when direction==1
 *   response:
 *     [  4 B ] int32_t rc (0=ok, <0=NVM err, >0=errno)
 *     [ 16 B ] nvm_cpl_t completion entry
 *     [  data_len B ] payload when direction==2 and rc==0
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <errno.h>
#include <glob.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define ADMIN_CMD_LEN   64
#define ADMIN_CPL_LEN   16
#define ADMIN_MAX_DATA  4096

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

static char *resolve_sock_path(int pid)
{
	static char buf[128];
	if (pid > 0) {
		snprintf(buf, sizeof(buf), "/tmp/bam-admin-%d.sock", pid);
		return buf;
	}

	glob_t g;
	int rc = glob("/tmp/bam-admin-*.sock", 0, NULL, &g);
	if (rc != 0)
		die("no /tmp/bam-admin-*.sock found. Use --pid.");
	if (g.gl_pathc > 1) {
		fprintf(stderr,
			"bam-admin-cli: multiple sockets found:\n");
		for (size_t i = 0; i < g.gl_pathc; i++)
			fprintf(stderr, "  %s\n", g.gl_pathv[i]);
		globfree(&g);
		die("pass --pid <pid> to disambiguate.");
	}
	snprintf(buf, sizeof(buf), "%s", g.gl_pathv[0]);
	globfree(&g);
	return buf;
}

/* ------------------------------------------------------------------ */
/*  NVMe command construction                                         */
/* ------------------------------------------------------------------ */
static void set_u32(uint8_t cmd[ADMIN_CMD_LEN], int dword_idx, uint32_t v)
{
	memcpy(cmd + dword_idx * 4, &v, 4);
}

static void build_header(uint8_t cmd[ADMIN_CMD_LEN],
			 uint8_t opcode, uint32_t nsid)
{
	memset(cmd, 0, ADMIN_CMD_LEN);
	/* dword[0]: cid<<16 | opcode (fuse=psdt=0). cid=1 arbitrary. */
	set_u32(cmd, 0, ((uint32_t)1 << 16) | (opcode & 0x7f));
	set_u32(cmd, 1, nsid);
}

/* ------------------------------------------------------------------ */
/*  One-shot transaction                                              */
/* ------------------------------------------------------------------ */
struct txn_result {
	int32_t rc;
	uint8_t cpl[ADMIN_CPL_LEN];
	uint8_t data[ADMIN_MAX_DATA];
	uint32_t data_len;
};

static void do_txn(const char *sock_path,
		   const uint8_t cmd[ADMIN_CMD_LEN],
		   uint32_t data_len, uint32_t direction,
		   const void *h2d_data,
		   struct txn_result *out)
{
	if (data_len > ADMIN_MAX_DATA)
		diev("data_len %u exceeds %u", data_len, ADMIN_MAX_DATA);

	int fd = connect_sock(sock_path);

	if (write_full(fd, cmd, ADMIN_CMD_LEN) < 0)
		die("write cmd");
	uint32_t hdr[2] = { data_len, direction };
	if (write_full(fd, hdr, sizeof(hdr)) < 0)
		die("write hdr");
	if (direction == 1 && data_len) {
		if (write_full(fd, h2d_data, data_len) < 0)
			die("write data");
	}

	if (read_full(fd, &out->rc, sizeof(out->rc)) < 0)
		die("read rc");
	if (read_full(fd, out->cpl, ADMIN_CPL_LEN) < 0)
		die("read cpl");

	out->data_len = 0;
	if (out->rc == 0 && direction == 2 && data_len) {
		if (read_full(fd, out->data, data_len) < 0)
			die("read data");
		out->data_len = data_len;
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

static void print_cpl(const uint8_t cpl[ADMIN_CPL_LEN])
{
	uint32_t d0, d1, d2, d3;
	memcpy(&d0, cpl +  0, 4);
	memcpy(&d1, cpl +  4, 4);
	memcpy(&d2, cpl +  8, 4);
	memcpy(&d3, cpl + 12, 4);
	printf("cpl: dw0=0x%08x dw1=0x%08x dw2=0x%08x dw3=0x%08x\n",
	       d0, d1, d2, d3);
}

/* ------------------------------------------------------------------ */
/*  Subcommands                                                       */
/* ------------------------------------------------------------------ */
static int cmd_id_ctrl(const char *sock)
{
	uint8_t cmd[ADMIN_CMD_LEN];
	build_header(cmd, 0x06, 0);       /* IDENTIFY */
	set_u32(cmd, 10, 0x01);            /* CNS=1: Identify Controller */

	struct txn_result r;
	do_txn(sock, cmd, 4096, 2, NULL, &r);
	report_rc(&r);
	if (r.rc == 0)
		hex_dump(r.data, r.data_len);
	return r.rc ? 1 : 0;
}

static int cmd_id_ns(const char *sock, uint32_t nsid)
{
	uint8_t cmd[ADMIN_CMD_LEN];
	build_header(cmd, 0x06, nsid);
	set_u32(cmd, 10, 0x00);            /* CNS=0: Identify Namespace */

	struct txn_result r;
	do_txn(sock, cmd, 4096, 2, NULL, &r);
	report_rc(&r);
	if (r.rc == 0)
		hex_dump(r.data, r.data_len);
	return r.rc ? 1 : 0;
}

static int cmd_get_log(const char *sock, uint8_t lid,
		       uint32_t size, uint32_t nsid)
{
	uint8_t cmd[ADMIN_CMD_LEN];
	build_header(cmd, 0x02, nsid);

	/* NUMD (0-based) = (size / 4) - 1, in the low 16 bits of dword[10]
	 * are NUMDL + LID. NVMe 1.3+ split: dword[10] = NUMDL<<16 | LID. */
	if (size == 0 || (size & 3))
		diev("get-log size %u must be nonzero multiple of 4", size);
	uint32_t numd = (size / 4) - 1;
	if (numd > 0xffff)
		diev("get-log size %u exceeds 256KB (NUMDL limit)", size);
	set_u32(cmd, 10, ((uint32_t)(numd & 0xffff) << 16) | lid);
	set_u32(cmd, 11, 0);
	set_u32(cmd, 12, 0);
	set_u32(cmd, 13, 0);

	struct txn_result r;
	do_txn(sock, cmd, size, 2, NULL, &r);
	report_rc(&r);
	if (r.rc == 0)
		hex_dump(r.data, r.data_len);
	return r.rc ? 1 : 0;
}

static int cmd_get_features(const char *sock, uint8_t fid, uint8_t sel,
			    uint32_t nsid, uint32_t data_len)
{
	uint8_t cmd[ADMIN_CMD_LEN];
	build_header(cmd, 0x0a, nsid);

	/* dword[10]: SEL[10:8] | FID[7:0]
	 *   SEL 000 current, 001 default, 010 saved, 011 supported caps */
	set_u32(cmd, 10, ((uint32_t)(sel & 0x7) << 8) | fid);

	/*
	 * Most Get Features commands return their value in completion
	 * dword[0]. Some (LBA Range Type, Host Identifier, Timestamp,
	 * Host Memory Buffer info, ...) also write a payload via PRP1;
	 * pass data_len > 0 to receive it.
	 */
	uint32_t direction = data_len ? 2 : 0;

	struct txn_result r;
	do_txn(sock, cmd, data_len, direction, NULL, &r);
	report_rc(&r);
	if (r.rc == 0) {
		print_cpl(r.cpl);
		if (r.data_len)
			hex_dump(r.data, r.data_len);
	}
	return r.rc ? 1 : 0;
}

static int parse_hex_cmd(const char *hex, uint8_t cmd[ADMIN_CMD_LEN])
{
	size_t n = 0;
	memset(cmd, 0, ADMIN_CMD_LEN);
	while (*hex && n < ADMIN_CMD_LEN) {
		while (*hex && !isxdigit((unsigned char)*hex))
			hex++;
		if (!*hex || !isxdigit((unsigned char)hex[1]))
			return -1;
		char buf[3] = { hex[0], hex[1], 0 };
		cmd[n++] = (uint8_t)strtoul(buf, NULL, 16);
		hex += 2;
	}
	return (int)n;
}

static int cmd_raw(const char *sock, const char *hex, uint32_t data_len)
{
	uint8_t cmd[ADMIN_CMD_LEN];
	int n = parse_hex_cmd(hex, cmd);
	if (n < 0)
		die("raw: malformed hex input");
	if (n != ADMIN_CMD_LEN)
		diev("raw: expected 64 bytes of hex, got %d", n);

	uint32_t direction = data_len ? 2 : 0;

	struct txn_result r;
	do_txn(sock, cmd, data_len, direction, NULL, &r);
	report_rc(&r);
	if (r.rc == 0 && r.data_len)
		hex_dump(r.data, r.data_len);
	return r.rc ? 1 : 0;
}

/* ------------------------------------------------------------------ */
/*  main                                                              */
/* ------------------------------------------------------------------ */
static void usage(void)
{
	fprintf(stderr,
"Usage: bam-admin-cli [--pid <pid>] <subcommand> [args...]\n"
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
"Without --pid, globs /tmp/bam-admin-*.sock. Multiple sockets => error.\n");
	exit(2);
}

int main(int argc, char **argv)
{
	int pid = 0;
	int i = 1;

	while (i < argc && strncmp(argv[i], "--", 2) == 0) {
		if (!strcmp(argv[i], "--pid") && i + 1 < argc) {
			pid = atoi(argv[i + 1]);
			i += 2;
		} else if (!strcmp(argv[i], "--help")) {
			usage();
		} else {
			diev("unknown flag %s", argv[i]);
		}
	}
	if (i >= argc)
		usage();

	const char *sock = resolve_sock_path(pid);
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
