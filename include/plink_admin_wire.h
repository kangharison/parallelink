#ifndef PLINK_ADMIN_WIRE_H
#define PLINK_ADMIN_WIRE_H

#include <stdint.h>

/*
 * parallelink admin socket: wire protocol
 * ----------------------------------------
 *
 * An out-of-band Unix-domain socket hosted by the parallelink fio
 * engine. External NVMe admin commands are forwarded over it to the
 * libnvm controller that already owns the device, so a user-space
 * tool can issue admin passthrough while a GPU workload is running
 * against the same SSD.
 *
 * This protocol mirrors libnvme's NVME_IOCTL_ADMIN_CMD payload
 * (struct nvme_passthru_cmd) one-to-one so that a libnvme-based
 * client (e.g. nvme-cli with the PLINK ioctl hook) can marshal its
 * command without any translation.
 *
 * Request:
 *   [ 72 B ] struct plink_nvme_passthru_cmd
 *   [ data_len B ] payload (only when the opcode is host->device
 *                  and cmd.data_len > 0)
 *
 * Reply:
 *   [  4 B ] int32_t  rc       (0=ok, >0=errno, <0=NVM error)
 *   [  4 B ] uint32_t result   (CQE dword0, echoed into cmd.result)
 *   [ data_len B ] payload     (only when direction is device->host,
 *                               cmd.data_len > 0 and rc == 0)
 *
 * Direction is derived from the opcode's low two bits per NVMe spec:
 *   00b = no data, 01b = h2d, 10b = d2h, 11b = bidir.
 */

#define PLINK_ADMIN_SOCKET_PATH "/tmp/nvme-admin.sock"

/* Max per-request data payload (matches PLINK_ADMIN_MAX_DATA in
 * gpu_engine.h — one NVMe memory page, PRP1-only). */
#define PLINK_ADMIN_WIRE_MAX_DATA 4096

/*
 * Self-contained mirror of Linux's `struct nvme_passthru_cmd`
 * (<linux/nvme_ioctl.h>). Kept here so that neither the engine nor
 * the CLI needs to pull in kernel headers, and so that the layout is
 * pinned on the wire regardless of host kernel version.
 *
 * Size must be exactly 72 bytes. Field order and offsets match the
 * kernel definition so a libnvme client can memcpy a struct
 * nvme_passthru_cmd straight into this.
 */
struct plink_nvme_passthru_cmd {
	uint8_t  opcode;
	uint8_t  flags;
	uint16_t rsvd1;
	uint32_t nsid;
	uint32_t cdw2;
	uint32_t cdw3;
	uint64_t metadata;
	uint64_t addr;
	uint32_t metadata_len;
	uint32_t data_len;
	uint32_t cdw10;
	uint32_t cdw11;
	uint32_t cdw12;
	uint32_t cdw13;
	uint32_t cdw14;
	uint32_t cdw15;
	uint32_t timeout_ms;
	uint32_t result;
};

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(struct plink_nvme_passthru_cmd) == 72,
	       "plink_nvme_passthru_cmd must match linux nvme_passthru_cmd");
#endif
#endif

/* Direction codes used by the engine's dispatcher. */
enum {
	PLINK_DIR_NONE = 0,
	PLINK_DIR_H2D  = 1,
	PLINK_DIR_D2H  = 2,
	PLINK_DIR_BIDI = 3,
};

/*
 * NVMe spec: bits [1:0] of the admin opcode encode data transfer
 * direction. Deriving it from the opcode keeps the wire protocol
 * flagless and matches what libnvme itself does internally.
 */
static inline int plink_admin_opcode_direction(uint8_t opcode)
{
	switch (opcode & 0x3) {
	case 0: return PLINK_DIR_NONE;
	case 1: return PLINK_DIR_H2D;
	case 2: return PLINK_DIR_D2H;
	default: return PLINK_DIR_BIDI;
	}
}

#endif /* PLINK_ADMIN_WIRE_H */
