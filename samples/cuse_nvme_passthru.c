/*
 * [한국어] CUSE 기반 NVMe 패스스루 디바이스 샘플 (cuse_nvme_passthru.c)
 *
 * === 파일의 역할 ===
 * libfuse3의 CUSE(Character device in Userspace) 인터페이스를 이용해 사용자
 * 공간에서 /dev/<name> 캐릭터 디바이스를 만들고, **수정되지 않은 nvme-cli**가
 * 그 디바이스로 보내는 NVME_IOCTL_ADMIN_CMD / NVME_IOCTL_IO_CMD ioctl을
 * 그대로 받는다. 본 샘플은 ioctl 디스패치 골격과 유저공간 데이터 버퍼를
 * 안전하게 가져오는 2-phase iovec 패턴까지만 보여주고, 실제 NVMe 명령 실행은
 * "TODO: BaM에 위임" 자리에 stub만 남겨 둔다.
 *
 * === 전체 아키텍처에서의 위치 ===
 * 기존 src/plink_ioctl_hook.c 방식:
 *   nvme-cli (libnvme를 --wrap=ioctl로 패치) → Unix 도메인 소켓
 *   → parallelink fio 엔진 → libnvm
 * CUSE 방식:
 *   nvme-cli (수정 없음) → open("/dev/cuse-nvme0") → ioctl(NVME_IOCTL_*)
 *   → 커널 fuse → libfuse → 본 프로세스 cuse_ioctl() 콜백 → libnvm
 * 즉 패치/wrap 없이 표준 디바이스 파일 의미론으로 통신한다.
 *
 * === 타 모듈과의 연결 ===
 * - 입력: nvme-cli (수정 없음)이 보내는 admin/IO 패스스루 ioctl
 * - 출력: libnvm(BaM)의 nvm_admin_xxx / nvm_cmd_xxx API (본 샘플에서는 stub)
 * - 의존: libfuse3 (cuse_lowlevel_main, fuse_reply_ioctl_retry 등),
 *         <linux/nvme_ioctl.h> (NVME_IOCTL_*, struct nvme_passthru_cmd)
 *
 * === 주요 함수/구조체 요약 ===
 * - main()                  : --name= 옵션 파싱, CUSE 디바이스 등록, 이벤트 루프
 * - cuse_open()/cuse_release(): 열림/닫힘 - 본 샘플은 별도 컨텍스트 없음
 * - cuse_ioctl()            : 핵심 - admin/IO 패스스루 ioctl 디스패치 (2-phase)
 * - handle_admin_passthru() : NVME_IOCTL_ADMIN_CMD 처리 (stub)
 * - handle_io_passthru()    : NVME_IOCTL_IO_CMD 처리 (stub)
 *
 * 빌드:
 *   gcc -O2 -Wall -D_FILE_OFFSET_BITS=64 \
 *       samples/cuse_nvme_passthru.c \
 *       $(pkg-config --cflags --libs fuse3) \
 *       -o cuse_nvme_passthru
 *
 * 실행/테스트:
 *   sudo ./cuse_nvme_passthru -f --name=cuse-nvme0
 *   # 다른 터미널에서 (수정 없는 nvme-cli):
 *   sudo nvme id-ctrl /dev/cuse-nvme0
 *   sudo nvme admin-passthru /dev/cuse-nvme0 \
 *        --opcode=0x06 --data-len=4096 --cdw10=1 -r
 */

#define FUSE_USE_VERSION 31
#define _GNU_SOURCE

#include <cuse_lowlevel.h>      /* [한국어] CUSE lowlevel API: cuse_lowlevel_main, struct cuse_info */
#include <fuse_lowlevel.h>      /* [한국어] fuse_req_t, fuse_reply_ioctl(_retry|_err), fuse_file_info */
#include <fuse_opt.h>           /* [한국어] --name=, -h 등 옵션 파싱 */

#include <sys/ioctl.h>          /* [한국어] _IOWR 매크로 — linux/nvme_ioctl.h 가 사용함 */
#include <linux/nvme_ioctl.h>   /* [한국어] NVME_IOCTL_ADMIN_CMD/IO_CMD, struct nvme_passthru_cmd  */

#include <errno.h>              /* [한국어] ENOTSUP/ENOMEM/ENOTTY 등 */
#include <stddef.h>             /* [한국어] offsetof — fuse_opt 매크로에서 사용 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/uio.h>            /* [한국어] struct iovec — fuse_reply_ioctl_retry 에 전달 */
#include <unistd.h>

/* ================================================================== */
/* 옵션 정의                                                           */
/* ================================================================== */

/*
 * [한국어]
 * --name=<X> 로 받은 디바이스 이름은 /dev/<X> 형태로 노출된다.
 * CUSE는 dev_info_argv 의 "DEVNAME=..." 항목으로 이 이름을 알아낸다.
 */
struct cuse_args {
	char *dev_name;   /* [한국어] /dev/<dev_name> — 미지정 시 cuse-nvme0 사용 */
	int   show_help;  /* [한국어] -h / --help 플래그 */
};

#define CUSE_OPT_KV(t, p) { t, offsetof(struct cuse_args, p), 1 }

static const struct fuse_opt cuse_opts[] = {
	CUSE_OPT_KV("--name=%s", dev_name), /* [한국어] /dev/<X> 이름 지정 */
	CUSE_OPT_KV("-h",        show_help),
	CUSE_OPT_KV("--help",    show_help),
	FUSE_OPT_END,
};

/* ================================================================== */
/* open / release                                                     */
/* ================================================================== */

/*
 * [한국어]
 * cuse_open - 디바이스 파일이 open(2)될 때마다 호출
 *
 * @req : libfuse 요청 핸들 (응답에 사용)
 * @fi  : open flags (O_RDONLY/O_RDWR 등). fi->fh 에 세션별 컨텍스트 보관 가능.
 *
 * 본 샘플은 별도 세션 상태가 없으므로 그대로 성공 반환만 한다.
 * 실제 통합에서는 nvme-cli 가 파일을 열 때 BaM 컨트롤러 핸들(nvm_ctrl_t*)을
 * fi->fh 에 매달거나, 아니면 그냥 전역 핸들을 쓰면 된다.
 */
static void cuse_open(fuse_req_t req, struct fuse_file_info *fi)
{
	fuse_reply_open(req, fi);  /* [한국어] open 성공 — 같은 fi 그대로 반환 */
}

/*
 * [한국어]
 * cuse_release - close(2) 시 호출. 자원 해제 자리.
 */
static void cuse_release(fuse_req_t req, struct fuse_file_info *fi)
{
	(void)fi;
	fuse_reply_err(req, 0);    /* [한국어] errno 0 == 성공 */
}

/* ================================================================== */
/* admin / IO passthru handlers (stub — 여기에 libnvm 호출 삽입)        */
/* ================================================================== */

/*
 * [한국어]
 * handle_admin_passthru - NVME_IOCTL_ADMIN_CMD 본처리
 *
 * @cmd     : nvme-cli가 채워서 보낸 입출력 구조체. addr/metadata 필드는
 *            *유저공간 가상주소*이므로 본 함수 안에서 직접 역참조하면 안 된다 —
 *            CUSE 콜백에서 미리 복사해 놓은 in_data/out_data 버퍼만 만진다.
 * @in_data : H2D 방향(opcode bit[1:0]==01)일 때 유저→커널→fuse를 거쳐 복사된 입력 버퍼.
 *            없으면 NULL.
 * @out_data: D2H 방향(==10)일 때 본 함수가 채워 돌려보낼 출력 버퍼. 없으면 NULL.
 * @result  : NVMe CQE dword0 (cmd.result 로 그대로 반사된다)
 *
 * 반환값:
 *    0     성공 (NVMe status=0)
 *   >0     NVMe spec 의 status code 값을 그대로 (libnvme 가 status로 해석)
 *   <0     호스트측 실패 (errno = -rc 로 fuse_reply_err 됨)
 *
 * 본 stub 은 opcode 0x06 (Identify) 만 더미 데이터로 채운다. 실제 통합 시
 * 다음과 같은 라인이 들어간다:
 *   nvm_aq_ref ref = (전역에서 관리)
 *   nvm_admin_identify_ctrl(ref, dma_buf);
 *   memcpy(out_data, dma_buf, cmd->data_len);
 */
static int handle_admin_passthru(struct nvme_passthru_cmd *cmd,
				 const void *in_data, void *out_data,
				 uint32_t *result)
{
	*result = 0;  /* [한국어] CQE dword0 기본값 */

	switch (cmd->opcode) {
	case 0x06: /* [한국어] Identify (Admin) */
		if (out_data && cmd->data_len) {
			memset(out_data, 0, cmd->data_len);
			/* [한국어] Identify Controller 데이터의 MN(Model Number) 위치(24..63)에 표식만 */
			static const char tag[] = "PARALLELINK-CUSE-STUB";
			const size_t off = 24;
			if (cmd->data_len > off) {
				size_t n = sizeof(tag) - 1;
				if (n > cmd->data_len - off) n = cmd->data_len - off;
				memcpy((char *)out_data + off, tag, n);
			}
		}
		(void)in_data;
		return 0;

	/* [한국어] 기타 admin opcode 들은 같은 식으로 분기. 필요한 것만 진짜로 처리. */
	default:
		fprintf(stderr,
			"[cuse] unhandled admin opcode 0x%02x nsid=%u dlen=%u\n",
			cmd->opcode, cmd->nsid, cmd->data_len);
		(void)in_data;
		(void)out_data;
		return ENOTSUP; /* [한국어] >0 으로 NVMe-side 실패로 보고 */
	}
}

/*
 * [한국어]
 * handle_io_passthru - NVME_IOCTL_IO_CMD 본처리 (stub)
 *
 * 실제 통합 시에는 cmd->cdw10/11 로 SLBA, cdw12 로 NLB(0-based)를 꺼내
 * libnvm 의 nvm_cmd_rw / nvm_queue_submit 경로로 위임한다.
 */
static int handle_io_passthru(struct nvme_passthru_cmd *cmd,
			      const void *in_data, void *out_data,
			      uint32_t *result)
{
	(void)in_data; (void)out_data;
	*result = 0;
	uint64_t slba = ((uint64_t)cmd->cdw11 << 32) | cmd->cdw10;
	fprintf(stderr,
		"[cuse] io passthru opcode=0x%02x nsid=%u slba=%llu nlb+1=%u dlen=%u (stub)\n",
		cmd->opcode, cmd->nsid, (unsigned long long)slba,
		(cmd->cdw12 & 0xFFFF) + 1, cmd->data_len);
	return ENOTSUP;
}

/* ================================================================== */
/* 핵심: ioctl 디스패처                                                 */
/* ================================================================== */

/*
 * [한국어]
 * cuse_ioctl - CUSE 의 ioctl 콜백.
 *
 * 호출 흐름 (2~3 phase):
 *  1) 처음 진입 시 in_bufsz/out_bufsz 가 0 이거나 작다. 우리는
 *     fuse_reply_ioctl_retry() 로 "이런 입력/출력 영역을 더 가져와 주세요"
 *     라고 회신한다. 커널이 user 공간에서 그 영역을 복사해 와서 우리를
 *     다시 호출한다.
 *  2) 두 번째 호출 시 in_buf 에 struct nvme_passthru_cmd 가 들어 있다.
 *     cmd.data_len 과 opcode 의 방향 비트로 추가 데이터 영역이 필요한지
 *     판단한다. 필요하면 한 번 더 retry.
 *  3) 마지막 호출에서 모든 데이터가 갖춰지면 실제 처리 후 fuse_reply_ioctl().
 *
 * 주의: arg 는 userspace 의 *원본 포인터* 이다. 우리는 이 포인터를
 *       struct iovec 의 base 로 그대로 넘겨서 커널에게 "여기서 가져와 주세요"
 *       라고 알려 줄 뿐, 직접 역참조하지 않는다.
 */
static void cuse_ioctl(fuse_req_t req, int cmd_no, void *arg,
		       struct fuse_file_info *fi, unsigned flags,
		       const void *in_buf, size_t in_bufsz, size_t out_bufsz)
{
	(void)fi;

	/* [한국어] 32-bit compat 호출은 본 샘플 범위 밖. */
	if (flags & FUSE_IOCTL_COMPAT) {
		fuse_reply_err(req, ENOSYS);
		return;
	}

	/* [한국어] NVME_IOCTL_* 들은 0x80000000 이상의 값(_IOC_READ|WRITE 비트)이라
	 *          signed int 와 비교하면 -Wtype-limits 경고가 난다. unsigned 로 캐스팅. */
	switch ((unsigned int)cmd_no) {
	case NVME_IOCTL_ADMIN_CMD:
	case NVME_IOCTL_IO_CMD: {
		const size_t cmd_sz = sizeof(struct nvme_passthru_cmd); /* [한국어] 72B */

		/* ---------------- phase 1: cmd 본체 수령 ---------------- */
		if (in_bufsz < cmd_sz) {
			/* [한국어] 입력으로 arg 위치의 72B 를 받고,
			 *          출력도 같은 위치에 72B 를 채워서 user 에게 돌려준다
			 *          (cmd.result 갱신 때문). */
			struct iovec in_iov  = { arg, cmd_sz };
			struct iovec out_iov = { arg, cmd_sz };
			fuse_reply_ioctl_retry(req, &in_iov, 1, &out_iov, 1);
			return;
		}

		/* [한국어] 두 번째 진입 — cmd 본체가 in_buf 앞에 있다. 복사해 와서 사용. */
		struct nvme_passthru_cmd cmd;
		memcpy(&cmd, in_buf, cmd_sz);

		/* [한국어] NVMe 스펙: opcode bit[1:0] == 데이터 방향
		 *          0=none, 1=h2d, 2=d2h, 3=bidi (PRP에서 의미 없음 → 미지원) */
		const int dir = cmd.opcode & 0x3;
		if (dir == 3) {
			fuse_reply_err(req, ENOTSUP);
			return;
		}

		const int need_in_data  = (dir == 1) && cmd.data_len; /* H2D */
		const int need_out_data = (dir == 2) && cmd.data_len; /* D2H */
		const size_t want_in    = cmd_sz + (need_in_data  ? cmd.data_len : 0);
		const size_t want_out   = cmd_sz + (need_out_data ? cmd.data_len : 0);

		/* ---------------- phase 2: 데이터 영역 수령 ---------------- */
		if (in_bufsz < want_in || out_bufsz < want_out) {
			struct iovec iv_in[2], iv_out[2];
			int ni = 0, no = 0;

			iv_in[ni++]  = (struct iovec){ arg, cmd_sz };
			if (need_in_data)
				iv_in[ni++]  = (struct iovec){
					(void *)(uintptr_t)cmd.addr, cmd.data_len };

			iv_out[no++] = (struct iovec){ arg, cmd_sz };
			if (need_out_data)
				iv_out[no++] = (struct iovec){
					(void *)(uintptr_t)cmd.addr, cmd.data_len };

			fuse_reply_ioctl_retry(req, iv_in, ni, iv_out, no);
			return;
		}

		/* ---------------- phase 3: 실처리 ---------------- */
		const void *in_data  = need_in_data
			? (const uint8_t *)in_buf + cmd_sz : NULL;

		/* [한국어] out_buf 는 우리가 준비해서 fuse_reply_ioctl 에 넘긴다.
		 *          레이아웃은 retry 시 알려준 out_iov 와 동일하게
		 *          [cmd 72B] [data_len B if D2H] 순. */
		size_t out_total = cmd_sz + (need_out_data ? cmd.data_len : 0);
		void *out = malloc(out_total);
		if (!out) {
			fuse_reply_err(req, ENOMEM);
			return;
		}
		memcpy(out, &cmd, cmd_sz);
		void *out_data = need_out_data ? (uint8_t *)out + cmd_sz : NULL;

		uint32_t result = 0;
		int rc;
		if ((unsigned int)cmd_no == NVME_IOCTL_ADMIN_CMD)
			rc = handle_admin_passthru(&cmd, in_data, out_data, &result);
		else
			rc = handle_io_passthru(&cmd, in_data, out_data, &result);

		if (rc < 0) {
			/* [한국어] 호스트측 실패 — ioctl(2) 가 -1/errno 로 보이게 함 */
			free(out);
			fuse_reply_err(req, -rc);
			return;
		}

		/* [한국어] cmd.result 갱신. nvme-cli 는 이걸 보고 CQE dword0 로 인식. */
		((struct nvme_passthru_cmd *)out)->result = result;

		/*
		 * [한국어] NVMe 패스스루 ioctl 의 반환 규약:
		 *   == 0  성공
		 *   >  0  NVMe status code (libnvme 는 그대로 status 로 사용)
		 *   <  0  ioctl 자체 실패는 위에서 fuse_reply_err 로 처리됨
		 * 따라서 rc(>=0) 를 그대로 retval 로 반환한다.
		 */
		fuse_reply_ioctl(req, rc, out, out_total);
		free(out);
		return;
	}

	default:
		/* [한국어] NVME_IOCTL_RESET, NVME_IOCTL_SUBSYS_RESET, NVME_IOCTL_*64 등
		 *          미지원. 필요해지는 순간 case 추가하면 된다. */
		fuse_reply_err(req, ENOTTY);
		return;
	}
}

/* ================================================================== */
/* CUSE 등록                                                           */
/* ================================================================== */

static const struct cuse_lowlevel_ops cuse_clop = {
	.open    = cuse_open,
	.release = cuse_release,
	.ioctl   = cuse_ioctl,
};

int main(int argc, char *argv[])
{
	struct fuse_args args = FUSE_ARGS_INIT(argc, argv);
	struct cuse_args ca   = { .dev_name = NULL, .show_help = 0 };

	if (fuse_opt_parse(&args, &ca, cuse_opts, NULL) < 0)
		return 1;

	if (ca.show_help) {
		fprintf(stderr,
			"usage: %s [fuse-opts] --name=<devname>\n"
			"  e.g. sudo %s -f --name=cuse-nvme0\n",
			argv[0], argv[0]);
		return 0;
	}
	if (!ca.dev_name)
		ca.dev_name = strdup("cuse-nvme0");

	/* [한국어] CUSE 는 디바이스 이름을 "DEVNAME=..." 환경문자열로 받는다. */
	char devname_buf[128];
	snprintf(devname_buf, sizeof(devname_buf), "DEVNAME=%s", ca.dev_name);
	const char *dev_info_argv[] = { devname_buf };

	struct cuse_info ci = {
		.dev_major     = 0,                      /* [한국어] 0 = 동적 할당 */
		.dev_minor     = 0,
		.dev_info_argc = 1,
		.dev_info_argv = dev_info_argv,
		/*
		 * [한국어] 핵심: nvme ioctl 들은 _IOC_READ|_IOC_WRITE 비트나 size
		 *          필드가 fuse 의 기본 검증을 못 통과한다. 임의 _IOC 번호를
		 *          그대로 받으려면 반드시 이 플래그를 켜야 한다.
		 */
		.flags         = CUSE_UNRESTRICTED_IOCTL,
	};

	/*
	 * [한국어] TODO: 여기서 libnvm 컨트롤러 초기화.
	 *   nvm_ctrl_t *ctrl;
	 *   nvm_ctrl_init(&ctrl, "/dev/libnvm0");
	 *   ... admin queue / DMA 버퍼 풀 셋업 ...
	 *   전역 또는 cuse_clop 의 userdata 에 보관해서 handle_*_passthru 에서 참조.
	 */

	return cuse_lowlevel_main(args.argc, args.argv, &ci, &cuse_clop, NULL);
}
