# parallelink design

이 문서는 현재 코드 기준의 설계 문서다. 오래된 구현 명세에 있던
`cudaMallocManaged()` hot path, GPU deadline timer, resident daemon 기반 IPC는
현재 설계가 아니다.

## Goal

parallelink는 fio external ioengine으로 로드되는 GPU-initiated NVMe I/O
엔진이다. fio는 workload 설정, 실행 시간 제어, 통계 수집을 맡고, 실제 I/O
submit/complete 루프는 CUDA persistent kernel이 BaM/libnvm device API를 통해
NVMe queue pair에 직접 접근한다.

핵심 목표는 CPU가 per-I/O submit 또는 completion path에 들어가지 않는 것이다.
CPU는 시작 시 리소스를 구성하고, 실행 중에는 통계용 completion counter를
주기적으로 복사하며, 종료 시 kernel shutdown과 리소스 해제를 수행한다.

## Repository Layout

```
parallelink/
├── build.sh                  # fio, BaM/libnvm, parallelink, nvme-cli 통합 빌드
├── CMakeLists.txt            # parallelink.so와 plink_hook 빌드
├── include/
│   ├── gpu_engine.h          # C/CUDA 공용 엔진 ABI
│   └── plink_admin_wire.h    # admin socket wire protocol
├── src/
│   ├── gpu_engine.c          # fio external ioengine callbacks
│   ├── gpu_worker.cu         # CUDA kernel, BaM/libnvm setup, admin RPC backend
│   └── plink_ioctl_hook.c    # nvme-cli/libnvme admin ioctl forwarding hook
├── extern/
│   ├── fio/                  # fio submodule
│   ├── bam/                  # BaM/libnvm submodule
│   └── nvme-cli/             # patched nvme-cli/libnvme user
└── patches/                  # local patches applied by build.sh
```

GPU/NVMe state는 fio engine 인스턴스가 직접 소유하고 `cleanup()`에서 해제한다.

## Build Artifacts

`build.sh`는 다음 산출물을 `dist/`에 모은다.

- `fio`: external engine을 로드하는 fio binary
- `parallelink.so`: fio external ioengine
- `libnvm.so`: BaM userspace library
- `libnvm.ko`: BaM kernel module, 빌드 가능할 때만 생성
- `nvme`: PLINK ioctl hook이 링크된 nvme-cli binary
- `run-fio.sh`: `FIO_EXT_ENG_DIR`와 `LD_LIBRARY_PATH` 설정 wrapper

`CMakeLists.txt`의 직접 산출물은 `parallelink.so`와 `libplink_hook.a`다.
`libplink_hook.a`는 nvme-cli/libnvme 빌드 시 `--wrap=ioctl` 경로로 링크된다.

## Runtime Data Path

```
fio job thread
  └─ parallelink.so
      ├─ fio_plink_queue(): io_u token만 ring에 보관
      ├─ fio_plink_getevents(): GPU done counter mirror polling
      └─ gpu_worker.cu
          ├─ Controller / QueuePair / page_cache_t 초기화
          └─ plink_io_worker<<<...>>>()
              ├─ BaM read_data() 또는 write_data()
              ├─ NVMe SQ/CQ MMIO 및 DMA path
              └─ device-memory done counter atomicAdd()
```

fio는 여전히 `io_u` completion path를 통해 bandwidth/IOPS를 계산한다. 다만
`io_u` 자체는 실제 payload가 아니라 accounting token이다. GPU kernel이 완료한
I/O 개수만큼 `getevents()`가 queued token을 completion으로 되돌려 fio 통계
경로를 진행시킨다.

## fio Engine

`src/gpu_engine.c`는 fio가 호출하는 `struct ioengine_ops ioengine`을 export한다.

- `init`: single job/thread mode를 검증하고, diskless dummy file을 등록한 뒤
  GPU와 NVMe 리소스를 초기화한다.
- `queue`: GPU에 I/O를 전달하지 않는다. fio accounting을 위한 `io_u` token을
  내부 ring에 넣는다.
- `commit`: no-op이다. GPU가 persistent kernel 내부에서 직접 submit한다.
- `getevents`: device-only done counter를 copy stream으로 host mirror에 복사하고
  새 completion 개수를 계산한다.
- `event`: ring에서 `io_u` token을 꺼내 fio에 completion으로 반환한다.
- `cleanup`: admin socket, GPU kernel, BaM/libnvm 리소스, ring을 정리한다.

엔진은 `numjobs=1`과 `thread=1`만 허용한다. fork mode는 CUDA context와 libnvm
file descriptor/DMA mapping을 안전하게 복제하지 못하므로 명시적으로 거부한다.

## GPU Worker

`src/gpu_worker.cu`는 host bring-up과 device kernel을 함께 가진다. 현재 전역
context `g_ctx`가 단일 fio job instance의 lifetime을 나타낸다.

리소스 구성:

- `Controller`: BaM/libnvm controller handle
- `page_cache_t`: GPU-resident data buffer와 PRP backing
- `Controller **d_ctrls`: device에서 queue pair에 접근하기 위한 controller table
- `cudaStream_t compute_stream`: persistent kernel 실행 stream
- `cudaStream_t copy_stream`: done counter mirror 복사용 side stream
- `plink_ctrl_block`: pinned mapped host memory에 있는 shutdown flag
- `d_done_count`: GPU device memory에 있는 completion counter

`plink_io_worker()`는 thread id를 기준으로 queue pair와 page-cache slot을 고른다.
각 thread는 `read_data()` 또는 `write_data()`를 반복 호출하고, 완료량을 local
counter에 모았다가 1024 I/O마다 `d_done_count`에 `atomicAdd()`한다. shutdown
flag도 같은 주기로 확인해 PCIe host-memory read 비용을 낮춘다.

## Memory Model

현재 hot path는 unified memory를 사용하지 않는다.

- Workload parameters는 kernel launch argument로 값 복사된다. device loop에서
  반복 접근하는 opcode, LBA range, block count, random/sequential 설정은 kernel
  parameter/register path에 있다.
- Completion counter는 순수 device memory다. GPU는 device atomic을 사용하고,
  CPU는 `cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, copy_stream)`로 mirror를
  가져온다.
- Shutdown flag는 `cudaHostAllocMapped` pinned host memory다. CPU는 host store로
  종료를 알리고 GPU는 mapped device pointer를 통해 가끔 읽는다.

이 분리는 CPU polling과 GPU atomic update가 같은 unified-memory page를 두고
page migration을 일으키는 상황을 피하기 위한 것이다.

## Admin Command Bridge

data path와 별도로 NVMe admin command를 넣기 위한 host-side bridge가 있다.

1. fio engine이 `/tmp/nvme-admin.sock` Unix-domain socket을 listen한다.
2. PLINK용으로 빌드된 nvme-cli/libnvme는 `NVME_IOCTL_ADMIN_CMD`를 hook한다.
3. hook은 Linux `struct nvme_passthru_cmd`와 data payload를 socket으로 보낸다.
4. engine helper thread가 요청을 raw 64-byte NVMe SQE로 변환한다.
5. `plink_admin_rpc()`가 BaM의 admin queue reference로 `nvm_raw_rpc()`를 호출한다.

admin data buffer는 4096 byte로 제한된다. server가 자기 DMA buffer의 PRP를
강제로 채우므로 client가 전달한 userspace address는 NVMe command에 직접 쓰이지
않는다.

## Workload Semantics

fio option은 다음 값을 GPU workload로 변환한다.

- `gpu_warps`: 실행할 GPU warp 수. 현재 total GPU threads는 `gpu_warps * 32`.
- `n_queues`: BaM/NVMe queue pair 수. kernel은 `(tid / 32) % n_queues`로 queue를
  고른다.
- `queue_depth`: 각 queue pair depth.
- `nvme_dev`: BaM/libnvm character device path.
- `rw`: `read`는 `PLINK_OP_READ`, 그 외 write workload는 `PLINK_OP_WRITE`.
- `bs`: host에서는 512-byte LBA 단위로 계산한 뒤 launch 시 실제 namespace LBA
  size로 변환한다.
- `size`: GPU가 순환할 LBA range와 fio accounting의 전체 I/O 수 계산에 쓰인다.

현재 kernel은 fio의 per-io offset stream을 소비하지 않는다. random/sequential
패턴은 GPU 내부에서 thread id 기반으로 생성된다.

## Design Constraints

- fio process 하나가 GPU context와 NVMe controller ownership을 갖는다.
- CPU는 per-I/O SQE 생성, MMIO doorbell, CQ polling에 참여하지 않는다.
- fio accounting과 실제 GPU I/O는 done counter를 통해 느슨하게 연결된다.
- admin command bridge는 host-side convenience path이며 throughput data path가
  아니다.
- `cleanup()`은 persistent kernel shutdown 이후 BaM resources를 해제해야 한다.

## Known Gaps

- `plink_gpu_ctx g_ctx`가 file-scope singleton이라 multi-job/multi-device 확장
  전에는 구조 변경이 필요하다.
- `wl.ios_per_thread`는 계산되지만 현재 kernel 종료 조건으로 쓰이지 않는다.
  fio runtime 종료가 host-side shutdown을 통해 제어한다.
- page cache size는 고정 `8192 * 4 KiB`다. workload와 queue topology에 맞춘
  sizing policy는 아직 없다.
- admin socket은 단일 helper thread가 동기 처리한다. admin path는 throughput
  critical path가 아니지만, 종료와 signal handling은 fio lifecycle과 맞아야 한다.
