# parallelink 설계서

## 1. 개요

### 1.1 목적

parallelink는 GPU에서 NVMe SSD로의 직접 I/O(PCIe P2P)를 fio 벤치마크 프레임워크에 통합하는 프로젝트이다.
기존 fio의 I/O 엔진(libaio, io_uring 등)은 모두 CPU 스레드 기반으로 커널을 경유하지만,
parallelink는 GPU persistent kernel이 CPU 개입 없이 자율적으로 NVMe 큐를 조작하여 I/O를 수행한다.

### 1.2 배경

기존 접근 방식의 한계:

| 방식 | 경로 | 한계 |
|------|------|------|
| fio + libaio | CPU → kernel → NVMe | CPU 오버헤드, P2P 불가 |
| fio + libcufile (GDS) | CPU → GDS driver → NVMe → GPU | CPU가 오케스트레이션 |
| libnvm fio_plugin (deprecated) | CPU → userspace NVMe | 동기식, iodepth=1, GPU 미활용 |

parallelink의 접근:

```
GPU Thread → NVMe SQ/CQ (PCIe P2P) → SSD
             CPU 개입 없는 자율 I/O 루프
```

### 1.3 의존성

| 컴포넌트 | 역할 | 소스 |
|----------|------|------|
| fio | 벤치마크 프레임워크, 통계 수집, job 관리 | extern/fio |
| libnvm | userspace NVMe 드라이버, GPU DMA 매핑 | extern/bam |
| CUDA Toolkit | GPU 커널 컴파일 및 런타임 | 시스템 |
| libnvm kernel module | NVMe BAR mmap, DMA 주소 변환 | extern/bam/module |

---

## 2. 아키텍처

### 2.1 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│  fio framework                                               │
│  ├─ job 파싱, 통계 수집, 리포팅                               │
│  └─ ioengine 콜백 호출                                       │
├──────────────┬──────────────────────────────────────────────┤
│  gpu_engine.c │  fio ioengine_ops 구현 (CPU 측)              │
│              │  ├─ init: GPU 초기화 + 커널 런치              │
│              │  ├─ queue/commit: no-op (GPU 자율)            │
│              │  └─ getevents: done_count 폴링               │
├──────────────┼──────────────────────────────────────────────┤
│  gpu_worker  │  GPU persistent kernel (CUDA)                 │
│   .cu        │  ├─ NVMe 커맨드 빌드                          │
│              │  ├─ sq_enqueue → cq_poll → resubmit 루프     │
│              │  └─ done_count, latency 기록                  │
├──────────────┼──────────────────────────────────────────────┤
│  libnvm      │  userspace NVMe 드라이버                      │
│  (extern/bam)│  ├─ lock-free parallel queue (GPU용)         │
│              │  ├─ NVMe 커맨드 API                           │
│              │  └─ Controller/QueuePair 관리                 │
├──────────────┼──────────────────────────────────────────────┤
│  libnvm      │  커널 모듈                                    │
│  module      │  ├─ NVMe BAR0 mmap (doorbell 접근)           │
│              │  ├─ CPU 메모리 DMA 매핑                       │
│              │  └─ GPU 메모리 DMA 매핑 (nvidia_p2p API)      │
├──────────────┴──────────────────────────────────────────────┤
│  Hardware: GPU ←─ PCIe P2P ─→ NVMe SSD                      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 레이어별 역할

#### fio framework (변경 없음)
- job 파일 파싱, 워크로드 파라미터 결정
- ioengine 콜백을 통해 I/O 수행
- BW, IOPS, latency 통계 수집 및 리포팅
- 기존 fio 코드를 수정하지 않고 external engine(.so)으로 로드

#### gpu_engine.c (CPU 측 엔진)
- fio의 `ioengine_ops` 인터페이스 구현
- 엔진 전용 옵션 정의 (gpu_warps, gpu_id, n_queues, nvme_dev)
- `init()`에서 GPU 리소스 초기화 및 persistent kernel 런치
- `getevents()`에서 GPU의 done_count를 폴링하여 fio에 완료 이벤트 전달
- `queue()`/`commit()`은 no-op (GPU가 자율 I/O)

#### gpu_worker.cu (GPU 측 커널)
- persistent kernel로 프로세스 종료 시까지 실행
- 각 GPU 스레드가 독립적으로 submit→complete→resubmit 루프
- libnvm의 lock-free parallel queue API 사용 (sq_enqueue, cq_poll, cq_dequeue)
- GPU clock64()로 per-I/O latency 측정
- atomicAdd로 글로벌 done_count 갱신

#### holder.cu (GPU 메모리 상주 데몬)
- NVMe 컨트롤러 상태와 GPU 메모리를 프로세스 수명과 분리
- CUDA IPC를 통해 fio 프로세스와 GPU 메모리 공유
- 반복 테스트 시 컨트롤러 리셋/큐 재생성 비용 제거

---

## 3. 상세 설계

### 3.1 fio ioengine 인터페이스 매핑

```c
static struct ioengine_ops ioengine = {
    .name       = "parallelink",
    .init       = fio_plink_init,       // GPU 초기화 + 커널 런치
    .queue      = fio_plink_queue,      // no-op (FIO_Q_QUEUED 리턴)
    .commit     = fio_plink_commit,     // no-op
    .getevents  = fio_plink_getevents,  // GPU done_count 폴링
    .event      = fio_plink_event,      // io_u 반환
    .cleanup    = fio_plink_cleanup,    // GPU shutdown
};
```

기존 fio 엔진과의 비교:

```
libaio 엔진:
  queue()     → ring buffer에 적재
  commit()    → io_submit() 시스콜로 커널에 배치 제출
  getevents() → io_getevents()로 완료 수확

parallelink 엔진:
  queue()     → no-op (GPU가 자체적으로 I/O 생성)
  commit()    → no-op (GPU가 직접 SQ doorbell write)
  getevents() → GPU done_count 폴링 (시스콜 없음)
```

### 3.2 엔진 옵션

```c
struct plink_options {
    struct thread_data *td;
    unsigned int gpu_warps;    // GPU warp 수 → 총 스레드 = gpu_warps × 32
    unsigned int gpu_id;       // CUDA device ID
    unsigned int n_queues;     // NVMe QueuePair 수
    char        *nvme_dev;     // libnvm 디바이스 경로
};
```

gpu_warps와 CUDA grid의 매핑:

```
gpu_warps=1    →    32 threads,   1 block
gpu_warps=32   →  1024 threads,   8 blocks
gpu_warps=128  →  4096 threads,  32 blocks
gpu_warps=512  → 16384 threads, 128 blocks
gpu_warps=2048 → 65536 threads, 512 blocks

threads_per_block = 128 (4 warps/block) 고정
n_blocks = (gpu_warps × 32) / 128
```

### 3.3 CPU-GPU 공유 상태

```c
struct plink_shared_state {
    /* GPU kernel 제어 */
    volatile int      shutdown;       // CPU→GPU: 종료 시그널
    volatile uint64_t done_count;     // GPU→CPU: 완료된 I/O 수

    /* 워크로드 파라미터 (init 시 CPU가 1회 설정) */
    uint8_t  opcode;                  // NVMe READ(0x02) / WRITE(0x01)
    int      random;                  // 1=random, 0=sequential
    uint32_t block_size;              // I/O 크기 (bytes)
    uint32_t n_blocks;                // I/O당 블록 수
    uint64_t lba_range;               // 전체 LBA 범위
    uint64_t ios_per_thread;          // 스레드당 I/O 횟수
    int      total_threads;           // 총 GPU 스레드 수

    /* latency 측정 */
    int      record_lat;
    uint64_t *latencies;              // per-thread latency (GPU clock)
};
```

메모리 배치: `cudaMallocManaged()`로 할당하여 CPU/GPU 양측에서 접근 가능.

### 3.4 GPU Persistent Kernel 동작

```
__global__ void plink_io_worker(state, qps, n_queues)
{
    tid = threadIdx.x + blockIdx.x * blockDim.x;
    qp = &qps[tid % n_queues];      // 스레드별 QueuePair 할당

    while (!shutdown && ios_done < ios_per_thread) {

        ① lba = next_lba();          // random 또는 sequential

        ② cmd 빌드
           cid = get_cid(&qp->sq);          // lock-free CID 할당
           nvm_cmd_header(&cmd, cid, op);    // NVMe 커맨드 헤더
           nvm_cmd_data_ptr(&cmd, prp1, prp2); // GPU VRAM 주소
           nvm_cmd_rw_blks(&cmd, lba, n_blocks);

        ③ sq_enqueue(&qp->sq, &cmd);        // SQ에 적재 + doorbell

        ④ cq_poll(&qp->cq, cid);            // CQ 완료 대기
           cq_dequeue(&qp->cq, ...);         // CQ 소비 + doorbell
           put_cid(&qp->sq, cid);            // CID 반환

        ⑤ atomicAdd(&done_count, 1);         // 글로벌 카운터 증가

        → 즉시 ①로 돌아감 (CPU 개입 없음)
    }
}
```

### 3.5 데이터 흐름

```
시간 ──────────────────────────────────────────────────────────→

CPU:  init()  ·····  getevents() ····  getevents() ····  cleanup()
       │               ↑                  ↑                │
       │ 커널 런치      │ done_count       │                │ shutdown=1
       ▼               │ 폴링             │                ▼

GPU:  [sub→cpl→sub→cpl→sub→cpl→sub→cpl→sub→cpl→···→stop]
       ↕       ↕       ↕       ↕       ↕
NVMe: [  I/O  ][  I/O  ][  I/O  ][  I/O  ][  I/O  ]

       GPU 스레드 수천 개가 동시에 위 루프를 수행
```

### 3.6 메모리 배치

```
GPU VRAM
┌──────────────────────────────────────────────────┐
│  SQ 엔트리들 (64B × depth) per QP                │ GPU write, NVMe read
│  CQ 엔트리들 (16B × depth) per QP                │ NVMe write, GPU read
│  sq_tickets (atomic uint64) per QP               │ lock-free 동기화
│  sq_cid[65536] (atomic uint32) per QP            │ CID 관리
│  cq_head_mark[] (atomic) per QP                  │ CQ head 추적
│  cq_pos_locks[] (atomic) per QP                  │ CQ 위치 락
│  데이터 버퍼 (I/O 대상)                            │ NVMe DMA 대상
└──────────────────────────────────────────────────┘
    ↕ PCIe P2P (NVMe 컨트롤러가 GPU VRAM에 직접 DMA)

NVMe BAR0 (mmap to GPU)
┌──────────────────────────────────────────────────┐
│  SQxTDBL (doorbell) → GPU가 직접 write           │
│  CQxHDBL (doorbell) → GPU가 직접 write           │
└──────────────────────────────────────────────────┘
```

SQ/CQ가 GPU VRAM에 있어야 하는 이유:
- GPU 스레드가 sq_enqueue(), cq_poll()을 직접 수행
- NVMe 컨트롤러는 PCIe P2P DMA로 GPU VRAM의 SQ/CQ에 접근
- 커널 메모리(dma_alloc_coherent)에 두면 GPU가 접근 불가

---

## 4. NVMe 디바이스 바인딩

### 4.1 libnvm 커널 모듈의 역할

```
libnvm 커널 모듈 (extern/bam/module/)
├─ NVMe BAR0 레지스터를 userspace에 mmap
│   → GPU/CPU가 doorbell, SQ/CQ에 직접 접근 가능
├─ CPU 메모리 DMA 매핑 (ioctl NVM_MAP_HOST_MEMORY)
│   → dma_map_page()로 물리주소 획득
│   → Admin Queue 등에 사용
└─ GPU 메모리 DMA 매핑 (ioctl NVM_MAP_DEVICE_MEMORY)
    → nvidia_p2p_get_pages() + nvidia_p2p_dma_map_pages()
    → GPU VRAM의 PCIe 물리주소 획득
    → NVMe PRP에 이 주소를 넣으면 SSD↔GPU 직접 전송
```

### 4.2 드라이버 바인딩 절차

```
기존 상태:  NVMe SSD ←bind→ nvme 드라이버 → /dev/nvme0n1 (커널 독점)
전환 후:    NVMe SSD ←bind→ libnvm 드라이버 → /dev/libnvm0 (userspace 제어)
```

기존 nvme 드라이버를 unbind해야 libnvm이 디바이스를 제어할 수 있다.
이 상태에서 /dev/nvme0n1 블록 디바이스는 사라지며, NVMe 큐 조작은
userspace(GPU)에서 직접 수행한다.

### 4.3 초기화 흐름

```
fio_plink_init()
  │
  ├─ open("/dev/libnvm0")                  // libnvm 캐릭터 디바이스
  │
  ├─ nvm_ctrl_init(&ctrl, fd)              // BAR0 mmap + 컨트롤러 식별
  │   └─ mmap(fd, 0) → NVMe 레지스터 접근
  │
  ├─ nvm_raw_ctrl_reset()                  // CC.EN=0 → 대기 → CC.EN=1
  │   └─ Admin SQ/CQ 설정, 컨트롤러 활성화
  │
  ├─ cudaMalloc() × n_queues               // GPU VRAM에 SQ/CQ 할당
  │
  ├─ ioctl(NVM_MAP_DEVICE_MEMORY)          // GPU 메모리 DMA 매핑
  │   └─ nvidia_p2p_get_pages()            // GPU 물리주소 획득
  │
  ├─ nvm_admin_cq_create / sq_create       // NVMe에 I/O 큐 등록
  │
  └─ plink_io_worker<<<grid, block>>>()    // persistent 커널 런치
```

---

## 5. Holder 데몬 설계

### 5.1 문제

프로세스 종료 시 GPU 메모리(SQ/CQ)와 DMA 매핑이 해제되어,
다음 fio 실행 시 NVMe 컨트롤러 full reset + 큐 재생성이 필요하다 (~200-1000ms).

### 5.2 해결: plink-holder

```
plink-holder (상주 프로세스)
├─ NVMe 컨트롤러 초기화 (1회)
├─ GPU VRAM에 SQ/CQ/제어구조체 할당 (cudaMalloc)
├─ DMA 매핑 등록
├─ NVMe I/O 큐 생성
├─ CUDA IPC handle을 PLINK_STATE_PATH에 저장
└─ sleep 루프 (메모리 유지만 담당)

fio 프로세스 (반복 실행)
├─ PLINK_STATE_PATH에서 IPC handle 읽기
├─ cudaIpcOpenMemHandle() → 기존 GPU 메모리 접속
├─ persistent 커널만 런치 (~1ms)
└─ 테스트 완료 후 cudaIpcCloseMemHandle() (메모리 해제 아님)
```

### 5.3 holder 유무에 따른 비용 비교

```
                    holder 없음           holder 사용
────────────────────────────────────────────────────
ctrl reset          ~100-500ms            0
queue create        ~10-50ms × n_qps     0
DMA mapping         ~5-10ms × n_qps      0
커널 런치           ~1ms                  ~1ms
IPC open            -                     ~0.1ms × n_qps
────────────────────────────────────────────────────
합계                ~200-1000ms           ~2ms
```

### 5.4 CUDA IPC 제약

- `cudaMalloc()`한 프로세스가 종료하면 GPU 메모리가 해제됨
- 따라서 holder 프로세스가 GPU 메모리의 수명을 관리해야 함
- holder가 종료되면 모든 IPC handle이 무효화되므로, 다음 fio 실행 시 full init 필요

---

## 6. 기존 구현과의 비교

### 6.1 deprecated/fio/fio_plugin.c와의 차이

libnvm 프로젝트에 포함된 기존 fio 플러그인(deprecated)과의 비교:

```
                    기존 fio_plugin.c       parallelink
────────────────────────────────────────────────────────────
I/O 모델            동기 (FIO_SYNCIO)       비동기 (GPU 자율)
iodepth             1 고정                  gpu_warps × 32
I/O 실행 주체       CPU pthread             GPU persistent kernel
NVMe 큐 조작        CPU (nvm_sq_enqueue)    GPU (sq_enqueue)
데이터 버퍼         CPU 메모리 (SISCI)       GPU VRAM (P2P)
DMA 경로            CPU↔NVMe               GPU↔NVMe (P2P)
상태 유지           reset 옵션 + RPC        holder + CUDA IPC
인터커넥트          SISCI (Dolphin)          PCIe 로컬
CUDA 의존           없음                    필수
```

기존 플러그인은 libnvm의 userspace NVMe 드라이버만 사용하며,
GPU-direct I/O, 대규모 병렬 submit, PCIe P2P 등 핵심 기능을 활용하지 않는다.

### 6.2 fio libaio 엔진과의 구조 비교

```
libaio:
  get_io_u()        → freelist에서 io_u 할당
  fio_libaio_prep() → io_prep_pread/pwrite (iocb 구성)
  fio_libaio_queue()→ ring buffer에 적재, FIO_Q_QUEUED
  fio_libaio_commit()→ io_submit() 시스콜 (커널에 배치 제출)
  fio_libaio_getevents()→ io_getevents() 시스콜 (완료 수확)

parallelink:
  fio_plink_init()  → GPU 커널 런치 (1회, 이후 GPU 자율)
  fio_plink_queue() → no-op
  fio_plink_commit()→ no-op
  fio_plink_getevents()→ GPU done_count 폴링 (시스콜 없음)
```

---

## 7. 파일 구조 및 빌드

### 7.1 프로젝트 구조

```
parallelink/
├── extern/
│   ├── fio/                 fio 벤치마크 (git submodule)
│   └── bam/                 libnvm 라이브러리 (git submodule)
├── include/
│   └── gpu_engine.h         CPU-GPU 공유 구조체, 함수 인터페이스
├── src/
│   ├── gpu_engine.c         fio ioengine_ops (C, CPU 측)
│   ├── gpu_worker.cu        persistent GPU kernel (CUDA)
│   └── holder.cu            GPU 메모리 상주 데몬 (CUDA)
├── docs/
│   └── design.md            본 설계서
├── CMakeLists.txt           빌드 설정
└── README.md                프로젝트 가이드
```

### 7.2 빌드 산출물

| 산출물 | 타입 | 설명 |
|--------|------|------|
| `parallelink.so` | shared library | fio external ioengine |
| `plink-holder` | executable | GPU 메모리 상주 데몬 |

### 7.3 빌드 의존성

```
parallelink.so
├── gpu_engine.c  ──→ fio headers (config-host.h, fio.h, optgroup.h)
│                 ──→ gpu_engine.h
└── gpu_worker.cu ──→ gpu_engine.h
                  ──→ libnvm headers (nvm_parallel_queue.h, queue.h, ...)
                  ──→ CUDA runtime

plink-holder
└── holder.cu     ──→ gpu_engine.h
                  ──→ libnvm headers
                  ──→ CUDA runtime
```

---

## 8. 향후 구현 항목

### Phase 1: 기본 동작
- [ ] gpu_worker.cu: libnvm Controller/QueuePair 초기화 구현
- [ ] gpu_worker.cu: GPU 데이터 버퍼 할당 + PRP 주소 설정
- [ ] gpu_worker.cu: persistent kernel 런치 및 I/O 루프 검증
- [ ] gpu_engine.c: getevents()에서 latency 정보를 fio io_u에 반영

### Phase 2: Holder 데몬
- [ ] holder.cu: NVMe 컨트롤러 + 큐 초기화 구현
- [ ] holder.cu: CUDA IPC handle 저장/로드
- [ ] gpu_engine.c: holder 모드 감지 및 IPC 접속 로직

### Phase 3: 성능 최적화
- [ ] warp 단위 배치 submit 최적화
- [ ] multi-GPU 지원
- [ ] fio latency percentile과 GPU clock64() 연동
- [ ] time_based 모드에서 runtime 제어

### Phase 4: 테스트 및 검증
- [ ] 단일 GPU + 단일 NVMe 기본 동작 테스트
- [ ] fio libaio 대비 성능 비교
- [ ] gpu_warps 스케일링 테스트
- [ ] 장시간 안정성 테스트
