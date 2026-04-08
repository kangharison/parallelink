================================================================================
parallelink 구현 명세서
대상: Claude Code (개발 에이전트)
참조: docs/design.md
================================================================================

이 문서는 parallelink 프로젝트의 구현 명세이다.
design.md의 설계를 기반으로, 각 파일별 구조체, 함수, 동작을
Claude Code가 바로 구현할 수 있는 수준으로 기술한다.

Phase 1 (기본 동작)을 우선 구현 대상으로 한다.
Phase 2 이후는 Phase 1 완료 후 진행한다.

================================================================================
목차
================================================================================

1. 프로젝트 구조 및 빌드
2. include/gpu_engine.h - 공유 구조체 및 인터페이스
3. src/gpu_engine.c - fio ioengine 구현 (CPU 측)
4. src/gpu_worker.cu - GPU persistent kernel (CUDA)
5. CMakeLists.txt - 빌드 설정
6. extern 의존성 설정
7. 제약사항 및 주의사항
8. 테스트 시나리오

================================================================================
1. 프로젝트 구조 및 빌드
================================================================================

디렉토리 레이아웃:

parallelink/
├── extern/
│   ├── fio/                     # git submodule: https://github.com/axboe/fio
│   └── bam/                     # git submodule: https://github.com/ZaidQureshi/bam
├── include/
│   └── gpu_engine.h             # CPU-GPU 공유 구조체, 상수, 함수 선언
├── src/
│   ├── gpu_engine.c             # fio ioengine_ops 구현 (순수 C, CPU 측)
│   ├── gpu_worker.cu            # GPU persistent kernel + 호스트 래퍼 (CUDA)
│   └── holder.cu                # GPU 메모리 상주 데몬 (Phase 3, 지금은 스텁)
├── docs/
│   └── design.md                # 설계서
├── CMakeLists.txt               # 빌드 설정
└── README.md

빌드 산출물:
  - parallelink.so       : fio external ioengine 공유 라이브러리
  - plink-holder         : GPU 메모리 상주 데몬 (Phase 3)

실행 방법:
  $ fio --ioengine=external:./parallelink.so --name=test \
        --gpu_warps=4 --iodepth=16 --nvme_dev=/dev/libnvm0 \
        --rw=randread --bs=4k --runtime=10 --time_based

================================================================================
2. include/gpu_engine.h
================================================================================

이 파일은 C와 CUDA 모두에서 include 가능해야 한다.
#ifdef __cplusplus / extern "C" 가드를 사용한다.

--------------------------------------------------------------------------------
2.1 상수 정의
--------------------------------------------------------------------------------

#define PLINK_MAX_WARPS         128     // 최대 warp 수 (= 최대 SQ 수)
#define PLINK_THREADS_PER_WARP  32      // NVIDIA warp 크기
#define PLINK_THREADS_PER_BLOCK 128     // CUDA block당 스레드 (= 4 warps)
#define PLINK_MAX_IODEPTH       1024    // warp당 최대 outstanding I/O
#define PLINK_DEFAULT_IODEPTH   16      // 기본 iodepth
#define PLINK_DEFAULT_GPU_WARPS 4       // 기본 warp 수
#define PLINK_NVME_BLOCK_SIZE   512     // NVMe 논리 블록 크기 (bytes)
#define PLINK_STATE_PATH        "/tmp/plink_state" // holder IPC 경로

// NVMe opcodes
#define NVME_OPC_READ           0x02
#define NVME_OPC_WRITE          0x01

--------------------------------------------------------------------------------
2.2 struct plink_shared_state
--------------------------------------------------------------------------------

이 구조체는 cudaMallocManaged()로 할당한다.
CPU(gpu_engine.c)와 GPU(gpu_worker.cu) 양측에서 접근한다.

struct plink_shared_state {
    // ── GPU kernel 제어 ──
    volatile int      shutdown;          // CPU→GPU: 1이면 커널 종료
    volatile uint64_t done_count;        // GPU→CPU: 완료된 I/O 총 수 (atomicAdd)

    // ── 실행 모드 ──
    int      time_based;                 // 1=시간 기반, 0=크기 기반
    uint64_t deadline_ns;                // time_based=1: GPU globaltimer 종료 시각 (ns)
    uint64_t ios_per_warp;               // time_based=0: warp당 수행할 I/O 횟수

    // ── 워크로드 파라미터 (init 시 CPU가 1회 설정, 이후 read-only) ──
    uint8_t  opcode;                     // NVME_OPC_READ(0x02) 또는 NVME_OPC_WRITE(0x01)
    int      random;                     // 1=random, 0=sequential
    uint32_t block_size;                 // I/O 크기 (bytes, 예: 4096)
    uint32_t n_blocks;                   // I/O당 NVMe 블록 수 (= block_size / 512)
    uint64_t lba_range;                  // 전체 LBA 범위 (I/O 주소 공간)
    int      total_warps;                // 총 warp 수 (= SQ 수 = gpu_warps)

    // ── async I/O 제어 ──
    uint32_t iodepth;                    // warp당 최대 outstanding I/O 수

    // ── latency 측정 ──
    int      record_lat;                 // 1이면 latency 기록
    uint64_t gpu_clock_freq;             // GPU clock 주파수 (Hz), ns 변환용

    // ── per-warp 통계 (GPU가 atomicAdd로 갱신) ──
    volatile uint64_t warp_submit_count[PLINK_MAX_WARPS];
    volatile uint64_t warp_complete_count[PLINK_MAX_WARPS];
};

구현 주의사항:
  - volatile 필드는 CPU/GPU 간 가시성 보장을 위해 필수
  - done_count는 GPU에서 atomicAdd, CPU에서 volatile read
  - shutdown은 CPU에서 write, GPU에서 volatile read
  - deadline_ns는 init 시 GPU globaltimer 현재값 + runtime_sec * 1e9로 계산
    → GPU에서 globaltimer를 읽으려면 커널 내에서 측정해야 하므로,
       init 시 더미 커널을 런치하여 globaltimer 값을 가져온 뒤 deadline 설정

--------------------------------------------------------------------------------
2.3 struct plink_engine_data
--------------------------------------------------------------------------------

fio thread_data의 io_ops_data에 저장되는 엔진 내부 상태.
CPU 측에서만 사용. gpu_engine.c에서 malloc으로 할당.

struct plink_engine_data {
    // ── 옵션 ──
    unsigned int gpu_warps;              // warp 수 (= SQ 수)
    unsigned int gpu_id;                 // CUDA device ID
    unsigned int iodepth;                // warp당 iodepth
    char        *nvme_dev;               // libnvm 디바이스 경로 (예: "/dev/libnvm0")

    // ── GPU 리소스 ──
    struct plink_shared_state *state;    // cudaMallocManaged된 공유 상태
    void        *gpu_data_bufs;          // GPU VRAM 데이터 버퍼 (전체)
    cudaStream_t stream;                 // CUDA 스트림

    // ── libnvm 리소스 ──
    int          nvme_fd;                // /dev/libnvm0 fd
    nvm_ctrl_t  *ctrl;                   // libnvm 컨트롤러 핸들
    // QueuePair 배열 포인터 (GPU VRAM에 할당)
    // libnvm의 nvm_queue_t 또는 parallelink 래퍼

    // ── fio 연동 ──
    uint64_t     prev_done_count;        // 이전 getevents 시점의 done_count
    struct io_u **events;                // getevents에서 반환할 io_u 배열

    // ── 커널 런치 상태 ──
    int          kernel_launched;        // 1이면 커널 실행 중
};

--------------------------------------------------------------------------------
2.4 함수 선언 (gpu_worker.cu에서 구현, gpu_engine.c에서 호출)
--------------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

// GPU 리소스 초기화: CUDA device 선택, 공유 상태 할당
int plink_gpu_init(struct plink_engine_data *ed);

// NVMe 컨트롤러 초기화 + I/O 큐 생성 + DMA 매핑
int plink_nvme_init(struct plink_engine_data *ed);

// GPU persistent kernel 런치 (1회)
int plink_launch_kernel(struct plink_engine_data *ed);

// GPU에서 현재 globaltimer 값을 가져오는 유틸리티 (deadline 설정용)
uint64_t plink_get_gpu_timestamp(int gpu_id);

// GPU clock 주파수 조회
uint64_t plink_get_gpu_clock_freq(int gpu_id);

// 정리: 커널 종료 대기 + 리소스 해제
void plink_cleanup(struct plink_engine_data *ed);

#ifdef __cplusplus
}
#endif

================================================================================
3. src/gpu_engine.c - fio ioengine 구현
================================================================================

순수 C 파일. fio 헤더와 gpu_engine.h를 include한다.
CUDA 호출은 직접 하지 않고, gpu_worker.cu의 래퍼 함수를 호출한다.

--------------------------------------------------------------------------------
3.1 fio 옵션 정의
--------------------------------------------------------------------------------

fio의 FIO_OPT 매크로를 사용하여 엔진 전용 옵션을 정의한다.

static struct fio_option options[] = {
    {
        .name   = "gpu_warps",
        .lname  = "GPU warp count",
        .type   = FIO_OPT_INT,
        .off1   = offsetof(struct plink_options, gpu_warps),
        .def    = "4",
        .help   = "Number of GPU warps (1 warp = 1 NVMe SQ, 32 threads)",
        .minval = 1,
        .maxval = PLINK_MAX_WARPS,
        .category = FIO_OPT_C_ENGINE,
    },
    {
        .name   = "gpu_id",
        .lname  = "CUDA GPU device ID",
        .type   = FIO_OPT_INT,
        .off1   = offsetof(struct plink_options, gpu_id),
        .def    = "0",
        .help   = "CUDA device index",
        .category = FIO_OPT_C_ENGINE,
    },
    {
        .name   = "iodepth",
        .lname  = "I/O depth per warp",
        .type   = FIO_OPT_INT,
        .off1   = offsetof(struct plink_options, iodepth),
        .def    = "16",
        .help   = "Outstanding I/O count per warp (async)",
        .minval = 1,
        .maxval = PLINK_MAX_IODEPTH,
        .category = FIO_OPT_C_ENGINE,
    },
    {
        .name   = "nvme_dev",
        .lname  = "libnvm device path",
        .type   = FIO_OPT_STR_STORE,
        .off1   = offsetof(struct plink_options, nvme_dev),
        .def    = "/dev/libnvm0",
        .help   = "libnvm character device (e.g. /dev/libnvm0)",
        .category = FIO_OPT_C_ENGINE,
    },
    { .name = NULL },
};

--------------------------------------------------------------------------------
3.2 fio_plink_init()
--------------------------------------------------------------------------------

호출 시점: fio 엔진 초기화 (job 시작 시 1회)
동작:
  1. plink_engine_data 할당 및 옵션 복사
  2. plink_gpu_init() 호출 → CUDA device 선택, 공유 상태 할당
  3. 공유 상태에 fio 워크로드 파라미터 설정:
     - opcode: td->o.td_ddir (DDIR_READ → 0x02, DDIR_WRITE → 0x01)
     - random: td_random(td)
     - block_size: td->o.min_bs[DDIR_READ] (또는 DDIR_WRITE)
     - n_blocks: block_size / PLINK_NVME_BLOCK_SIZE
     - lba_range: td->o.file_size_high / PLINK_NVME_BLOCK_SIZE
       (또는 NVMe namespace 크기를 identify로 조회)
     - time_based: td->o.time_based
     - iodepth: 옵션에서 읽은 값
     - total_warps: gpu_warps
  4. plink_nvme_init() 호출 → NVMe 컨트롤러 초기화, I/O 큐 생성
     - SSD Get Feature (Number of Queues)로 최대 SQ 수 확인
     - gpu_warps > max_sq이면 gpu_warps = max_sq로 제한 + 경고 출력
  5. 실행 모드 설정:
     - time_based=1: deadline_ns = plink_get_gpu_timestamp() + runtime * 1e9
     - time_based=0: ios_per_warp = total_io_size / (block_size * total_warps)
  6. plink_launch_kernel() 호출 → persistent kernel 런치 (비동기, 즉시 리턴)
  7. ed->kernel_launched = 1
  8. return 0

에러 처리:
  - CUDA 초기화 실패 → log_err() + return 1
  - libnvm 디바이스 open 실패 → log_err() + return 1
  - NVMe 컨트롤러 리셋 실패 → log_err() + return 1

--------------------------------------------------------------------------------
3.3 fio_plink_queue()
--------------------------------------------------------------------------------

호출 시점: fio가 I/O를 큐잉할 때 (매 I/O)
동작: no-op. GPU가 자율적으로 I/O를 생성하므로 fio의 io_u를 사용하지 않음.
반환: FIO_Q_QUEUED

구현:
  static enum fio_q_status fio_plink_queue(struct thread_data *td,
                                            struct io_u *io_u)
  {
      return FIO_Q_QUEUED;
  }

--------------------------------------------------------------------------------
3.4 fio_plink_commit()
--------------------------------------------------------------------------------

호출 시점: fio가 큐잉된 I/O를 제출할 때
동작: no-op. GPU가 직접 SQ doorbell을 write하므로 CPU 제출 불필요.
반환: 0

--------------------------------------------------------------------------------
3.5 fio_plink_getevents()
--------------------------------------------------------------------------------

호출 시점: fio가 완료된 I/O를 수확할 때
동작:
  1. state->done_count를 volatile read
  2. 새로 완료된 수 = done_count - prev_done_count
  3. min_events 이상이 될 때까지 spin-wait (usleep(1) 삽입)
  4. prev_done_count 갱신
  5. 완료된 수만큼 io_u를 events 배열에 채움
  6. 완료된 수 반환

구현 주의:
  - fio의 min_events가 0이면 non-blocking으로 즉시 반환
  - timeout_usec이 주어지면 해당 시간 초과 시 현재까지 수확한 것 반환
  - io_u의 resid = 0, error = 0으로 설정

static int fio_plink_getevents(struct thread_data *td,
                                unsigned int min_events,
                                unsigned int max_events,
                                const struct timespec *timeout)
{
    struct plink_engine_data *ed = td->io_ops_data;
    uint64_t current = ed->state->done_count;
    uint64_t new_completions = current - ed->prev_done_count;

    // min_events 대기
    while (new_completions < min_events) {
        usleep(1);
        current = ed->state->done_count;
        new_completions = current - ed->prev_done_count;
        // timeout 체크 (생략 가능, Phase 1에서는 단순 spin)
    }

    unsigned int ret = (new_completions > max_events)
                       ? max_events : (unsigned int)new_completions;
    ed->prev_done_count += ret;

    for (unsigned int i = 0; i < ret; i++) {
        ed->events[i] = io_u_alloc_or_reuse(td); // 또는 사전 할당된 io_u
        ed->events[i]->resid = 0;
        ed->events[i]->error = 0;
    }

    return ret;
}

주의: fio의 io_u 관리와 GPU 자율 I/O 모델 사이에 괴리가 있음.
GPU가 I/O를 자체 생성하므로 fio의 io_u는 "가상" 완료 이벤트 역할만 함.
Phase 1에서는 done_count 기반으로 단순 처리하고,
Phase 4에서 fio latency 연동 시 정교화함.

--------------------------------------------------------------------------------
3.6 fio_plink_event()
--------------------------------------------------------------------------------

호출 시점: fio가 getevents()에서 반환된 이벤트의 io_u를 요청할 때
동작: events[event] 반환

static struct io_u *fio_plink_event(struct thread_data *td, int event)
{
    struct plink_engine_data *ed = td->io_ops_data;
    return ed->events[event];
}

--------------------------------------------------------------------------------
3.7 fio_plink_cleanup()
--------------------------------------------------------------------------------

호출 시점: fio 엔진 종료 시
동작:
  1. state->shutdown = 1 (GPU 커널에 종료 시그널)
  2. cudaDeviceSynchronize() 또는 cudaStreamSynchronize()로 커널 종료 대기
  3. 통계 출력: total done_count, per-warp submit/complete count
  4. plink_cleanup() 호출 → GPU 메모리/NVMe 리소스 해제
  5. ed free

--------------------------------------------------------------------------------
3.8 ioengine_ops 등록
--------------------------------------------------------------------------------

static struct ioengine_ops ioengine = {
    .name           = "parallelink",
    .version        = FIO_IOOPS_VERSION,
    .flags          = FIO_NOIO,     // fio가 자체 I/O를 발행하지 않음
    .init           = fio_plink_init,
    .queue          = fio_plink_queue,
    .commit         = fio_plink_commit,
    .getevents      = fio_plink_getevents,
    .event          = fio_plink_event,
    .cleanup        = fio_plink_cleanup,
    .options        = options,
    .option_struct_size = sizeof(struct plink_options),
};

static void fio_init fio_plink_register(void)
{
    register_ioengine(&ioengine);
}

static void fio_exit fio_plink_unregister(void)
{
    unregister_ioengine(&ioengine);
}

================================================================================
4. src/gpu_worker.cu - GPU persistent kernel + 호스트 래퍼
================================================================================

CUDA 파일. nvcc로 컴파일.
GPU 디바이스 코드(커널)와 호스트 래퍼 함수를 모두 포함한다.

--------------------------------------------------------------------------------
4.1 호스트 래퍼 함수 구현
--------------------------------------------------------------------------------

이 함수들은 extern "C"로 선언하여 gpu_engine.c에서 호출 가능하게 한다.

---- plink_gpu_init() ----

int plink_gpu_init(struct plink_engine_data *ed)
{
    동작:
    1. cudaSetDevice(ed->gpu_id)
    2. cudaMallocManaged(&ed->state, sizeof(plink_shared_state))
    3. memset(ed->state, 0, sizeof(plink_shared_state))
    4. cudaStreamCreate(&ed->stream)
    5. ed->state->gpu_clock_freq = plink_get_gpu_clock_freq(ed->gpu_id)
    6. events 배열 할당: ed->events = calloc(max_events, sizeof(io_u*))
    7. return 0 (성공) 또는 -1 (실패)
}

---- plink_nvme_init() ----

int plink_nvme_init(struct plink_engine_data *ed)
{
    동작:
    1. ed->nvme_fd = open(ed->nvme_dev, O_RDWR)
    2. nvm_ctrl_init(&ed->ctrl, ed->nvme_fd)
       → BAR0 mmap, CAP/VS 읽기
    3. nvm_raw_ctrl_reset(ed->ctrl)
       → CC.EN=0 → CSTS.RDY=0 대기 → Admin SQ/CQ 설정 → CC.EN=1 → CSTS.RDY=1 대기
    4. NVMe Identify Controller (Admin 커맨드)
       → MQES (Maximum Queue Entries Supported) 확인
    5. NVMe Get Feature - Number of Queues (Feature ID 0x07)
       → 최대 I/O SQ 수 확인
       → ed->gpu_warps = min(ed->gpu_warps, max_io_sq)
       → 제한 시 stderr에 경고 출력
    6. for (i = 0; i < ed->gpu_warps; i++):
       a. cudaMalloc(&sq_mem, sq_entry_size * queue_depth)
          → GPU VRAM에 SQ 할당
       b. cudaMalloc(&cq_mem, cq_entry_size * queue_depth)
          → GPU VRAM에 CQ 할당
       c. ioctl(ed->nvme_fd, NVM_MAP_DEVICE_MEMORY, ...)
          → GPU VRAM의 PCIe 물리 주소 획득 (nvidia_p2p_get_pages)
       d. NVMe Admin: Create I/O CQ (opcode 0x05)
          → CQ의 GPU 물리 주소 등록
       e. NVMe Admin: Create I/O SQ (opcode 0x01)
          → SQ의 GPU 물리 주소 등록, 연결 CQ ID 지정
    7. 데이터 버퍼 할당:
       total_buf_size = gpu_warps * iodepth * block_size
       cudaMalloc(&ed->gpu_data_bufs, total_buf_size)
       ioctl(NVM_MAP_DEVICE_MEMORY, ...) → 데이터 버퍼 DMA 매핑
    8. NVMe BAR0 doorbell 영역을 GPU 접근 가능하게 매핑:
       doorbell_uva = mmap(nvme_fd, BAR0_offset + 0x1000, ...)
       cudaHostRegister(doorbell_uva, size, cudaHostRegisterIoMemory)
       cudaHostGetDevicePointer(&doorbell_gpu_ptr, doorbell_uva, 0)
    9. QueuePair 구조체 배열을 GPU VRAM에 할당하고 초기화:
       각 QP에 sq_ptr, cq_ptr, sq_doorbell, cq_doorbell,
       data_buf_base, depth, phase=1 등 설정
   10. return 0
}

libnvm API 사용 시:
  - libnvm의 기존 Controller, QueuePair API를 가능한 한 재사용
  - 단, BaM의 nvm_parallel_queue.cuh(GPU용 lock-free 큐)는
    async 모델에 맞게 수정이 필요할 수 있음
  - 특히 cq_poll이 blocking인 부분을 non-blocking으로 변경해야 함

---- plink_launch_kernel() ----

int plink_launch_kernel(struct plink_engine_data *ed)
{
    동작:
    1. CUDA grid/block 계산:
       threads_per_block = PLINK_THREADS_PER_BLOCK  // 128
       n_blocks = ed->gpu_warps / (PLINK_THREADS_PER_BLOCK / PLINK_THREADS_PER_WARP)
                = ed->gpu_warps / 4
       if (n_blocks == 0) n_blocks = 1
    2. 커널 런치 (비동기):
       plink_io_worker<<<n_blocks, threads_per_block, 0, ed->stream>>>(
           ed->state,
           qp_array_gpu_ptr,
           ed->gpu_warps
       );
    3. cudaGetLastError() 확인
    4. return 0
}

주의: 커널 런치 후 cudaDeviceSynchronize()를 호출하지 않는다.
      커널은 persistent로 실행되며, cleanup 시에만 sync한다.

---- plink_get_gpu_timestamp() ----

uint64_t plink_get_gpu_timestamp(int gpu_id)
{
    동작:
    1. 작은 커널을 런치하여 %%globaltimer 값을 읽어옴
    2. cudaDeviceSynchronize()
    3. 읽어온 ns 값 반환

    __global__ void _get_timestamp(uint64_t *out) {
        uint64_t t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        *out = t;
    }
}

---- plink_get_gpu_clock_freq() ----

uint64_t plink_get_gpu_clock_freq(int gpu_id)
{
    동작:
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    return (uint64_t)prop.clockRate * 1000; // kHz → Hz
}

---- plink_cleanup() ----

void plink_cleanup(struct plink_engine_data *ed)
{
    동작:
    1. ed->state->shutdown = 1
    2. cudaStreamSynchronize(ed->stream) → 커널 종료 대기
    3. NVMe Admin: Delete I/O SQ/CQ (각 큐에 대해)
    4. GPU 메모리 해제: cudaFree(sq_mem), cudaFree(cq_mem), cudaFree(data_bufs)
    5. cudaFree(ed->state)
    6. cudaStreamDestroy(ed->stream)
    7. close(ed->nvme_fd)
    8. nvm_ctrl_free(ed->ctrl)
}

--------------------------------------------------------------------------------
4.2 GPU 디바이스 코드 - QueuePair 구조체
--------------------------------------------------------------------------------

GPU VRAM에 상주하는 per-warp QueuePair 정보:

struct plink_qp {
    // SQ 관련
    volatile uint8_t  *sq;              // SQ 엔트리 배열 (GPU VRAM, 64B per entry)
    volatile uint32_t *sq_doorbell;     // SQ tail doorbell (NVMe BAR0, GPU 매핑)
    uint32_t           sq_tail;         // 현재 SQ tail
    uint32_t           sq_depth;        // SQ 깊이

    // CQ 관련
    volatile uint8_t  *cq;              // CQ 엔트리 배열 (GPU VRAM, 16B per entry)
    volatile uint32_t *cq_doorbell;     // CQ head doorbell (NVMe BAR0, GPU 매핑)
    uint32_t           cq_head;         // 현재 CQ head
    uint8_t            cq_phase;        // 현재 phase bit (1로 시작)
    uint32_t           cq_depth;        // CQ 깊이

    // 데이터 버퍼
    uint8_t           *data_buf_base;   // per-warp 데이터 버퍼 시작 주소 (GPU VRAM)
    uint64_t          *data_buf_phys;   // per-slot DMA 물리 주소 배열

    // CID 관리
    uint32_t           next_cid;        // 다음 사용할 Command ID
    uint32_t           cid_mask;        // depth - 1 (modulo용)

    // per-slot 상태 (iodepth 크기 배열)
    uint64_t          *slot_submit_time; // per-slot submit 시각 (clock64)
    int               *slot_active;      // per-slot active 플래그
};

--------------------------------------------------------------------------------
4.3 GPU 디바이스 코드 - NVMe 커맨드 빌드
--------------------------------------------------------------------------------

__device__ void plink_build_nvme_cmd(
    uint8_t *sqe,           // 64-byte SQ entry 포인터
    uint8_t opcode,         // NVME_OPC_READ 또는 NVME_OPC_WRITE
    uint16_t cid,           // Command ID
    uint32_t nsid,          // Namespace ID (보통 1)
    uint64_t lba,           // Starting LBA
    uint32_t n_blocks,      // 블록 수 (0-based: 실제 블록 수 - 1)
    uint64_t prp1,          // PRP1 (데이터 버퍼 물리 주소)
    uint64_t prp2           // PRP2 (4KB 초과 시)
)
{
    // 64-byte SQE를 0으로 초기화
    for (int i = 0; i < 16; i++)
        ((uint32_t*)sqe)[i] = 0;

    // CDW0: opcode + CID
    ((uint32_t*)sqe)[0] = opcode | ((uint32_t)cid << 16);

    // CDW1: NSID
    ((uint32_t*)sqe)[1] = nsid;

    // PRP1 (offset 24, 8 bytes)
    ((uint64_t*)(sqe + 24))[0] = prp1;

    // PRP2 (offset 32, 8 bytes)
    ((uint64_t*)(sqe + 32))[0] = prp2;

    // CDW10: Starting LBA (lower 32 bits)
    ((uint32_t*)sqe)[10] = (uint32_t)(lba & 0xFFFFFFFF);

    // CDW11: Starting LBA (upper 32 bits)
    ((uint32_t*)sqe)[11] = (uint32_t)(lba >> 32);

    // CDW12: Number of Logical Blocks (0-based)
    ((uint32_t*)sqe)[12] = n_blocks - 1;
}

--------------------------------------------------------------------------------
4.4 GPU 디바이스 코드 - SQ/CQ 조작
--------------------------------------------------------------------------------

__device__ void plink_sq_submit(struct plink_qp *qp, uint8_t *cmd)
{
    // 1) 현재 tail 위치에 커맨드 복사 (64 bytes)
    uint32_t slot = qp->sq_tail;
    uint8_t *dst = (uint8_t*)qp->sq + slot * 64;
    for (int i = 0; i < 16; i++)
        ((uint32_t*)dst)[i] = ((uint32_t*)cmd)[i];
    __threadfence();

    // 2) tail 전진
    qp->sq_tail = (slot + 1) % qp->sq_depth;
    // doorbell은 batching을 위해 별도 호출
}

__device__ void plink_sq_ring_doorbell(struct plink_qp *qp)
{
    // SQ tail doorbell write (PCIe MMIO → NVMe BAR0)
    *(qp->sq_doorbell) = qp->sq_tail;
    __threadfence_system();
}

__device__ int plink_cq_poll(struct plink_qp *qp,
                              uint16_t *out_cid,
                              uint16_t *out_status)
{
    // CQ head 위치의 CQE 확인
    uint8_t *cqe = (uint8_t*)qp->cq + qp->cq_head * 16;
    uint16_t status_field = ((uint16_t*)(cqe + 14))[0];
    uint8_t phase = status_field & 0x1;

    // phase bit가 현재 기대값과 다르면 → 아직 완료 없음
    if (phase != qp->cq_phase)
        return 0;  // non-blocking: 즉시 리턴

    // 완료 확인
    *out_cid = ((uint16_t*)(cqe + 12))[0];
    *out_status = (status_field >> 1) & 0x7FF;

    // CQ head 전진
    qp->cq_head = (qp->cq_head + 1) % qp->cq_depth;
    if (qp->cq_head == 0)
        qp->cq_phase ^= 1;

    return 1;  // 1개 완료 수확
}

__device__ void plink_cq_ring_doorbell(struct plink_qp *qp)
{
    // CQ head doorbell write
    *(qp->cq_doorbell) = qp->cq_head;
    __threadfence_system();
}

중요: plink_cq_poll()은 non-blocking이다.
      완료가 없으면 0을 반환하고 즉시 리턴한다.
      이것이 BaM의 cq_poll(blocking)과의 핵심 차이이다.

--------------------------------------------------------------------------------
4.5 GPU 디바이스 코드 - LBA 생성
--------------------------------------------------------------------------------

__device__ uint64_t plink_next_lba(
    struct plink_shared_state *state,
    int warp_id,
    uint64_t *seq_counter,   // per-warp sequential counter
    uint64_t seed            // per-warp random seed
)
{
    if (state->random) {
        // xorshift64 기반 PRNG
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return seed % state->lba_range;
    } else {
        uint64_t lba = (*seq_counter) * state->n_blocks;
        (*seq_counter)++;
        if (lba >= state->lba_range)
            *seq_counter = 0;
        return lba % state->lba_range;
    }
}

--------------------------------------------------------------------------------
4.6 GPU 디바이스 코드 - 종료 조건
--------------------------------------------------------------------------------

__device__ bool plink_should_stop(
    struct plink_shared_state *state,
    uint64_t warp_ios_done
)
{
    if (state->shutdown)
        return true;

    if (state->time_based) {
        uint64_t now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        return (now >= state->deadline_ns);
    } else {
        return (warp_ios_done >= state->ios_per_warp);
    }
}

--------------------------------------------------------------------------------
4.7 GPU 디바이스 코드 - 메인 커널
--------------------------------------------------------------------------------

__global__ void plink_io_worker(
    struct plink_shared_state *state,
    struct plink_qp *qps,       // GPU VRAM의 QP 배열
    int num_warps
)
{
    int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = global_tid / 32;
    int lane_id = global_tid % 32;

    // 범위 체크
    if (warp_id >= num_warps) return;

    struct plink_qp *qp = &qps[warp_id];
    uint64_t seq_counter = warp_id;  // sequential용
    uint64_t seed = (uint64_t)warp_id * 6364136223846793005ULL + 1442695040888963407ULL;
    uint64_t warp_ios_done = 0;
    uint32_t outstanding = 0;
    uint32_t next_slot = 0;

    // ═══ Phase 1: Initial Fill ═══
    // lane 0만 submit을 담당 (Phase 1 단순 구현)
    // Phase 4에서 warp 내 역할 분담 최적화
    if (lane_id == 0) {
        while (outstanding < state->iodepth && !plink_should_stop(state, warp_ios_done)) {
            uint16_t cid = next_slot;
            uint64_t lba = plink_next_lba(state, warp_id, &seq_counter, seed);
            uint64_t prp1 = qp->data_buf_phys[next_slot];

            uint8_t cmd[64];
            plink_build_nvme_cmd(cmd, state->opcode, cid, 1, lba,
                                  state->n_blocks, prp1, 0);
            plink_sq_submit(qp, cmd);
            qp->slot_submit_time[next_slot] = clock64();
            qp->slot_active[next_slot] = 1;

            outstanding++;
            next_slot = (next_slot + 1) % state->iodepth;
        }
        plink_sq_ring_doorbell(qp);  // 배치 doorbell
    }

    // warp 동기화 (모든 lane이 Phase 2에 진입)
    __syncwarp(0xFFFFFFFF);

    // ═══ Phase 2: Steady-State Loop ═══
    while (!plink_should_stop(state, warp_ios_done)) {

        // ── Step A: CQ Poll (lane 0이 담당, Phase 1 단순 구현) ──
        if (lane_id == 0) {
            uint16_t cid, status;
            int new_submits = 0;

            // non-blocking poll: 완료된 것만 수확
            while (plink_cq_poll(qp, &cid, &status)) {
                // latency 기록
                if (state->record_lat) {
                    uint64_t lat = clock64() - qp->slot_submit_time[cid];
                    // latency 배열에 기록 (간단 구현: per-warp 누적)
                }
                qp->slot_active[cid] = 0;
                outstanding--;
                warp_ios_done++;
                atomicAdd((unsigned long long*)&state->done_count, 1ULL);
                atomicAdd((unsigned long long*)&state->warp_complete_count[warp_id], 1ULL);
            }
            // CQ doorbell 업데이트
            plink_cq_ring_doorbell(qp);

            // ── Step B: Refill ──
            while (outstanding < state->iodepth && !plink_should_stop(state, warp_ios_done)) {
                // 비활성 슬롯 찾기
                while (qp->slot_active[next_slot])
                    next_slot = (next_slot + 1) % state->iodepth;

                uint16_t cid = next_slot;
                uint64_t lba = plink_next_lba(state, warp_id, &seq_counter, seed);
                uint64_t prp1 = qp->data_buf_phys[next_slot];

                uint8_t cmd[64];
                plink_build_nvme_cmd(cmd, state->opcode, cid, 1, lba,
                                      state->n_blocks, prp1, 0);
                plink_sq_submit(qp, cmd);
                qp->slot_submit_time[next_slot] = clock64();
                qp->slot_active[next_slot] = 1;

                outstanding++;
                new_submits++;
                next_slot = (next_slot + 1) % state->iodepth;

                atomicAdd((unsigned long long*)&state->warp_submit_count[warp_id], 1ULL);
            }

            // ── Step C: Doorbell Batching ──
            if (new_submits > 0) {
                plink_sq_ring_doorbell(qp);
            }
        }

        // warp 동기화
        __syncwarp(0xFFFFFFFF);
    }

    // ═══ Drain: 남은 outstanding I/O 완료 대기 ═══
    if (lane_id == 0) {
        while (outstanding > 0) {
            uint16_t cid, status;
            if (plink_cq_poll(qp, &cid, &status)) {
                qp->slot_active[cid] = 0;
                outstanding--;
                warp_ios_done++;
                atomicAdd((unsigned long long*)&state->done_count, 1ULL);
                atomicAdd((unsigned long long*)&state->warp_complete_count[warp_id], 1ULL);
            }
        }
        plink_cq_ring_doorbell(qp);
    }
}

Phase 1 구현 참고:
  - lane_id == 0만 SQ/CQ 조작을 수행 (단순화)
  - Phase 4에서 warp 내 다수 lane이 submit/poll을 분담하도록 최적화
  - 이 단순 구현에서도 async 동작 (iodepth > 1)은 정상 동작함

================================================================================
5. CMakeLists.txt
================================================================================

cmake_minimum_required(VERSION 3.18)
project(parallelink LANGUAGES C CXX CUDA)

find_package(CUDA REQUIRED)

# fio 헤더 경로
set(FIO_DIR ${CMAKE_SOURCE_DIR}/extern/fio)
# libnvm 경로
set(BAM_DIR ${CMAKE_SOURCE_DIR}/extern/bam)

# fio external engine은 반드시 shared library로 빌드
add_library(parallelink SHARED
    src/gpu_engine.c
    src/gpu_worker.cu
)

target_include_directories(parallelink PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${FIO_DIR}
    ${BAM_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# CUDA 설정
set_target_properties(parallelink PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;80;90"     # Volta, Ampere, Hopper
    POSITION_INDEPENDENT_CODE ON      # -fPIC (shared library 필수)
)

target_link_libraries(parallelink PRIVATE
    ${CUDA_LIBRARIES}
    # libnvm 라이브러리 (빌드 후 링크)
)

# fio 빌드 (사전에 extern/fio에서 ./configure && make)
# fio의 config-host.h가 필요하므로 fio를 먼저 빌드해야 함

# plink-holder (Phase 3)
# add_executable(plink-holder src/holder.cu)
# target_include_directories(plink-holder PRIVATE ...)
# target_link_libraries(plink-holder PRIVATE ...)

================================================================================
6. extern 의존성 설정
================================================================================

---- fio ----

git submodule add https://github.com/axboe/fio extern/fio
cd extern/fio
./configure
make -j$(nproc)
# config-host.h가 생성되어야 gpu_engine.c가 컴파일 가능

---- bam (libnvm) ----

git submodule add https://github.com/ZaidQureshi/bam extern/bam
cd extern/bam
git submodule update --init --recursive
mkdir -p build && cd build
cmake .. -DNVIDIA=/usr/src/nvidia-<version>/
make libnvm

# 커널 모듈 빌드
cd extern/bam/build/module
make

# 커널 모듈 로드
sudo make load

# NVMe unbind (테스트 대상 SSD)
echo -n "0000:XX:XX.X" > /sys/bus/pci/devices/0000:XX:XX.X/driver/unbind
# → /dev/libnvm0 생성 확인

================================================================================
7. 제약사항 및 주의사항
================================================================================

7.1 하드웨어 요구사항:
  - NVIDIA Datacenter GPU (Volta 이상): V100, A100, H100, B200
    → Tesla T4는 BAR1 256MB 제한으로 동작 불가
  - NVMe SSD (아무 제조사 가능)
  - PCIe P2P 지원 시스템 (IOMMU 비활성, ACS 비활성)
  - GPU와 NVMe가 같은 PCIe switch 아래 배치 권장 (1 hop)

7.2 소프트웨어 요구사항:
  - Linux kernel 5.x (6.x는 libnvm 커널 모듈 호환성 문제 가능)
  - CUDA 11.x 또는 12.x
  - NVIDIA 드라이버 470+ (nvidia_p2p API 지원)
  - BIOS: Above 4G Decoding 활성화

7.3 libnvm 수정 필요 사항:
  - nvm_parallel_queue.cuh의 cq_poll(): blocking → non-blocking 변경 필요
    → 기존: phase bit 일치할 때까지 spin
    → 변경: phase bit 불일치 시 즉시 return 0
  - 또는 parallelink 자체적으로 SQ/CQ 조작 함수를 구현
    (위 명세의 plink_sq_submit, plink_cq_poll 등)

7.4 fio 연동 제약:
  - GPU가 자율 I/O를 하므로 fio의 io_u는 형식적으로만 사용
  - fio의 verify, trim, write_barrier 등 고급 기능은 Phase 1에서 미지원
  - fio의 latency 통계는 Phase 1에서 GPU→CPU done_count 기반 근사치
    Phase 4에서 GPU clock64() 기반 정밀 연동

7.5 Warp-SQ 1:1 매핑 제약:
  - gpu_warps는 SSD가 지원하는 최대 I/O SQ 수 이하여야 함
  - 일반적으로 NVMe SSD는 64~128개 SQ 지원
  - gpu_warps가 SSD 한계를 초과하면 자동 제한 + 경고

================================================================================
8. 테스트 시나리오 (Phase 1 검증)
================================================================================

8.1 기본 동작 확인:

  # 최소 설정으로 동작 확인
  fio --ioengine=external:./parallelink.so \
      --name=basic --gpu_warps=1 --iodepth=1 \
      --nvme_dev=/dev/libnvm0 --rw=read --bs=4k \
      --runtime=5 --time_based

  기대 결과: done_count > 0, 에러 없음

8.2 Async 동작 확인 (iodepth > 1):

  fio --ioengine=external:./parallelink.so \
      --name=async --gpu_warps=1 --iodepth=16 \
      --nvme_dev=/dev/libnvm0 --rw=randread --bs=4k \
      --runtime=10 --time_based

  기대 결과: iodepth=1 대비 IOPS 향상 확인

8.3 Warp 스케일링:

  for w in 1 4 16 64; do
    fio --ioengine=external:./parallelink.so \
        --name=scale_${w} --gpu_warps=${w} --iodepth=16 \
        --nvme_dev=/dev/libnvm0 --rw=randread --bs=4k \
        --runtime=10 --time_based
  done

  기대 결과: warp 수 증가에 따른 IOPS 증가 (SSD 한계까지)

8.4 Size-based 모드:

  fio --ioengine=external:./parallelink.so \
      --name=size --gpu_warps=4 --iodepth=16 \
      --nvme_dev=/dev/libnvm0 --rw=randread --bs=4k \
      --io_size=1G

  기대 결과: 정확히 1GB 수행 후 종료

8.5 Write 테스트:

  fio --ioengine=external:./parallelink.so \
      --name=write --gpu_warps=4 --iodepth=16 \
      --nvme_dev=/dev/libnvm0 --rw=write --bs=4k \
      --runtime=10 --time_based

  기대 결과: NVMe write 정상 동작

8.6 장시간 안정성:

  fio --ioengine=external:./parallelink.so \
      --name=stability --gpu_warps=16 --iodepth=64 \
      --nvme_dev=/dev/libnvm0 --rw=randread --bs=4k \
      --runtime=3600 --time_based

  기대 결과: 1시간 연속 실행, 에러 없음, 메모리 누수 없음

================================================================================
END OF SPECIFICATION
================================================================================
