# GPU I/O 개선 방안 제시

> 분석 기반: `analysis_bottleneck.md`, `analysis_warp.md`
> 대상 파일: `src/gpu_worker.cu`, `extern/bam/include/nvm_parallel_queue.h`

---

## 우선순위 요약표

| # | 개선 항목 | Impact | 구현 난이도 | Risk | 권장 순서 |
|---|---|---|---|---|---|
| 4 | Opcode hoisting (template) | Med | Easy | Low | **1순위** |
| 5 | lba_range check hoisting | Med | Easy | Low | **1순위** |
| 6 | `__syncwarp()` 제거 | Med | Easy | Med | **1순위** |
| 7 | Read/Write 분리 커널 | Med | Easy | Low | **2순위** |
| 3 | Multi-CID in-flight 파이프라이닝 | **High** | Medium | Med | **3순위** |
| 2 | Per-thread Queue | **High** | Medium | Med | **3순위** |
| 1 | Warp-cooperative I/O | **High** | Hard | High | **4순위** |
| 8 | 아키텍처 전환 (async model) | **High** | Hard | High | **장기** |

---

## 개선 1: Warp-Cooperative I/O (Warp-level batching)

### 현황 분석

`sq_enqueue` (`nvm_parallel_queue.h:168`)에 이미 warp-cooperative 구현이 **주석 처리된 채** 존재한다:

```cpp
// nvm_parallel_queue.h:170-181 (주석 처리된 원본 시도)
//uint32_t mask = __activemask();
//uint32_t active_count = __popc(mask);
//uint32_t leader = __ffs(mask) - 1;
//uint32_t lane = lane_id();
/* if (lane == leader) { */
/*     ticket = sq->in_ticket.fetch_add(active_count, simt::memory_order_acquire); */
/* } */
/* ticket = __shfl_sync(mask, ticket, leader); */
/* ticket += __popc(mask & ((1 << lane) - 1)); */
```

이 시도의 핵심 아이디어는 올바르지만 불완전했다. `in_ticket` atomic은 leader 하나만 수행하면 되지만,
**각 lane의 cmd (LBA, data ptr 등)를 leader가 대표 제출할 수 없다는 근본 문제**가 있다 —
각 thread는 서로 다른 slba, pc_entry를 가지며, NVMe 커맨드를 개별적으로 SQ에 써야 한다.

따라서 "1개 SQ entry로 32개 I/O"는 불가능하고, 실질적 목표는:
**32개 atomic `fetch_add`를 1개 leader의 `fetch_add(32)`로 대체 → atomic contention 32× 감소**.

### 개선 방향

```cuda
// nvm_parallel_queue.h sq_enqueue 내 수정 (의사코드)
inline __device__
uint16_t sq_enqueue_warp(nvm_queue_t* sq, nvm_cmd_t* cmd, ...) {
    uint32_t mask = __activemask();
    uint32_t active_count = __popc(mask);
    uint32_t lane = threadIdx.x & 31;
    uint32_t leader_lane = __ffs(mask) - 1;  // 최하위 active lane

    uint32_t ticket;
    if (lane == leader_lane) {
        // leader만 active_count만큼 한 번에 ticket 확보
        ticket = sq->in_ticket.fetch_add(active_count, simt::memory_order_relaxed);
    }
    // 각 lane에게 자신의 ticket 분배:
    // leader: ticket+0, 다음 active lane: ticket+1, ...
    ticket = __shfl_sync(mask, ticket, leader_lane);
    uint32_t lane_offset = __popc(mask & ((1u << lane) - 1u));
    ticket += lane_offset;

    uint32_t pos = ticket & sq->qs_minus_1;
    uint64_t id  = get_id(ticket, sq->qs_log2);

    // 이후 각 lane은 자신의 pos/id로 기존과 동일하게 진행
    // (ticket 대기, cmd 복사, tail_mark 설정 등)
    // ...
}
```

### 효과 및 한계

**효과**:
- `in_ticket.fetch_add` atomic: warp당 32회 → 1회 (**32× atomic contention 감소**)
- Global memory traffic: 동일한 atomic variable에 32개 write → 1개 write
- 이론적 sq_enqueue 처리량: 최대 32× 향상 (ticket 획득 단계만)

**한계 및 주의사항**:
- `pc_entry` (page cache slot)는 여전히 per-thread — `__shfl_sync`로 공유 불필요
- `tail_lock`/`head_lock` TAS spinlock은 여전히 존재 → doorbell ring은 여전히 직렬화
- `cq_poll`은 변경 없음 — 각 thread가 자신의 CID를 독립적으로 대기
- 단, ticket 대기 loop (#1, #2 barrier)에서 32개 thread가 **서로 다른 ticket id**를 가지므로
  순차 진행 구조는 동일 → **평균 대기 시간 자체는 줄지 않음**
- BaM의 page_cache는 per-page slot이므로 lane별 독립 pc_entry는 그대로 유지 가능

**결론**: atomic contention 감소 효과는 실재하나, 9개 barrier 중 1개만 개선.
ticket 직렬화 구조 자체를 바꾸지 않는 한 근본 한계 돌파 불가.

**평가**: Impact=High, 구현 난이도=Hard, Risk=High (BaM 내부 수정 필요)

---

## 개선 2: Per-Thread Queue (분리된 병렬성)

### 현황

```cuda
// gpu_worker.cu:118
int q_idx = (tid / 32) % n_queues;  // warp당 1 queue
```

32개 threads가 하나의 `nvm_queue_t`를 공유 → 9개 spin-wait barrier 전체에서 경합 발생.

### 개선안

```cuda
// 변경 후
int q_idx = tid % n_queues;  // thread당 1 queue
```

`n_queues >= total_threads` 조건 충족 시, 각 thread는 **전용 SQ/CQ**를 가져
`in_ticket`, `tail_lock`, `head_lock`, `cq->pos_locks` 등 **모든 shared atomic이 경합 없음**.

### 실현 가능성 분석

**NVMe 큐 한계**:
- NVMe 스펙: 최대 65,535개 I/O 큐 (Controller Capabilities 레지스터 MQES/NCQR 필드)
- 실제 고성능 SSD (Samsung PM9A3, Micron 9400): 64~128개 큐가 일반적
- BaM `Controller` 생성자: `(uint64_t)n_queues` 파라미터로 제어 → 하드웨어 한계 내

**스레드 수 대비 큐 수**:
- 현재: `gpu_warps * 32` threads (기본값 예: 64 warps × 32 = 2048 threads)
- 2048개 큐: 고성능 NVMe (엔터프라이즈 SSD)에서는 가능, 소비자급 SSD에서는 불가
- 현실적 접근: `n_queues = min(hw_max_queues, total_threads)`
  → `q_idx = tid % n_queues` 로 remainder mapping

**비용**:
- 메모리: 큐당 SQ+CQ 메모리 = `2 × queue_depth × 64B` + 관리 구조체
  - queue_depth=128, n_queues=128: 128 × (2 × 128 × 64B) = 2MB — 수용 가능
- BaM Controller 초기화: 큐 수만큼 NVMe Set Features 어드민 커맨드 → 초기화 시간 증가

**효과**:
- Intra-warp queue contention **완전 제거** → `in_ticket` atomic 경합 0
- `tail_lock`, `head_lock`: 각 thread 전용이므로 항상 uncontested
- `cq_poll` 선형 스캔: 독립 큐이므로 항상 자신의 CID만 존재 → 1회 스캔에서 즉시 발견
- 이론적 throughput: (단일 thread throughput) × total_threads (NVMe 한계까지)

**평가**: Impact=High, 구현 난이도=Medium (host init 변경 포함), Risk=Medium (HW 의존)

---

## 개선 3: Multi-CID In-Flight 파이프라이닝

### 현황: Depth-1 Submit-Wait

```
[Thread 0]: submit(cid=0) → wait → submit(cid=0) → wait → ...
[Thread 1]: submit(cid=1) → wait → submit(cid=1) → wait → ...
```

각 thread는 1개 I/O가 완료될 때까지 다음 submit 불가. NVMe queue depth (BaM: 65,536 CIDs/queue)를
사실상 thread 수 (= warp × 32)만큼만 활용.

### 개선안: N-deep Pipeline per Thread

```cuda
// 각 thread가 N개 I/O를 연속 제출 후 일괄 완료 대기
__global__ void plink_io_worker_pipelined(..., int pipeline_depth) {
    // ...
    uint16_t pending_cids[MAX_PIPELINE_DEPTH];
    uint32_t pending_pos[MAX_PIPELINE_DEPTH];   // SQ slot positions
    uint32_t pending_locs[MAX_PIPELINE_DEPTH];  // CQ completion locs
    uint32_t pending_heads[MAX_PIPELINE_DEPTH];
    uint64_t pending_slbas[MAX_PIPELINE_DEPTH];

    int in_flight = 0;

    while (true) {
        // SUBMIT PHASE: pipeline_depth만큼 발행
        while (in_flight < pipeline_depth) {
            uint64_t slba = compute_next_lba(...);  // curand or sequential

            // BaM get_cid + sq_enqueue (non-blocking submit)
            uint16_t cid = get_cid(&qp->sq);
            nvm_cmd_t cmd;
            // ... build cmd with cid, slba, pc_entry ...
            uint16_t pos = sq_enqueue(&qp->sq, &cmd);

            pending_cids[in_flight]  = cid;
            pending_pos[in_flight]   = pos;
            pending_slbas[in_flight] = slba;
            in_flight++;
        }

        // COMPLETION PHASE: 순서대로 wait
        for (int i = 0; i < in_flight; i++) {
            uint32_t loc, head;
            uint32_t cq_loc = cq_poll(&qp->cq, pending_cids[i], &loc, &head);
            cq_dequeue(&qp->cq, cq_loc, &qp->sq, loc, head);
            sq_dequeue(&qp->sq, pending_pos[i]);
            put_cid(&qp->sq, pending_cids[i]);
        }
        in_flight = 0;

        pending_done += pipeline_depth;
        // ... shutdown check, done counter flush ...
    }
}
```

### 핵심 변경 포인트 (while(true) loop)

현재 `gpu_worker.cu:129-156`:
```cuda
// BEFORE (depth=1):
while (true) {
    if (lba_max > ...) slba = curand_lba(...);
    read_data(pc, qp, slba, wl.n_blocks, pc_entry);  // submit+wait 내장
    pending_done++;
    // ...
    __syncwarp();
}
```

변경 후 (depth=N):
```cuda
// AFTER (depth=N, read_data 분해 필요):
// BaM의 read_data = get_cid + sq_enqueue + cq_poll + cq_dequeue + sq_dequeue + put_cid
// 이를 분해하여 submit/wait 단계를 분리

while (true) {
    // Submit N
    for (int d = 0; d < N && !ctrl->shutdown; d++) {
        slba[d] = compute_lba(d);
        cids[d] = get_cid(&qp->sq);
        build_read_cmd(&cmds[d], pc, slba[d], wl.n_blocks, pc_entry, cids[d]);
        pos[d]  = sq_enqueue(&qp->sq, &cmds[d]);
    }
    // Wait N
    for (int d = 0; d < N; d++) {
        uint32_t loc, head;
        uint32_t cq_loc = cq_poll(&qp->cq, cids[d], &loc, &head);
        cq_dequeue(&qp->cq, cq_loc, &qp->sq, loc, head);
        sq_dequeue(&qp->sq, pos[d]);
        put_cid(&qp->sq, cids[d]);
    }
    pending_done += N;
}
```

### 효과

- **N개 I/O가 NVMe에 동시 in-flight** → NVMe 내부 병렬 처리 (NCQ 활용)
- 이론 처리량: N × per-thread-IOPS (NVMe 포화까지)
- BaM CID pool: 65,536개/queue → N=32까지는 충분
- Warp 관점: 32 threads × N in-flight = 32N개 동시 I/O / queue
  - N=8: 256 outstanding I/Os — 현재 depth=32와 동일하지만 ticket contention 없이

**제약사항**:
- `read_data`/`write_data` BaM 헬퍼가 submit과 wait을 하나로 묶고 있음
  → BaM 헬퍼를 우회하여 `get_cid` + `sq_enqueue` + `cq_poll` + 나머지를 직접 호출해야 함
- BaM 헬퍼(`nvm_io.h`)의 내부 구조 변경 필요 — 또는 parallelink 전용 헬퍼 작성
- `pending_cids[N]`, `pending_pos[N]` 등 per-thread 배열이 레지스터/로컬 메모리 사용 증가
  - N=8: 4개 uint16/uint32 배열 × 8 = 32개 추가 값 → register pressure 증가
  - `__launch_bounds__` 조정 필요할 수 있음

**평가**: Impact=High, 구현 난이도=Medium, Risk=Medium (BaM I/O helper 분해)

---

## 개선 4: Opcode Hoisting (즉각적 효과)

### 현황

```cuda
// gpu_worker.cu:142 — while(true) 내부, 매 iteration 실행
if (wl.opcode == PLINK_OP_READ)
    read_data(pc, qp, slba, wl.n_blocks, pc_entry);
else
    write_data(pc, qp, slba, wl.n_blocks, pc_entry);
```

`wl.opcode`는 kernel parameter (by-value, 레지스터) → 루프 전체에서 **불변**.
`read_data`/`write_data`는 blocking I/O call → 컴파일러가 side-effect-free로 증명 불가
→ NVCC가 branch를 매 iteration마다 실행.

### 개선안 A: Template Parameter

```cuda
template <bool IS_READ>
__global__ __launch_bounds__(64, 32)
void plink_io_worker_random_t(struct plink_ctrl_block *ctrl,
                               uint64_t *d_done_count,
                               struct plink_workload wl,
                               Controller **ctrls,
                               page_cache_d_t *pc,
                               int n_queues)
{
    // ... 동일 초기화 ...
    while (true) {
        // ...
        if constexpr (IS_READ)
            read_data(pc, qp, slba, wl.n_blocks, pc_entry);
        else
            write_data(pc, qp, slba, wl.n_blocks, pc_entry);
        // ...
    }
}

// plink_gpu_launch 내 dispatch:
if (wl.opcode == PLINK_OP_READ)
    plink_io_worker_random_t<true><<<...>>>(...)
else
    plink_io_worker_random_t<false><<<...>>>(...)
```

`if constexpr`는 컴파일 타임에 분기를 제거 → PTX 레벨에서 branch instruction 완전 소거.

### 개선안 B: 단순 루프 복제 (더 간단)

`while(true)` body를 read/write용으로 각각 복제 (이미 random/sequential 분리와 동일 패턴):
```cuda
if (wl.opcode == PLINK_OP_READ) {
    while (true) {
        // ... read_data 고정 ...
    }
} else {
    while (true) {
        // ... write_data 고정 ...
    }
}
```
단점: 코드 중복. `개선 7`과 통합하면 자연스럽게 해결됨.

### 효과

- PTX branch instruction 1개/iteration 제거 → 미미하지만 무료로 얻는 최적화
- I/O 시간 대비 branch cost는 무시 가능 수준 (1~2 cycles vs 50μs I/O)
- **주 가치**: 코드 명확성 향상, 컴파일러 최적화 기회 확대
- sequential 커널도 동일 적용 가능 (`gpu_worker.cu:201-204`)

**평가**: Impact=Low~Med, 구현 난이도=Easy, Risk=Low

---

## 개선 5: lba_range Check Hoisting

### 현황

```cuda
// gpu_worker.cu:135 — while(true) 내부, 매 iteration
if (lba_max > (uint64_t)wl.n_blocks) {
    uint64_t bound = lba_max - (uint64_t)wl.n_blocks;
    uint64_t r = ((uint64_t)curand(&rng) << 32) | (uint64_t)curand(&rng);
    slba = r % bound;
}
```

`lba_max`는 `wl.lba_range`로 초기화 후 불변 (레지스터). 조건 `lba_max > wl.n_blocks`도 불변.
단, `curand()` 호출은 루프마다 필요 (random LBA를 계속 생성해야 함) → 내부 로직은 유지.

### 개선안

```cuda
// 루프 전: 불변 조건 평가
const bool do_random_lba = (lba_max > (uint64_t)wl.n_blocks);
const uint64_t bound = do_random_lba ? (lba_max - (uint64_t)wl.n_blocks) : 0;

while (true) {
    if (do_random_lba) {
        uint64_t r = ((uint64_t)curand(&rng) << 32) | (uint64_t)curand(&rng);
        slba = r % bound;
    }
    // ...
}
```

또는 template parameter로:
```cuda
template <bool DO_RANDOM_LBA, bool IS_READ>
__global__ void plink_io_worker_random_t(...) { ... }
```

### 효과

- 컴파일러가 NVCC `-O3`에서 이미 이를 레지스터에 캐시할 가능성 있음
- 개선 4와 결합 시 template 폭발 (4가지 조합) → 루프 복제 방식 B가 더 현실적

**평가**: Impact=Low, 구현 난이도=Easy, Risk=Low

---

## 개선 6: `__syncwarp()` 제거 / 재고

### 현황

```cuda
// gpu_worker.cu:155 (random), :218 (sequential)
__syncwarp();
```

### 분석 (`analysis_warp.md` §4 기반)

`sq_enqueue`의 ticket 직렬화로 인해 warp 내 threads는 이미 순차적으로 SQ에 write된다:
- Thread 0이 cq_poll에서 spin하는 동안 Thread 1은 sq_enqueue 대기 중
- Thread 31은 Thread 0~30이 모두 sq_enqueue를 완료해야 시작 가능

따라서 `__syncwarp()`의 실제 역할:
1. **중복 장벽**: ticket 시스템이 이미 순서를 강제 — `__syncwarp()`는 추가 동기화 없음
2. **성능 저하**: 빠른 thread (I/O가 일찍 완료)가 늦은 thread를 기다려야 함
   - Thread 0: 45μs에 cq_dequeue 완료 → `__syncwarp()` 에서 Thread 31 대기 (최대 105μs)
   - Thread 0의 낭비: 60μs = 60,000 clock cycles

### 제거 효과

`__syncwarp()` 제거 시:
- 빠른 thread는 즉시 다음 iteration의 `sq_enqueue` 티켓 획득 시작
- 결과: **pipeline overlapping** — Thread 0의 N+1번째 submit과 Thread 31의 N번째 wait가 겹침
- Warp의 effective throughput이 max-latency bound → average-latency bound로 개선

### Risk 분석

- `sq_enqueue` ticket 시스템 자체가 순서를 보장하므로, 동일 queue 내 SQ 슬롯 충돌 없음
- `pc_entry = tid % pc->n_pages` — thread별 고유 → 제거 후에도 page cache 충돌 없음
- `d_done_count` atomic 업데이트: `(pending_done & 1023ULL) == 0` 조건 → threads 간 독립적
- **단, warp divergence가 증가**할 수 있음: 빠른 thread가 다음 iteration으로 진행할 때
  늦은 thread와 다른 code path에 있으면 SIMT divergence 발생. 그러나 `while(true)` loop은
  모든 thread가 동일 분기(read 또는 write)를 타므로 divergence 최소.
- `__syncwarp()` 제거가 안전한 조건: warp 내 threads 간 shared memory 통신 없음 ✓

**결론**: 제거 권장. 단, 동일 pc_entry 공유 여부 재확인 필요.

**평가**: Impact=Med, 구현 난이도=Easy, Risk=Med (테스트로 검증 필요)

---

## 개선 7: Read/Write 분리 커널

### 현황

현재 `plink_gpu_launch` (`gpu_worker.cu:385`)는 `wl.random`으로만 커널을 분기:

```cpp
if (wl.random)
    plink_io_worker_random<<<...>>>(...)
else
    plink_io_worker_sequential<<<...>>>(...)
```

각 커널 내에는 여전히 `if (wl.opcode == PLINK_OP_READ)` 분기가 존재.

### 개선안

```cpp
// 4가지 커널로 완전 분리
if (wl.random && wl.opcode == PLINK_OP_READ)
    plink_io_worker_random_read<<<...>>>(...)
else if (wl.random && wl.opcode == PLINK_OP_WRITE)
    plink_io_worker_random_write<<<...>>>(...)
else if (!wl.random && wl.opcode == PLINK_OP_READ)
    plink_io_worker_sequential_read<<<...>>>(...)
else
    plink_io_worker_sequential_write<<<...>>>(...)
```

또는 `개선 4`의 template 방식으로 2×2 조합 생성:
```cpp
plink_io_worker</*random=*/true,  /*is_read=*/true><<<...>>>(...)
plink_io_worker</*random=*/true,  /*is_read=*/false><<<...>>>(...)
plink_io_worker</*random=*/false, /*is_read=*/true><<<...>>>(...)
plink_io_worker</*random=*/false, /*is_read=*/false><<<...>>>(...)
```

### 효과

- Opcode branch (`개선 4`)와 lba_range check (`개선 5`)를 동시에 해결
- 커널당 최적화 기회 확대 (NVCC 최적화 범위 축소)
- 기존 random/sequential 분리 패턴의 자연스러운 확장

**평가**: Impact=Med, 구현 난이도=Easy, Risk=Low

---

## 개선 8: 근본 아키텍처 재고 — Async Submit/Poll 분리

### 핵심 문제 재진술

```
CPU NVMe I/O: submit → OS yields CPU → NVMe interrupt → resume = 0 wasted cycles
GPU BaM I/O: submit → spin-wait ~50,000 cycles → complete = 50,000 wasted cycles/thread
```

NVMe는 **레이턴시 바운드** (50~200μs). GPU는 **처리량 머신** (수천 thread 병렬).
이 둘은 근본적으로 충돌한다. GPU가 CPU를 이기려면 다음 조건이 필요하다:

1. **Many independent in-flight I/Os**: GPU 스레드 수 × in-flight depth가 NVMe queue depth에 도달
2. **OR Large batching**: 1 GPU command = N pages (현재 n_blocks=1이면 4KB/I/O)
3. **OR Async completion notification**: GPU thread가 spin 없이 다른 작업을 수행하다가 완료 시 재개

### Async 아키텍처 스케치

**Producer-Consumer 분리**:
```
[Submit Warps]    [NVMe]       [Completion Warps]
submit → SQ → → → → → → → → CQ → poll → dequeue
    ↑                                      ↓
    ←←←←←←← free CID queue ←←←←←←←←←←←←←
```

구현 방향:
- Submit warps: CID free list에서 CID 획득 → NVMe command 빌드 → sq_enqueue → 다음 CID로 이동
- Completion warps: CQ 전체를 순회하며 완료된 entry 발견 → cq_dequeue → CID free list 반환
- **두 그룹이 독립적으로 실행** — submit warp는 cq_poll에서 절대 block하지 않음

이를 위한 BaM 수정:
- `sq_enqueue`에서 ticket-wait 이후 바로 반환 (tail_lock까지만)
- `cq_poll`을 별도 warp 그룹이 전담
- CID free list를 lock-free queue (MPMC ring buffer)로 구현

### GPU > CPU 처리량 조건

현재 parallelink 설정 (4KB I/O, 단일 쓰레드당 1 in-flight):
- CPU (fio, io_uring): 1개 core × 32 in-flight = 32 × 20KIOPS = 640K IOPS
- GPU (parallelink): 2048 threads × 1 in-flight = 2048 × 20KIOPS → NVMe saturated (500K~1M IOPS)

**결론**: 현재 per-thread blocking 모델에서도 thread 수가 충분하면 NVMe를 포화시킬 수 있다.
문제는 **GPU 2048 threads가 모두 spin-wait 중이면 SM이 완전히 멈춘다**는 것.

GPU의 실제 throughput advantage:
- NVMe가 IOPS를 포화시키는 데 필요한 동시 I/O 수 = IOPS × latency = 500K × 50μs = **25개**
- GPU는 2048개 thread를 동시 spin-wait시켜 25개 in-flight를 유지 → **극도로 낭비**
- 개선 3 (N-deep pipelining) + 개선 2 (per-thread queue): 소수의 thread가 많은 in-flight를 담당

**장기 권장**: BaM async 아키텍처 + per-thread queue + N-deep pipelining 조합이 근본 해법.

**평가**: Impact=High, 구현 난이도=Hard, Risk=High (BaM 전체 구조 변경)

---

## 실행 계획 (단계별)

### Phase 1: 즉각 적용 (1일, 코드 변경 최소)

1. **`__syncwarp()` 제거** (`gpu_worker.cu:155, 218`) — 1줄 변경
2. **`wl.opcode` 루프 외부 if 이동** (개선 4, 방식 B) — while(true) body 복제
3. **lba_range check 루프 외부 이동** (개선 5) — `const bool + bound` 사전 계산
4. **벤치마크**: 위 3개 변경 후 throughput 측정

### Phase 2: 단기 구조 개선 (1주, 중간 변경)

5. **Read/Write 분리 커널** (개선 7) — template 또는 복제 방식
6. **Per-thread queue 실험** (개선 2) — `q_idx = tid % n_queues`, n_queues 파라미터 증가
   - 먼저 소수 thread로 테스트 (q_idx collision 없는 조건 확인)
7. **벤치마크**: 큐 수 vs 처리량 곡선 측정

### Phase 3: 중기 성능 최적화 (2~4주, 큰 변경)

8. **Multi-CID 파이프라이닝** (개선 3) — BaM I/O helper 분해, N=4/8/16 실험
9. **Warp-cooperative ticket** (개선 1) — `in_ticket.fetch_add(active_count)` 복원
10. **프로파일링**: `ncu` (Nsight Compute)로 barrier 별 stall cycles 측정

### Phase 4: 장기 아키텍처 전환 (수개월)

11. **Async Submit/Completion 분리** (개선 8) — BaM 내부 수정 또는 별도 구현
12. **대용량 I/O (multi-block)** — n_blocks 증가로 PCIe 효율 개선 (1I/O = 64KB)

---

## 핵심 인사이트

> **BaM의 ticket lock은 GPU SIMT parallelism을 CPU sequential execution으로 강제 변환한다.**
> 32 threads가 동시에 실행되지만 1개씩 차례로 I/O를 제출하고, 각자 50μs씩 spin-wait한다.
> `개선 2` (per-thread queue)로 경합을 제거하고, `개선 3` (pipelining)으로 latency를 숨기는 것이
> 근본적이고 가장 효과적인 경로다.
>
> `개선 4, 5, 6, 7`은 NVMe latency에 비해 무시 가능한 오버헤드를 제거하지만,
> 코드 품질 향상과 미래 최적화 기반을 마련한다는 점에서 즉시 적용 가치가 있다.
