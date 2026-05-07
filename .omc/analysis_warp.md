# CUDA WARP 실행 모델 분석

> 분석 대상: `src/gpu_worker.cu`, `extern/bam/include/nvm_parallel_queue.h`

---

## 1. 1:1 Queue-Warp 매핑 분석

### 매핑 구조

```c
// gpu_worker.cu:118
int q_idx = (tid / 32) % n_queues;
QueuePair *qp = &(ctrls[0]->d_qps[q_idx]);
```

Warp 내 32개 threads (tid = N×32 ~ N×32+31) 모두 동일한 `q_idx` 계산
→ 동일한 `QueuePair` 포인터 사용.

### sq_enqueue 내부 직렬화 추적

`sq_enqueue` (nvm_parallel_queue.h:168)의 실행 흐름:

**Step 1: 티켓 획득 (atomic, 비직렬)**
```c
ticket = sq->in_ticket.fetch_add(1, simt::memory_order_relaxed);  // line 175
uint32_t pos = ticket & (sq->qs_minus_1);
uint64_t id  = get_id(ticket, sq->qs_log2);  // = (ticket / qs) * 2
```
32개 threads가 동시에 `in_ticket`에 `fetch_add`하여 각각 고유한 ticket 번호(0~31)를 받는다.

**Step 2: 순서 대기 (직렬 spin-wait) ← 핵심 병목**
```c
while (sq->tickets[pos].val.load(simt::memory_order_relaxed) != id) {
    __nanosleep(ns);  // 8→256ns exponential backoff
    if (ns < 256) ns *= 2;
}
```
- `tickets[pos]`는 해당 SQ 슬롯이 이전 사용자로부터 반환(release)되었을 때만 `id`와 일치
- Thread N은 ticket N-1을 가진 thread가 `sq->tickets[pos].val.fetch_add(1)` (line 327)을 호출해야 비로소 진행 가능
- **결과: 32개 threads가 SQ 슬롯 접근을 완전히 직렬화하여 순서대로 한 번에 하나씩 진행**

**Step 3: 커맨드 복사 및 doorbell ring**
```c
for (uint32_t i = 0; i < 64/sizeof(copy_type); i++)
    queue_loc[i] = cmd_[i];  // 64바이트 NVMe 커맨드 복사
// ...
sq->tail_mark[pos].val.store(LOCKED, ...);
// tail_lock 획득 → move_tail() → doorbell write (mmio store)
```
`tail_lock`은 여러 thread 중 하나만 doorbell을 ring하도록 함 (배치 가능).
하지만 SQ 슬롯 접근 자체는 Step 2의 ticket 대기로 직렬화되어 있음.

### 동시 CID 수 ("in flight" 분석)

- **Submission phase**: 32개 thread가 sq_enqueue를 하나씩 순서대로 완료 → NVMe controller에 커맨드를 순차적으로 전달
- **Completion phase**: 각 thread는 자신의 CID로 `cq_poll`에서 spin → **독립적으로** 대기

따라서 이론적으로는 32개 CID가 NVMe device에 동시 in flight 가능 (queue depth ≥ 32인 경우). 그러나:
- 각 thread는 sq_enqueue → cq_poll → cq_dequeue 를 **동기적(blocking)**으로 실행
- Thread N이 sq_enqueue 중에 Thread N-1은 이미 cq_poll에서 대기 중
- NVMe 입장에서는 "파이프라이닝" 효과로 복수 커맨드 처리 가능하지만, GPU thread당 항상 1개 CID만 outstanding

**실효 queue depth**: Warp당 동시 in-flight는 최대 32개이나, 직렬 제출로 인해 실제 평균 in-flight는 훨씬 적음. NVMe latency(50~100μs)에 비해 직렬 submission overhead가 무시할 수 없는 수준.

---

## 2. Warp Scheduler와 spin-wait 동작

### `__nanosleep`의 본질

`__nanosleep(ns)`는 PTX `nanosleep` 명령으로, **compute 명령**이다:
- Global memory load/store와 달리 "메모리 대기" 상태로 전환되지 않음
- Warp는 `ns` 나노초 동안 **실행 자격 정지(pause)** 상태가 됨
- 이 기간 동안 SM 스케줄러는 다른 warp를 선택할 수 있음

### Memory stall vs. spin-wait 비교

| 특성 | Memory Stall (global load) | Spin-wait (`__nanosleep`) |
|---|---|---|
| Warp 상태 | "Waiting for memory" → 즉시 unschedule | Sleep/pause 후 재도전 |
| 스케줄러 반응 | 즉시 다른 warp 선택 (latency hiding) | ns 경과 후 warp가 ready로 복귀 |
| Warp slot 점유 | 계속 점유 (pending) | 계속 점유 |
| SM 활용 효율 | 다른 warp의 compute로 gap 메움 | 대기 warp들도 sleep 상태면 SM stall |
| 대기 시간 | ~수백 ns (L2/HBM latency) | ~50-100μs (NVMe latency) |

### "모든 Warp가 NVMe 대기 중" 시나리오

NVMe latency: 50,000~100,000 ns
`__nanosleep` 최대: 256 ns → NVMe 완료까지 약 **200~400회 wake-up-and-check** 반복

SM 내 모든 warp가 `cq_poll` loop에 진입하면:
1. 모든 warp가 `__nanosleep(8~256)` 실행 → 최대 256ns sleep
2. 256ns 후 일부 warp가 ready → CQ 스캔 → 아직 completion 없음 → 다시 sleep
3. **어떤 warp도 compute 작업을 하지 않는 기간이 수만 ns 지속**
4. **SM complete stall**: Issue slot에서 실행 가능한 warp가 0개

메모리 bound 커널과 달리, NVMe spin-wait는 SM의 latency hiding 메커니즘을 완전히 우회한다. 수십 마이크로초 규모의 stall을 수백 나노초 단위의 `__nanosleep`으로는 절대 숨길 수 없다.

---

## 3. Warp 수 증가해도 성능이 개선되지 않는 이유

### 동일 큐에 N개 Warp 추가 시 발생하는 contention

`q_idx = (tid / 32) % n_queues`이므로, `n_warps > n_queues`이면 여러 warp가 동일 큐를 공유.

**Contention point 1: `in_ticket` atomic**
```c
ticket = sq->in_ticket.fetch_add(1, simt::memory_order_relaxed);
```
- 1 warp × 32 threads → 32개 동시 atomic
- N warp × 32 threads → 32N개 동시 atomic → serialization latency 증가

**Contention point 2: `tail_lock` (doorbell serialization)**
```c
bool new_cont = sq->tail_lock.fetch_or(LOCKED, simt::memory_order_acquire) == LOCKED;
```
- 더 많은 thread가 lock 경쟁 → average wait time 증가

**Contention point 3: `cq_poll` 선형 스캔**
```c
for (size_t i = 0; i < cq->qs_minus_1; i++) {  // 전체 CQ 순회
    uint32_t cid = (cpl_entry & 0x0000ffff);
    if ((cid == search_cid) && (phase == search_phase)) { ... }
}
```
- 32N개 threads가 모두 동일한 CQ를 동시에 스캔
- Cache line thrashing: `cq->vaddr` 페이지에 대한 경쟁적 read → PCIe traffic 증가

**Contention point 4: `get_cid` / `put_cid`**
```c
id = sq->cid_ticket.fetch_add(1, simt::memory_order_relaxed) & (65535);
uint64_t old = sq->cid[id].val.fetch_or(LOCKED, simt::memory_order_acquire);
```
- CID pool 경쟁 → CID 충돌 시 재시도 loop

### 성능 한계 분석

NVMe IOPS 한계 (예: NVMe SSD ~500K IOPS):
- Per-thread max IOPS = 1 / (50μs latency) = 20,000 IOPS
- 1 warp (32 threads, 1 queue): **이론 최대 = min(32×20K, 500K) ≈ 500K IOPS**
- 2 warp (64 threads, 1 queue): contention overhead가 추가되나 device 한계에 이미 도달
- **N warp 추가 → queue contention 증가 + NVMe device 한계 초과 없음 → 순수 overhead만 추가**

결론: 현재 blocking 1-CID-per-thread 설계에서 Warp 수 증가는 동일 큐에 대한 경쟁만 심화시키며 throughput을 개선하지 않는다.

---

## 4. Hot loop 내 if-else 분기 분석

### 분석 대상 (plink_io_worker_random)

```c
// gpu_worker.cu:129 - while(true) 내부
while (true) {
    // ...
    if (lba_max > (uint64_t)wl.n_blocks) {  // 루프 불변 조건 (line 135)
        // random LBA 계산
    }
    if (wl.opcode == PLINK_OP_READ)         // 루프 불변 조건 (line 142)
        read_data(...);
    else
        write_data(...);
    // ...
    __syncwarp();
}
```

**루프 불변 조건 (Loop-invariant conditions)**:
- `lba_max`는 kernel 진입 시 `wl.lba_range` 로 초기화 (레지스터), 루프 내 변경 없음
- `wl.opcode`는 by-value 파라미터 → 레지스터, 루프 내 변경 없음
- NVCC는 이를 최적화하여 호이스팅할 수 있으나, `while(true)` 내 blocking call(`read_data`/`write_data`) 때문에 컴파일러가 side-effect-free로 증명 못해 **실제 호이스팅이 보장되지 않음**

**Warp divergence 여부**:
- 동일 warp 내 모든 threads는 동일한 `wl.opcode`와 동일한 `lba_max`를 가짐 (같은 kernel parameter)
- → **두 조건 모두 warp divergence 없음**: 32개 threads 전원이 동일한 branch를 선택
- 단, NVCC가 호이스팅하지 않으면 매 iteration마다 branch instruction이 실행되는 **비용은 발생**

### `__syncwarp()` 분석

```c
__syncwarp();  // gpu_worker.cu:155, 218
```

**의도**: 모든 warp lanes가 다음 iteration에 진입하기 전에 동기화.

**실제 동작과 문제점**:

BaM I/O 함수(read_data/write_data) → `sq_enqueue` (ticket-based 직렬화) → `cq_poll` (개별 CID 대기):
- Thread 0은 45μs에 완료, Thread 31은 105μs에 완료 가능
- `__syncwarp()`는 마지막 thread가 완료될 때까지 **빠른 threads를 강제 대기**

**결과**: 
- Warp의 effective throughput = min(thread0_latency, ..., thread31_latency) → **max latency에 바운드**
- sq_enqueue 자체가 직렬화(ticket ordering)를 강제하므로 `__syncwarp()`는 중복적인 barrier
- `__syncwarp()` 제거 시: 빠른 thread들이 다음 iteration으로 진행 가능 → 실효 pipeline depth 증가
- **`__syncwarp()`는 현재 코드에서 성능을 저하시키고 있음**

---

## 5. `__launch_bounds__` 및 Occupancy 분석

### 파라미터 해석

```c
__global__ __launch_bounds__(64, 32)
void plink_io_worker_random(...)
```

- `maxThreadsPerBlock = 64`: 블록당 최대 64 threads
- `minBlocksPerMultiprocessor = 32`: SM당 최소 32 blocks 동시 실행 가능하도록 register 사용 제한

**이론적 최대 occupancy**:
- 32 blocks/SM × 64 threads/block = **2048 threads/SM = 64 warps/SM**
- A100 SM: 최대 2048 threads/SM, 64 warps/SM → 100% theoretical occupancy

### Register 사용량 추정

```c
curandState rng;  // ≈ 48 bytes = 12 registers (4B each)
// tid, q_idx, qp, pc_entry, lba_max, pending_done, slba, r, bound, ns, ticket 등
// 추정 합계: ~40-56 registers/thread
```

CUDA register 파일 (A100): 65,536 registers/SM
- 40 regs × 2048 threads = 81,920 → **SM register limit 초과**
- 실제 occupancy: `floor(65536 / (40 × 64)) = floor(65536/2560) = 25 blocks/SM`
- 25 blocks × 64 threads = 1600 threads/SM = 50 warps/SM ≈ **78% occupancy**

`__launch_bounds__(64, 32)` 지시는 NVCC에게 "register spilling을 감수하더라도 32 blocks/SM을 허용하라"는 hint. Spill이 발생하면 local memory (L1/L2 경유) 접근이 추가되어 오히려 성능 저하 가능.

### Occupancy가 성능에 미치는 영향: NVMe spin-wait의 맥락

**전통적 latency hiding 이론**: 높은 occupancy → 더 많은 ready warps → 메모리 stall을 다른 warp로 커버

**NVMe spin-wait에서의 현실**:
- 50개 warps/SM 중 50개 전부가 `cq_poll`에서 `__nanosleep` spin → **zero ready warps**
- Occupancy가 50% → 25% → 10%로 감소해도 결과 동일: 모든 활성 warp가 NVMe latency를 기다리는 중
- NVMe latency(50~100μs) >> `__nanosleep` max(256ns) >> 메모리 latency(수백ns)
- **Occupancy 최적화는 NVMe I/O bound 커널에서 완전히 무의미**

추가: `__launch_bounds__` 지정은 BaM block benchmark와의 공정한 비교를 위한 것(코드 주석 참조)이며, 실제 throughput 개선 효과는 없음.

---

## 6. 핵심 결론

### 아키텍처적 병목 요약

| 문제 | 위치 | 영향 |
|---|---|---|
| sq_enqueue 직렬화 | `nvm_parallel_queue.h:175-327` | Warp 내 32 threads가 순차 제출 |
| NVMe latency ≫ nanosleep | `cq_poll` spin loop | SM 전체 stall (no ready warps) |
| 1:1 warp-queue 매핑 + N warp | `gpu_worker.cu:118` | N배 contention, 0배 throughput 향상 |
| `__syncwarp()` | `gpu_worker.cu:155,218` | Max-latency bound, pipelining 차단 |
| Loop-invariant branches | `gpu_worker.cu:135,142` | 매 iteration 불필요한 condition check |
| High occupancy + spin-wait | `__launch_bounds__(64,32)` | 이론적 최대 occupancy가 실효 없음 |

### 근본 원인

현재 설계는 **blocking, synchronous, per-thread I/O** 모델로, GPU의 SIMT 병렬성을 활용하지 못한다:

1. **GPU 병렬성 모델 불일치**: GPU는 수천 개의 thread가 compute를 병렬 실행하도록 설계됨. I/O completion spin-wait은 모든 thread가 동일한 external event를 기다리는 구조 → GPU의 병렬성이 경쟁적 contention으로 전환

2. **Latency hiding 불가**: NVMe latency(50μs)는 GPU memory latency(수백ns)의 100~1000배. 어떤 수의 warp를 가져와도 50μs 구멍을 메울 수 없음 (다른 warps도 동일하게 대기 중)

3. **Queue depth 저활용**: 현재 1 thread = 1 CID outstanding. NVMe queue depth 1024를 사용하려면 동시에 1024개 I/O를 in-flight해야 하나, blocking 모델에서는 thread 수로만 결정되며 warp당 최대 32개에 그침

### 개선 방향 (상위 레벨)

- **비동기 submit-poll 분리**: submit warp와 poll warp를 분리하여 submission pipeline 유지
- **Warp-cooperative I/O**: 1 warp = 1 I/O 제출 (warp reduction으로 single thread가 대표 제출), 나머지 threads는 다른 I/O 준비
- **Queue별 전담 warp 감소**: n_queues = n_warps로 고정하여 cross-warp queue contention 제거
- **`__syncwarp()` 제거**: 직렬화된 sq_enqueue와 독립적인 cq_poll을 감안하면 barrier 불필요
