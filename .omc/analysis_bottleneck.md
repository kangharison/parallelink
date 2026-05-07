# GPU I/O 병목 분석

> 분석 대상: BaM (Block Accelerator for Memory) 기반 GPU NVMe I/O 경로
> 핵심 파일: `nvm_parallel_queue.h`, `queue.h`, `gpu_worker.cu`, `gpu_engine.h`

---

## 1. PCIe Latency 구조

### CPU NVMe 경로 (기준선)

```
CPU core
  │ (캐시 코히어런트, 피코초 단위)
  ▼
PCIe Root Complex
  │ PCIe downstream (1 hop, ~100–500ns)
  ▼
NVMe Controller
  │ (내부 처리)
  ▼
DMA → Host DRAM (CQ 완료)
```

- SQ: Host DRAM (CPU 직접 쓰기 → L3 cache)
- 도어벨: CPU → PCIe Root Complex → NVMe BAR (1 PCIe write)
- CQ 완료: NVMe DMA → Host DRAM (캐시 코히어런트)
- **총 PCIe 트랜잭션: 도어벨 1회 write + CQ DMA 1회**

### GPU BaM NVMe 경로

```
GPU SM
  │ (GPU device memory → SQ entry write, 수백 클록)
  ▼
st.mmio.relaxed.sys.global.u32  ← 도어벨 ring (nvm_parallel_queue.h:303, 481)
  │ PCIe upstream (GPU→CPU root complex, hop 1)
  ▼
Host PCIe Root Complex
  │ PCIe downstream (root complex→NVMe, hop 2)
  ▼
NVMe Controller
  │ P2P DMA read: SQ entry from GPU device memory (hop 3: NVMe→GPU)
  ▼
NVMe Controller processes I/O
  │ P2P DMA write: CQ entry to GPU device memory (hop 4: NVMe→GPU)
  ▼
GPU SM polls cq->vaddr (GPU device memory)
```

- SQ/CQ 메모리: GPU device memory에 위치, `createDma()`로 NVMe에 DMA 매핑 (`queue.h:122–124`)
- 도어벨: `cudaHostGetDevicePointer`로 NVMe BAR을 GPU 주소 공간에 매핑 (`queue.h:190–195, 207–212`)
- **총 PCIe 트랜잭션: 도어벨 write 2hop + SQ DMA read (NVMe←GPU) + CQ DMA write (NVMe→GPU) = 최소 4회 PCIe 트랜잭션**

### 핵심 차이: GPU-initiated MMIO write

```cuda
// nvm_parallel_queue.h:303
asm volatile ("st.mmio.relaxed.sys.global.u32 [%0], %1;"
              :: "l"(sq->db), "r"(new_db) : "memory");
```

- GPU에서 발행하는 MMIO write는 CPU가 발행하는 것보다 **고유 레이턴시가 높음**
- GPU PCIe 업스트림 포트 → 호스트 루트 컴플렉스 → PCIe 다운스트림 포트 → NVMe BAR
- P2P (GPU↔NVMe) 경로는 IOMMU 검사, PCIe TLP ordering 보장이 추가됨
- CPU 경로 대비 **도어벨 레이턴시: 약 2–5× 증가** (PCIe 왕복 hop 2배)
- NVMe가 SQ 엔트리를 읽을 때도 GPU 메모리에서 P2P DMA — CPU 경로는 Host DRAM (로컬) DMA vs GPU 경로는 PCIe를 통한 원격 DMA

---

## 2. Spin-wait 체인 분석 (read_data / write_data 1회 호출당 barrier 수)

`read_data` / `write_data` 내부 호출 순서: `get_cid` → `sq_enqueue` → `cq_poll` → `cq_dequeue` → `sq_dequeue` → `put_cid`

### sq_enqueue 내 spin barrier

**[Barrier #1] Ticket 대기 — relaxed load**
```c
// nvm_parallel_queue.h:188–201
while ((sq->tickets[pos].val.load(simt::memory_order_relaxed) != id)) {
    __nanosleep(ns);  // ns: 8→16→...→256 exponential backoff
}
```
자신의 티켓 번호가 현재 "turn"이 될 때까지 busy-wait.

**[Barrier #2] Ticket 대기 — acquire load (메모리 펜스용 2차 루프)**
```c
// nvm_parallel_queue.h:204–217
while ((sq->tickets[pos].val.load(simt::memory_order_acquire) != id)) {
    __nanosleep(ns);
}
```
relaxed와 별개로, acquire semantics로 동일 조건을 재확인하는 두 번째 spin loop. 

**[Barrier #3] tail_mark + tail_lock 복합 대기**
```c
// nvm_parallel_queue.h:286–323
cont = sq->tail_mark[pos].val.load(relaxed) == LOCKED;
while(cont) {
    // tail_lock TAS 시도 (실패 시 계속 spin)
    new_cont = sq->tail_lock.fetch_or(LOCKED, acquire) == LOCKED;
    if (!new_cont) {
        move_tail(sq, cur_tail);  // 연속 tail_mark 스캔
        // MMIO doorbell write (PCIe)
        sq->tail_lock.store(UNLOCKED, release);
    }
    cont = sq->tail_mark[pos].val.load(relaxed) == LOCKED;  // 다시 체크
    if (cont) __nanosleep(ns);
}
```
자신의 slot이 tail에 포함되어 도어벨이 울릴 때까지 spin. tail_lock 경합 포함.

### cq_poll 내 spin barrier

**[Barrier #4] CQ 선형 스캔 무한 루프**
```c
// nvm_parallel_queue.h:384–421
while (true) {
    for (size_t i = 0; i < cq->qs_minus_1; i++) {
        // CQ 전체를 선형 스캔하며 search_cid 탐색
    }
    __nanosleep(ns);  // 못 찾으면 sleep 후 재시도
}
```
NVMe가 완료를 기록할 때까지 전체 큐를 반복 스캔.

### cq_dequeue 내 spin barrier

**[Barrier #5] pos_locks 0 대기 — relaxed**
```c
// nvm_parallel_queue.h:429–442
while ((cq->pos_locks[pos].val.load(simt::memory_order_relaxed) != 0)) {
    __nanosleep(ns);
}
```

**[Barrier #6] pos_locks 획득 — acquire (TAS)**
```c
// nvm_parallel_queue.h:445–458
while ((cq->pos_locks[pos].val.fetch_or(1, simt::memory_order_acquire) != 0)) {
    __nanosleep(ns);
}
```

**[Barrier #7] head_mark + head_lock 복합 대기 (CQ doorbell)**
```c
// nvm_parallel_queue.h:466–501
while (cont) {
    new_cont = cq->head_lock.fetch_or(LOCKED, acquire) == LOCKED;
    if (!new_cont) {
        move_head_cq(cq, cur_head, sq);
        // MMIO CQ doorbell write (PCIe)
        cq->head_lock.store(UNLOCKED, release);
    }
    cont = cq->head_mark[pos].val.load(relaxed) == LOCKED;
    if (cont) __nanosleep(ns);
}
```

**[Barrier #8] head 전진 확인 루프**
```c
// nvm_parallel_queue.h:504–532
do {
    // cq->head가 loc_를 넘었는지 확인
    new_head = cq->head.load(simt::memory_order_relaxed);
    __nanosleep(ns);
} while(true);
```

### sq_dequeue 내 spin barrier

**[Barrier #9] SQ head_mark + head_lock 복합 대기**
```c
// nvm_parallel_queue.h:338–376
while (cont) {
    new_cont = sq->head_lock.exchange(LOCKED, acquire) == LOCKED;
    if (!new_cont) {
        move_head_sq(sq, cur_head);
        sq->head_lock.store(UNLOCKED, release);
    }
    cont = sq->head_mark[pos].val.load(relaxed) == LOCKED;
    if (cont) __nanosleep(ns);
}
```

### 요약 테이블

| 번호 | 위치 | Barrier 종류 | MMIO 포함 |
|------|------|-------------|-----------|
| 1 | sq_enqueue | ticket relaxed spin | ✗ |
| 2 | sq_enqueue | ticket acquire spin | ✗ |
| 3 | sq_enqueue | tail_mark + tail_lock + **SQ doorbell** | **✓** |
| 4 | cq_poll | CQ 선형 스캔 무한 루프 | ✗ |
| 5 | cq_dequeue | pos_locks relaxed wait | ✗ |
| 6 | cq_dequeue | pos_locks TAS acquire | ✗ |
| 7 | cq_dequeue | head_mark + head_lock + **CQ doorbell** | **✓** |
| 8 | cq_dequeue | head 전진 확인 루프 | ✗ |
| 9 | sq_dequeue | head_mark + head_lock SQ head advance | ✗ |

**단일 I/O 완료까지 통과해야 하는 spin-wait barrier: 9개**
**그 중 PCIe MMIO write 포함 barrier: 2개 (Barrier #3, #7)**

CUDA warp는 `__nanosleep` 중에도 다른 warp로 컨텍스트 스위치되지 **않는다** — `__nanosleep`은 warp 전체를 sleep시키는 것이 아니라 해당 warp가 다른 instruction을 실행하지 못하도록 clock을 소비하는 방식이므로, 이 9개의 barrier 각각에서 warp의 compute 자원이 낭비된다.

---

## 3. Atomic Serialization (Ticket Contention)

### 코드 구조

```c
// nvm_parallel_queue.h:175
ticket = sq->in_ticket.fetch_add(1, simt::memory_order_relaxed);
uint32_t pos = ticket & (sq->qs_minus_1);
uint64_t id  = get_id(ticket, sq->qs_log2);  // = (ticket / qs) * 2
```

```cuda
// gpu_worker.cu:118–119
int q_idx = (tid / 32) % n_queues;  // warp당 하나의 QueuePair
QueuePair *qp = &(ctrls[0]->d_qps[q_idx]);
```

### 경합 분석

**Warp 단위 queue 공유**: 32 threads가 하나의 `nvm_queue_t sq`를 공유.

`in_ticket.fetch_add(1)` 호출 시:
- 32 threads가 동시에 atomic fetch_add를 경쟁
- Thread 0: ticket=0, id=0, pos=0 → `tickets[0]`이 0이 될 때까지 대기 없이 즉시 진행
- Thread 1: ticket=1, id=0, pos=1 → `tickets[1]`이 0이 될 때 진행 (Thread 0이 `tickets[1]`을 advance해야 함)
- Thread N: ticket=N → Thread N-1이 자신의 SQ write를 완료하고 `tickets[N%qs].fetch_add(1)`을 호출해야 진행 가능

**Serialization chain**: Thread 0 → (완료 후) → Thread 1 → ... → Thread 31
- 평균 대기: warp 내 순서 N번 thread = N × (SQ write time)
- SQ write = 64B command를 GPU DMA memory에 write = 수백 ns
- 32번째 thread의 대기 시간 = 31 × (SQ write time) ≈ 수 µs

**Queue depth 제약**: Queue depth가 32보다 작으면 (`pos = ticket & qs_minus_1`이 wrap-around 발생) 이미 처리되지 않은 slot에 write를 시도하여 추가 대기 발생.

**SIMT 처형 패턴**: GPU SIMT 모델에서 32 threads는 "동시에" 발행되지만, ticket 시스템은 이를 순차적으로 강제한다. 즉 **32× 병렬화 잠재력이 32× 직렬화로 역전**된다.

---

## 4. CQ Poll 선형 스캔 문제

### 코드

```c
// nvm_parallel_queue.h:384–421
while (true) {
    uint32_t head = cq->head.load(simt::memory_order_relaxed);
    for (size_t i = 0; i < cq->qs_minus_1; i++) {
        uint32_t loc    = (head + i) & (cq->qs_minus_1);
        uint32_t cpl_entry = ((nvm_cpl_t*)cq->vaddr)[loc].dword[3];  // 16B 읽기
        uint32_t cid    = (cpl_entry & 0x0000ffff);
        bool     phase  = (cpl_entry & 0x00010000) >> 16;
        if ((cid == search_cid) && (phase == search_phase)) { return loc; }
        if (phase != search_phase) break;  // early exit (단, 순서가 맞을 때만)
    }
    __nanosleep(ns);
}
```

### 문제점

**O(D) per thread per outer loop iteration**:
- Queue depth D (기본값 최대 PLINK_MAX_QUEUE_DEPTH=1024, queue.h에서 sq_size min 적용)
- 매 outer loop마다 D개 CQ entry를 선형 스캔
- 완료가 늦게 오면 수백~수천 번의 outer loop 반복

**공유 메모리 동시 읽기**:
- 32 threads가 동일한 CQ 메모리 영역(`cq->vaddr`)을 동시에 읽음
- CQ 전체 크기 = D × 16B. D=128이면 2KB → 32개의 cache line (64B)
- 32 threads가 각각 독립적으로 동일 2KB를 스캔 → L1/L2 cache hit이지만, 초기 loads는 모두 cold
- Phase bit 불일치 시 early-break가 없으면 D-1개 entry를 전부 읽음

**CQ memory 위치**: GPU device memory (DMA mapped). NVMe 컨트롤러가 PCIe P2P write로 CQ entry를 채움. GPU가 이를 읽는 것 자체는 빠르지만, 완료가 아직 안 온 경우 캐시된 stale value를 읽어 캐시 무효화 이슈 없음 — 그러나 NVMe가 write한 시점에 GPU L2 cache invalidation이 필요하므로 `simt::memory_order_relaxed`에도 불구하고 하드웨어 레벨 캐시 coherence 비용이 발생.

**Multiple threads scanning same CQ**: N threads in same warp, all different `search_cid`, all scanning same CQ:
- 총 메모리 reads = N × D per outer iteration
- D=128, N=32: 4096회 16B reads = 64KB/iteration
- 모두 같은 2KB 범위에 집중 → L1 cache에 fit하지만 warp divergence (각 thread가 다른 entry에서 hit) 발생
- 완료가 분산 도착하면 일부 thread가 수천 번 outer loop를 돌 때까지 나머지도 같은 CQ를 반복 스캔

---

## 5. tail_lock / head_lock 직렬화

### tail_lock (SQ 도어벨 직렬화)

```c
// nvm_parallel_queue.h:288–311
bool new_cont = sq->tail_lock.load(simt::memory_order_relaxed) == LOCKED;
if (!new_cont) {
    new_cont = sq->tail_lock.fetch_or(LOCKED, simt::memory_order_acquire) == LOCKED;
    if (!new_cont) {
        uint32_t tail_move_count = move_tail(sq, cur_tail);
        if (tail_move_count) {
            // PCIe MMIO write (도어벨)
            asm volatile ("st.mmio.relaxed.sys.global.u32 [%0], %1;"
                          :: "l"(sq->db), "r"(new_db) : "memory");
            sq->tail.store(new_tail, simt::memory_order_release);
        }
        sq->tail_lock.store(UNLOCKED, simt::memory_order_release);
    }
}
```

- **디바이스 전역 TAS(Test-And-Set) spinlock**: 하나의 큐 전체에서 오직 1개 thread만 동시에 도어벨을 ring 가능
- `move_tail()`은 연속적으로 LOCKED된 `tail_mark` 슬롯을 스캔하여 배치 제출을 시도하지만, lock 자체는 여전히 단일 thread 독점
- 32 threads × multiple warps가 같은 큐를 공유할 경우 모든 thread가 이 lock을 경합
- MMIO write 자체가 ~수 µs이므로, lock 보유 시간이 길고 경합 심화
- **Unfair TAS**: fetch_or 기반이라 공정성 보장 없음 → 특정 thread가 계속 실패하여 tail_mark가 영구히 LOCKED 상태 유지 가능

### head_lock (CQ/SQ head advance 직렬화)

```c
// nvm_parallel_queue.h:468 (cq_dequeue)
bool new_cont = cq->head_lock.fetch_or(LOCKED, simt::memory_order_acquire) == LOCKED;

// nvm_parallel_queue.h:340 (sq_dequeue)
bool new_cont = sq->head_lock.exchange(LOCKED, simt::memory_order_acquire) == LOCKED;
```

- CQ head_lock: CQ 도어벨을 ring하고 head pointer를 advance하는 직렬화 포인트
- SQ head_lock: SQ head를 advance하여 NVMe에 "완료된 SQ 슬롯이 있음"을 알리는 포인트
- **각 I/O마다 2개의 head_lock 경합** (SQ + CQ)
- head_lock을 획득한 thread는 `move_head_cq()` 내에서 또 다른 PCIe MMIO write 수행

### move_tail / move_head_cq / move_head_sq 내부 스캔

```c
// nvm_parallel_queue.h:60–79 (move_tail)
while (pass) {
    pass = (((cur_tail+count+1) & qs_minus_1) != (head & qs_minus_1));
    if (pass) {
        pass = (tail_mark[(cur_tail+count)&qs_minus_1].val.exchange(UNLOCKED, relaxed)) == LOCKED;
        if (pass) count++;
    }
}
```

- lock 보유 중에 추가적인 CAS 연산 반복 수행
- 다른 thread들은 이 기간 동안 tail_lock spin 지속 → warp stall amplification

---

## 6. 요약: 왜 GPU가 CPU보다 느린가

### 근본 원인 계층 분석

#### [원인 1] PCIe 경로의 비대칭성 (레이턴시 2–5× 증가)

| 경로 구성요소 | CPU | GPU (BaM) |
|---|---|---|
| SQ write | Host DRAM 직접 (ns 단위) | GPU device mem → PCIe P2P DMA (수백ns) |
| 도어벨 write | CPU → PCIe root → NVMe (1 hop) | GPU → PCIe upstream → root → NVMe (2 hop) |
| SQ fetch (NVMe 측) | NVMe DMA from Host DRAM | NVMe P2P DMA from GPU mem (PCIe 재진입) |
| CQ write (NVMe 측) | NVMe DMA to Host DRAM | NVMe P2P DMA to GPU mem |
| CQ read | CPU cache coherent | GPU polling |

**결과**: GPU 경로의 PCIe 트랜잭션 수 ~4×, MMIO write 레이턴시 ~2× → **전체 I/O 레이턴시 베이스라인 2–5× 높음**

#### [원인 2] 9개 Spin-wait Barrier가 GPU Warp를 점유

- CPU의 경우 I/O 대기 중 OS가 다른 프로세스로 컨텍스트 스위치 → CPU 활용률 유지
- GPU warp는 `__nanosleep` 동안 **다른 연산을 수행하지 못하고** clock cycle을 소모
- 9개 barrier × NVMe 레이턴시(~100µs) = 수 ms의 warp stall per I/O
- 32 warp 중 대부분이 spin 상태이면 SM utilization 실제로 낮지만, 하드웨어는 바쁜 것처럼 보임 (throughput ≠ utilization)

#### [원인 3] SIMT 32-thread가 Ticket 시스템으로 순차화됨

- GPU의 강점: 32 threads 병렬 실행
- BaM ticket 시스템: 이를 1-thread-at-a-time 직렬 실행으로 강제
- **32× 병렬화 이득이 32× 직렬화 패널티로 상쇄**
- 큐 하나당 actual throughput = (1 thread의 throughput) / 32 이하

#### [원인 4] CQ 선형 스캔의 O(D×T) 메모리 트래픽

- 32 threads × queue depth D entries × N iterations = 32×D×N reads
- CPU의 경우 CQ를 head부터 순서대로 처리 → O(1) per completion
- GPU: 각 thread가 자신의 CID를 찾기 위해 전체 CQ를 O(D) 스캔
- D=128일 때 single warp iteration = 4096회 읽기

#### [원인 5] 직렬화 Lock이 PCIe MMIO Write를 포함

- tail_lock, head_lock 획득 thread만 도어벨을 ring 가능
- 도어벨 ring = PCIe MMIO write (~수 µs)
- Lock 보유 시간 = PCIe write time → 다른 31개 threads가 이 시간 동안 전부 spin
- CPU는 이 문제가 없음: 단일 CPU core가 도어벨을 ring → 경합 자체가 없음

### 종합: Latency vs. Throughput 관점

```
CPU NVMe I/O per operation:
  SQ write (1ns) + doorbell (500ns) + NVMe process (~100µs) + CQ read (1ns)
  ≈ 100µs total, O(1) barriers

GPU BaM I/O per operation (32 threads/queue):
  [ticket wait] + [sq write] + [tail_lock + MMIO doorbell]
  + [cq_poll scan × D] + [pos_lock] + [head_lock + MMIO doorbell]
  + [head advance check]
  × 9 barrier × serialization factor 32
  ≈ 수 ms effective per-thread latency
```

**결론**: BaM의 GPU I/O 경로는 GPU의 SIMT 병렬 실행 모델에 맞게 설계되지 않았다. Ticket 기반 직렬화, 전역 TAS spinlock, 선형 CQ 스캔, PCIe 왕복이 더 많은 MMIO write — 이 모든 요소가 GPU thread 32개를 CPU 1개 core보다 느리게 만든다. GPU가 I/O를 빠르게 하려면 thread당 독립 큐, lock-free 완료 통보, warp-cooperative batch submission이 필요하다.
