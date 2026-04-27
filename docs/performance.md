# Performance stack analysis

이 문서는 현재 코드에서 성능에 영향을 줄 수 있는 stack을 adversarial하게 점검한
메모다. BaM/fio/nvme-cli submodule이 이 checkout에서는 초기화되어 있지 않아서
BaM 내부 구현은 직접 확인하지 못했다. 따라서 BaM 내부 함수의 세부 비용은
현재 wrapper 호출과 주석에서 드러나는 호출 관계를 기준으로 분류한다.

## Summary

현재 parallelink의 throughput-critical path는 CPU fio callback이 아니라 GPU
kernel 안의 BaM `read_data()` / `write_data()` stack이다. CPU 쪽 `getevents()`는
fio 통계와 job 진행률에는 영향을 주지만, GPU가 NVMe에 I/O를 내는 data path를
직접 막지는 않는다.

가장 먼저 의심할 후보는 세 가지다.

1. Benchmark를 `BUILD_TYPE=Debug`로 돌리는 경우. CUDA `-G -O0`가 들어가므로
   측정값은 production path가 아니다.
2. `plink_io_worker()`의 per-I/O `__syncwarp()`. 필요성이 코드에 증명되어 있지
   않고, 각 warp가 가장 느린 lane에 매 I/O마다 맞춰질 수 있다.
3. `read_data()` / `write_data()` 안의 BaM queue primitive contention. 현재
   32 lanes가 같은 queue pair를 공유하므로 queue lock, cid allocation, CQ polling
   비용이 실제 dominant stack일 가능성이 높다.

## Stack Map

| Stack | Code | Data path 영향 | Optimization 후보 |
| --- | --- | --- | --- |
| Build mode | `build.sh`, `CMakeLists.txt` | 매우 큼. Debug는 host `-O0`, CUDA `-G -O0`로 빌드된다. | 성능 측정은 반드시 `BUILD_TYPE=Release ./build.sh`로 분리한다. 기본값을 Release로 바꾸는 것은 debug workflow와 trade-off가 있어 별도 결정이 필요하다. |
| GPU I/O helper | `plink_io_worker()` -> `read_data()` / `write_data()` | 실제 NVMe submit/complete hot path. | BaM 내부 profile 필요. Nsight Compute에서 device atomic, MMIO, spin wait stall을 우선 본다. |
| Queue mapping | `q_idx = (tid / 32) % n_queues` | warp 단위로 queue pair를 공유한다. queue primitive가 thread-safe일수록 atomic contention 가능성이 커진다. | `gpu_warps`, `n_queues`, `queue_depth` sweep으로 scaling knee를 찾는다. queue당 active lanes를 줄이는 variant를 비교한다. |
| Warp sync | per-I/O `__syncwarp()` | 잠재적으로 큼. 한 lane의 느린 CQ wait가 warp 전체 loop 진행을 늦출 수 있다. | feature macro로 on/off A/B를 만들고 correctness를 먼저 검증한다. 주석의 "required" 원인을 BaM queue invariant로 좁혀야 한다. |
| Done counter | thread-local `pending_done`, 1024 I/O마다 `atomicAdd()` | 낮음. device atomic을 amortize하고 있어 per-I/O 비용은 작다. | stats granularity가 문제일 때만 batch size를 option화한다. throughput 최적화 1순위는 아니다. |
| Shutdown flag | mapped pinned host read 1024 I/O마다 | 낮음. PCIe read지만 amortized다. | Ctrl+C latency가 문제면 polling period를 줄이되 throughput 영향 측정이 필요하다. |
| fio accounting poll | `getevents()` -> `plink_gpu_poll_done()` -> `cudaMemcpyAsync` + `cudaStreamSynchronize` | data path 직접 영향은 낮음. fio progress, reporting latency, CPU overhead에는 영향. | polling interval과 max events를 profile한다. 더 낮은 overhead가 필요하면 host-mapped progress page 또는 less-frequent mirror copy를 검토한다. |
| `io_u` token ring | `fio_plink_queue()` / `fio_plink_event()` | GPU I/O는 계속 돌지만 fio-visible completion rate와 job accounting에는 영향. | token ring을 fio `iodepth`와 분리할지 검토한다. 단, fio accounting semantics가 바뀌므로 조심해야 한다. |
| Page cache sizing | fixed `8192` pages of 4 KiB | 현재 최대 `gpu_warps=128`이면 4096 threads라 collision은 없다. 더 큰 launch에는 collision 가능. | `total_threads`와 `queue_depth` 기반 sizing으로 바꾼다. |
| Admin bridge | Unix socket helper -> `nvm_raw_rpc()` | throughput data path 아님. admin command를 동시에 많이 쏘지 않는 한 영향 낮음. | 종료 안정성은 중요하지만 throughput tuning 대상은 아니다. |

## Detailed Notes

### 1. Debug build is not a benchmark build

`build.sh`의 기본값은 `BUILD_TYPE=Debug`다. `CMakeLists.txt`는 Debug에서 CUDA에
`-G -g -O0`를 넣고, BaM patch도 Debug일 때 device debug flags를 넣는다.
CUDA `-G`는 device optimization을 크게 제한하므로 이 상태에서 얻은 IOPS는
엔진 구조의 한계가 아니라 debug artifact일 수 있다.

결론: 성능 숫자는 `BUILD_TYPE=Release`로만 비교해야 한다. 이 항목은 코드
hot path가 아니라 build stack 문제지만, 실제 측정에는 가장 큰 영향을 줄 수 있다.

### 2. `__syncwarp()` is the most suspicious local code path

현재 kernel은 I/O 한 번마다 `__syncwarp()`를 호출한다.

```c
read_data(pc, qp, slba, wl.n_blocks, pc_entry);
...
__syncwarp();
```

주석은 "required"라고 되어 있지만 어떤 memory ordering, queue invariant, BaM
precondition 때문인지 설명하지 않는다. 만약 BaM helper가 lane-independent하게
동작한다면 이 barrier는 불필요한 warp-level serialization이다. 반대로 BaM queue
primitive가 warp-cooperative 동작을 전제한다면 제거하면 correctness가 깨질 수 있다.

결론: 바로 제거하면 안 된다. `PLINK_SYNCWARP_EACH_IO` 같은 compile-time switch로
A/B를 만들고, data integrity와 hang 여부를 먼저 검증한 뒤 성능을 비교해야 한다.

### 3. BaM queue contention is probably dominant

현재 mapping은 warp 하나가 queue pair 하나를 공유하는 형태다.

```c
int q_idx = (tid / 32) % n_queues;
QueuePair *qp = &(ctrls[0]->d_qps[q_idx]);
```

그러나 실제 I/O helper 호출은 lane 0만 하는 구조가 아니라 모든 GPU thread가
호출한다. 따라서 한 warp의 32 lanes가 같은 `QueuePair`에서 command id allocation,
SQ enqueue, CQ poll, CQ dequeue를 동시에 시도할 가능성이 있다. BaM primitive가
device-wide atomics와 backoff를 쓴다면 이 contention이 throughput curve의
dominant stack일 수 있다.

결론: `gpu_warps == n_queues`만 볼 것이 아니라 queue당 active thread 수를 바꾸는
실험이 필요하다. 예를 들어 one-thread-per-queue, one-warp-per-queue,
multi-warp-per-queue를 나눠 비교해야 병목 위치가 보인다.

### 4. CPU polling is not the I/O bottleneck, but can distort fio results

`fio_plink_getevents()`는 매 polling마다 device counter 8 bytes를 copy stream으로
복사하고 synchronize한다. 새 event가 없으면 1 ms sleep한다. 이 stack은 GPU가 NVMe에
I/O를 내는 것을 직접 막지 않는다. 하지만 fio가 볼 수 있는 completion cadence와
runtime 종료 시점의 accounting에는 영향을 준다.

특히 `max`가 queued `io_u` token 수로 clamp되기 때문에 GPU가 이미 많은 I/O를
끝냈더라도 fio는 ring에 들어간 token 수만큼만 한 번에 수확한다. 이 설계는 fio
accounting을 보존하기 위한 것이지만, reporting latency와 apparent burstiness를
만든다.

결론: throughput 자체보다 fio-visible metrics 검증 대상이다. GPU-side hardware
counter와 fio output을 같이 비교해야 한다.

### 5. Done counter batching is a reasonable trade-off

`pending_done`을 1024 I/O마다 device atomic으로 더하는 것은 per-I/O atomic 비용을
줄이는 좋은 선택이다. 이 값이 크면 fio 통계 갱신이 둔해지고, 작으면 atomic traffic이
늘어난다.

결론: 현재 값은 합리적인 출발점이다. 낮은 queue depth 또는 낮은 IOPS workload에서
progress가 뭉쳐 보이면 option화할 수 있지만, throughput 최적화의 첫 후보는 아니다.

## Recommended Measurement Plan

1. `BUILD_TYPE=Release`와 `BUILD_TYPE=Debug`를 분리해 baseline을 만든다.
2. `gpu_warps`, `n_queues`, `queue_depth` sweep을 하고 scaling knee를 찾는다.
3. Nsight Compute에서 kernel stall reason, atomic throughput, memory dependency,
   barrier stall을 확인한다.
4. `__syncwarp()` A/B switch를 만들어 correctness test 후 IOPS를 비교한다.
5. BaM 내부 queue primitive의 spin/backoff count를 계측한다.
6. fio output과 GPU `d_done_count` final value를 같이 기록해 accounting drift를 본다.

## Current Verdict

현재 코드 안에 CPU가 per-I/O data path에 끼어드는 명백한 병목은 보이지 않는다.
다만 성능 숫자를 망칠 수 있는 stack은 분명히 있다. 가장 위험한 것은 Debug build로
benchmark하는 것, 그 다음은 per-I/O `__syncwarp()`, 그 다음은 queue pair 내부
device atomic contention이다. 이 세 가지를 순서대로 검증하는 것이 비용 대비 가장
좋다.
