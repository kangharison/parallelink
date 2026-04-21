# parallelink 작업 로그 (시행착오 기록)

이 파일은 parallelink 프로젝트에서 이루어지는 **모든 작업의 과정**을 기록한다.
목적은 "완료 체크리스트"가 아니라 **시행착오(試行錯誤) 로그**다 — 나중에 같은 문제를
다시 만났을 때 과거의 시도·실패·원인·해결 경로를 재구성할 수 있어야 한다.

작성 규칙과 엔트리 템플릿은 [`CLAUDE.md`](./CLAUDE.md)의 "작업 기록 규칙" 섹션 참조.

원칙 요약:
- 실패한 시도도 전부 남긴다 (결과만 적지 않는다).
- 에러 메시지/명령어 출력은 원문 그대로 붙여넣는다.
- 시도할 때마다 실시간으로 덧붙인다 (사후 요약 금지).

---

## 2026-04-13 — 작업 로그 체계 구축

**상태**: 완료
**목표**: parallelink 작업의 시행착오를 나중에 재구성할 수 있도록 로그 파일과 기록 규칙을 마련한다.

### 시도 기록
1. **시도 1**: `work.md`를 단순 완료 체크리스트 포맷으로 초안 작성 (`[완료]/[진행 중]` 한 줄 항목).
   - 변경: `work.md` 신규 생성, `CLAUDE.md`에 "작업 기록 규칙" 섹션 추가.
   - 결과: 사용자 피드백 —
     ```
     해당 작업은 시행착오에 대해 정리하기 위함이야.
     현재처럼 작업을 한다면 추후에 정리할 수 없기 때문에
     ```
   - 가설: 상태/완료 여부 중심 포맷은 "무엇을 시도해봤고 왜 실패했는지"를 담지 못해
     추후 회고·재현이 불가능하다. 로그의 목적 자체가 잘못 잡혔다.

2. **시도 2**: 로그의 목적을 "시행착오 보존"으로 재정의하고 템플릿 재설계.
   - 변경: `CLAUDE.md` "작업 기록 규칙" 섹션을 전면 재작성 — 목표/시도/원인/해결/교훈을
     필수 필드로 지정, 실패 시도와 에러 원문 보존을 MUST로 명시. `work.md`도 동일 포맷으로 갱신.
   - 결과: 엔트리 템플릿이 과정 전체(가설→시도→실패→원인→교훈)를 포함하게 됨.

### 원인 분석
초기 설계 시 "작업 진행 상황 관리"(태스크 트래커)와 "시행착오 보존"(랩 노트북)을 구분하지 않았다.
사용자의 의도는 후자였음. 전자는 TodoWrite/이슈 트래커 영역, 후자는 별도 영속 파일이 필요한 영역.

### 해결
`CLAUDE.md`의 규칙을 "시행착오 로그"로 재정의하고 필수 8개 필드(날짜·목표·시도·원인·해결·교훈·상태·관련)
를 MUST로 박아 두었다. `work.md`도 동일 템플릿으로 본 엔트리부터 시작.

### 교훈
- 로그 파일을 새로 만들 때는 **"완료 체크용인가, 회고용인가"** 를 먼저 정하고 포맷을 고른다.
  두 용도는 필요한 필드가 완전히 다르다 — 상태 플래그 vs. 서사(narrative).
- 사용자가 "정리"라는 단어를 쓰면 사후 회고/재현 가능성을 요구하는 경우가 많다. 단순 목록이 아닌
  과정 기록 포맷을 기본값으로 고려할 것.
- 실시간 기록 규칙을 MUST로 두지 않으면 "끝난 뒤 한 번에 요약"되어 중간 시도가 유실된다.

### 관련
- 파일: `CLAUDE.md`, `work.md`
- 커밋: 미커밋
- 참고: 상위 `../CLAUDE.md` 주석 방법론 (직접 연관은 없음)

---

## 2026-04-13 — `build.sh` 실행 시 `g++11: no such file or directory` / `nvcc fatal: failed to preprocess host compiler properties`

**상태**: 완료
**목표**: `./build.sh`로 fio + libnvm + parallelink 전체 빌드 성공.

### 증상
사용자가 `./build.sh` 실행 시 다음 에러 발생:

```
g++11: no such file or directory
nvcc fatal: failed to preprocess host compiler properties
```

주의: 에러 메시지의 컴파일러 이름이 `g++-11`이 아니라 **`g++11`** (하이픈 없음). 이것이 핵심 단서.

### 시도 기록
1. **시도 1**: 시스템에 `g++-11`이 실제 있는지 확인.
   - 명령: `ls /usr/bin/gcc-* /usr/bin/g++-*`
   - 결과: `/usr/bin/gcc-11`, `/usr/bin/g++-11`, `/usr/bin/gcc-13`, `/usr/bin/g++-13` 모두 존재.
   - 가설: 컴파일러는 설치돼 있음 → 파일 부재가 원인이 아님.

2. **시도 2**: nvcc가 `-ccbin g++-11` 인자로 정상 동작하는지 단독 재현.
   - 명령:
     ```
     echo 'int main(){return 0;}' > /tmp/t.cu
     /usr/local/cuda/bin/nvcc -ccbin g++-11 /tmp/t.cu -o /tmp/t
     /usr/local/cuda/bin/nvcc -ccbin /usr/bin/g++-11 /tmp/t.cu -o /tmp/t
     ```
   - 결과: 두 경우 모두 에러 없이 성공.
   - 가설: nvcc + CUDA 13.2 + GCC 11 조합 자체는 정상. `build.sh`가 실제로 nvcc에 넘기는 문자열이 `g++-11`이 아닐 가능성이 큼.

3. **시도 3**: `build.sh`의 env fallback 로직(`CXX="${CXX:-g++-11}"` 등)을 검토.
   - 발견: 쉘 env에 `CXX=g++11`(하이픈 누락) 같은 값이 이미 있으면, `:-` fallback은 덮어쓰지 않으므로 그대로 nvcc `-ccbin`에 전달됨.
   - 가설(현재 가장 유력): **사용자 쉘의 환경변수에 `g++11` 오타가 들어 있다.** `~/.bashrc`/`~/.zshrc`/`~/.profile` 또는 이전 세션에서 `export CXX=g++11` 류 설정이 있었을 가능성.

4. **시도 4 (원인 확정)**: 사용자 확인 — 실제 빌드 PC에는 **gcc 12.4.0만 설치**돼 있음. `gcc-11`/`g++-11`은 부재.
   - 결과: `build.sh`의 하드코딩 기본값(`gcc-11`/`g++-11`)이 nvcc `-ccbin`으로 전달 → 바이너리 없음 → nvcc가 `failed to preprocess host compiler properties` 출력.
   - (초기 "g++11 vs g++-11" 하이픈 가설은 폐기. 이전 조사 환경은 샌드박스였고 빌드 PC와 무관했다 — 조사 범위 착오.)

### 원인 분석 (확정)
`build.sh`가 호스트 컴파일러 기본값을 `gcc-11`/`g++-11`로 하드코딩하고 있었으나, **실제 빌드 PC에는 gcc 12.4.0만 설치**돼 있어 해당 바이너리가 존재하지 않았다. nvcc는 존재하지 않는 `-ccbin` 대상을 exec 시도하다 실패했고, 그 결과가 `failed to preprocess host compiler properties` 에러. CUDA 13.2는 GCC 12를 공식 지원하므로 컴파일러 버전 자체엔 문제 없음 — 단순히 스크립트 기본값이 실제 환경과 어긋났던 것.

### 해결
`build.sh` 두 가지 수정:
1. 기본값을 `gcc-12`/`g++-12`로 변경 (헤더 주석, `CC`/`CXX`/`CUDA_HOST_CXX` 세 변수).
2. 설정 직후 `command -v`로 세 컴파일러 존재를 사전 검증 — 없으면 nvcc에 도달하기 전에 명확한 메시지로 중단. 이로써 향후 동일 클래스 에러(env 오타, 바이너리 부재)가 의미 불명의 nvcc 에러로 둔갑하지 않는다.

### 교훈
- **조사 환경과 실행 환경의 구분을 먼저 확인하라.** 이번엔 AI가 자기 샌드박스의 `/usr/bin/gcc-*` 출력을 보고 "g++-11이 존재하니 원인은 env 오타"라 추정했으나, 사용자의 빌드 PC는 전혀 다른 머신이었다. 환경 가정을 세울 땐 "이 명령을 어느 머신에서 실행했는가"를 반드시 확인할 것.
- `build.sh`처럼 툴체인을 하드코딩하는 스크립트는 **바이너리 존재 검증을 앞단에 두어야 한다.** nvcc 같은 하위 도구의 에러 메시지는 원인으로부터 한두 단계 떨어져 있어 디버깅이 느려진다 (`failed to preprocess host compiler properties` ← 실제 원인은 단순 "파일 없음").
- `:-` fallback은 env가 비어 있을 때만 기본값을 쓴다. 즉 "env 오염" 가설과 "기본값 부적절" 가설은 모두 동일 증상을 만든다 — 에러만으로 구분이 안 되므로 **빌드 PC의 컴파일러 실제 설치 상태 확인이 항상 1단계**.
- CMakeCache는 `BUILD_TYPE`이 같으면 재구성되지 않는다. 컴파일러 관련 에러 디버깅 시 `build/`·`extern/bam/build/` 캐시 제거가 필수.

### 관련
- 파일: `build.sh` (기본값 gcc-12 변경 + 컴파일러 존재 사전 검증 추가)
- 커밋: 미커밋
- 참고: CUDA 13.2 Release Notes — host compiler 지원 GCC 12 포함.

---

## 2026-04-13 — [2/4] BaM CMake configure 재발 (gcc-12 기본값 적용 후에도 동일)

**상태**: 진행 중 (실제 로그 미수신)
**목표**: [2/4] BaM configure 실패 원인 확정.

### 시도 기록
1. **시도 1**: 정황상 "[2/4]에서 configure incomplete"만 전달받음. 실제 에러 본문(CMake Error / nvcc fatal 원문) 미확보.
   - 결과: 원인 특정 불가. 로그 재요청.
   - 가설: (a) `gcc-12`/`g++-12` 바이너리가 실제로 없는데 `build.sh`의 pre-check 추가 전 캐시가 남아 통과됐거나, (b) Ubuntu에 `gcc` = 12.4.0은 있지만 `gcc-12`(하이픈)라는 패키지는 미설치라 pre-check가 "not found"로 멈춰 있는데 사용자가 이를 [2/4] 실패로 오인.

2. **시도 2**: BaM `extern/bam/CMakeLists.txt` 확인.
   - 결과: `project(libnvm LANGUAGES CUDA C CXX)`만 있고 컴파일러 하드코딩 없음.
   - 결론: BaM 측 설정 문제가 아님. [2/4] 실패는 CMake의 CUDA 컴파일러 프로브 자체에서 발생.

3. **시도 3 (예방적 개선)**: `build.sh`를 자동 fallback 구조로 리팩터.
   - 변경:
     - `gcc-12`/`g++-12`가 있으면 우선 사용, 없으면 시스템 기본 `gcc`/`g++`로 fallback.
     - `CXX -dumpversion`으로 major를 뽑아 8 ≤ major ≤ 14 아니면 중단 (CUDA 13.x 지원 범위).
     - 선택된 컴파일러를 `selected CXX = ...`로 출력해 nvcc에 실제 넘어가는 값 가시화.
   - 의도: "사용자 빌드 PC 기본 `gcc`가 12.4.0"이라는 조건에서 `gcc-12` 패키지 추가 설치 없이도 빌드 성공.

### 원인 분석 (미확정)
실제 로그 없이 추정 중. 다음 중 하나 이상이 유력:
- `gcc-12` 패키지 미설치 (Ubuntu 기본 `gcc`는 있지만 하이픈 붙은 버전 심볼릭은 별도 패키지).
- CUDA_ARCHS="100;120" (Blackwell sm_100/sm_120) — CUDA 13.2는 지원하므로 낮은 가능성이지만, 드라이버/툴킷 mismatch 시 프로브 단계에서 실패 가능.
- stale `extern/bam/build/CMakeCache.txt`가 남아 옛 컴파일러 경로로 재시도.

### 다음 단계 (사용자 확인 필요)
1. `which gcc-12 g++-12` 출력.
2. `find extern/bam/build -name CMakeError.log -exec tail -80 {} \;` 결과.
3. `./build.sh 2>&1 | tee /tmp/plink-build.log` 후 `grep -B2 -A20 -iE 'error|fatal' /tmp/plink-build.log` 의 전체 출력.

### 교훈 (잠정)
- **"동일한 이슈"라는 사용자 보고는 재현 검증이 필요하다.** 실제로 동일한 메시지인지 확인 없이 "gcc-12 기본값 적용이 안 됐다"고 단정하면 엉뚱한 방향으로 간다. 에러 원문 확보가 모든 디버깅의 선행 조건.
- Ubuntu의 `gcc` vs `gcc-N`은 **같은 설치가 아니다.** 후자는 별도 패키지. 버전 문자열이 12.x라고 `gcc-12`가 PATH에 있으리라 가정하지 말 것.

### 관련
- 파일: `build.sh` (자동 fallback + 버전 검증 추가)
- 커밋: 미커밋
- 참고: 이전 엔트리("build.sh 실행 시 `g++11` 에러")의 연속 건.

---

## 2026-04-13 — 원인 확정: CMake가 읽는 `CUDAHOSTCXX` env의 오타

**상태**: 해결 제안 (사용자 검증 대기)
**목표**: `-ccbin=g++11` 문자열이 어디서 유입되는지 특정하고 재발 방지.

### 결정적 단서
사용자가 준 실제 에러:
```
CMake Error at /usr/share/cmake-3.28/Modules/CMakeDetermineCompilerId.cmake:780 (message):
  Compiling the CUDA compiler identification source file "CMakeCUDACompilerId.cu" failed.
  Compiler: /usr/local/cuda/bin/nvcc
  Build flags:
  Id flags: --keep;--keep-dir;tmp;-ccbin=g++11 -v
```

`-ccbin=g++11` (하이픈 없음) — `build.sh`는 이 문자열을 생성하지 않음 (기본값은 `g++-12` 또는 시스템 `g++`).
→ 값이 **스크립트 바깥**에서 주입되고 있다는 뜻.

### 원인 분석 (확정 근거)
CMake는 `CMAKE_CUDA_HOST_COMPILER`가 CLI/cache로 명시되지 않을 때 **환경변수 `CUDAHOSTCXX`** 를 초기값으로 읽는다 (CMake 공식 문서에 명시된 동작). 동일 패턴으로 `CUDACXX`도 읽어 `CMAKE_CUDA_COMPILER` 초기화.

**그런데 `build.sh`는 `-DCMAKE_CUDA_HOST_COMPILER=...`로 명시하고 있다.** 즉 env의 우선순위는 CLI보다 낮을 텐데 왜 env 값이 이겼는가?

두 가지 가능성:
1. (가장 유력) CMake의 CUDA 컴파일러 **식별(identification) 단계**는 CLI 플래그가 CMakeCache에 적용되기 **전에** 실행되며, 이 프로브 단계에선 env `CUDAHOSTCXX`가 적용될 수 있다. 프로브가 실패하면 CLI의 값이 쓰일 기회조차 없다.
2. `extern/bam/build/CMakeCache.txt`가 stale하게 `g++11`을 캐시. — 가능성 낮음 (사용자가 `rm -rf` 수행한 경우).

어느 쪽이든 **env `CUDAHOSTCXX=g++11`의 존재가 필요충분 조건**. 사용자의 `~/.bashrc` 또는 이전 세션 export가 원흉으로 추정.

### 해결
1. 사용자 측:
   - `env | grep -iE 'CUDAHOSTCXX|CUDACXX'` 로 존재 확인.
   - `~/.bashrc`/`~/.zshrc`/`~/.profile`에서 `export CUDAHOSTCXX=g++11` (또는 유사 오타) 라인 제거.
   - 새 쉘에서 `rm -rf build extern/bam/build && ./build.sh`.

2. `build.sh` 개선 (재발 방지):
   - `selected CXX` 출력 직후 `export CUDAHOSTCXX="${CUDA_HOST_CXX}"`, `export CUDACXX="${NVCC}"`, `unset CCBIN` 추가.
   - 이제 쉘에 오타 env가 남아 있어도 스크립트 내부에서 스크립트가 선택한 값으로 덮어쓰므로, CMake 프로브 단계에서도 올바른 값이 보인다.

### 교훈
- **CMake는 환경변수를 조용히 읽는다.** `CUDAHOSTCXX`, `CUDACXX`, `CC`, `CXX`, `CFLAGS`, `CXXFLAGS`, `LDFLAGS` 등은 CLI 명시가 없으면 초기값으로 사용된다. 빌드 스크립트가 호출자 env로부터 **완전히 격리**되길 원하면 명시적으로 export/unset 해야 한다.
- nvcc `-ccbin=<X>`의 `<X>`는 CMake의 `CMAKE_CUDA_HOST_COMPILER` 값 그대로다. 에러에 찍힌 `-ccbin=...` 문자열은 **원인 변수 이름을 역추적하는 강력한 단서**다. 메시지에 있는 문자열을 그대로 grep해 원흉을 찾아야 한다.
- **"build.sh에서 명시했으니 안전"이라는 가정은 틀릴 수 있다.** CMake의 컴파일러 식별 단계는 설정(configure) 앞단에서 실행되며, 이 단계의 입력은 env가 지배한다. 스크립트 격리는 `export`를 포함해 재작성하는 것이 안전하다.
- 삽질 요약: (1) "env 오염" 가설을 먼저 세웠다 → 샌드박스 조사로 철회 → (2) "gcc-11 하드코딩" 가설로 전환해 코드 수정 → (3) 그래도 재발 → (4) 실제 에러 로그(`-ccbin=g++11`)를 받고 나서야 (1)번 가설이 처음부터 옳았음이 드러남. **에러 원문을 1번째 시도에서 받아냈어야 했다** — 시도 2~4가 전부 비용.

### 관련
- 파일: `build.sh` (CUDA env export 고정 추가)
- 커밋: 미커밋
- 참고: CMake 공식 문서 — "Environment Variables for Languages" / `CUDAHOSTCXX`.

---

## 2026-04-13 — cuda-gdb / VSCode 디버깅 셋업 정리

**상태**: 완료 (동작 확인)
**목표**: VSCode에서 cuda-gdb로 parallelink(호스트 + GPU 커널)를 디버깅 가능한 상태로 만들기까지 필요한 작업을 일괄 정리한다. 실제 겪은 시행착오(`-var-create: unable to create variable object` 등)를 정리해 다음 환경 셋업 시 재삽질을 방지.

### 체크리스트 — cuda-gdb 디버깅을 시작하려면 필요한 것

아래 순서로 모두 충족해야 VSCode Variables/Watch 패널에서 변수가 보이고 브레이크포인트가 적중한다.

#### 1. 툴체인
- [ ] CUDA Toolkit 13.x 설치, `/usr/local/cuda/bin/cuda-gdb` 존재.
- [ ] 호스트 컴파일러는 GCC 12 (또는 CUDA 지원 범위 8~14) — `build.sh`가 자동 선택.
- [ ] 쉘 env에 `CUDAHOSTCXX`/`CUDACXX` 오타가 없을 것 (있으면 CMake의 CUDA 프로브가 잘못된 `-ccbin`을 쓴다 — 이전 엔트리 참조).

#### 2. Debug 빌드
- [ ] `BUILD_TYPE=Debug`로 빌드 (현재 `build.sh` 기본값). 재구성 필요 시:
  ```bash
  rm -rf build extern/bam/build
  ./build.sh
  ```
- [ ] parallelink `CMakeLists.txt`가 Debug에서 `-G -g -O0`을 CUDA flags에 붙이는지 확인 (현재 걸려 있음).
- [ ] **fio는 별도로 debug 심볼 필요** — fio의 기본 `./configure`는 `-g`를 적극 넣지 않는다. 호스트 측 fio 변수를 보려면:
  ```bash
  cd extern/fio
  make clean
  ./configure --extra-cflags="-g -O0"
  make -j
  ```
  `readelf -S fio | grep debug`로 debug 섹션 존재 확인.
- [ ] libnvm(BaM)도 Debug로 빌드되는지 — `build.sh`가 `CMAKE_BUILD_TYPE=${BUILD_TYPE}`을 넘기므로 같이 Debug.

#### 3. VSCode 설정
- [ ] `.vscode/launch.json`에 `"type": "cuda-gdb"` 구성이 존재 (현재 `parallelink: CUDA + C/C++ (cuda-gdb)`).
- [ ] `debuggerPath: /usr/local/cuda/bin/cuda-gdb` 정확.
- [ ] "NVIDIA Nsight Visual Studio Code Edition" 확장 설치 (cuda-gdb 타입 지원).
- [ ] `environment`에 `FIO_EXT_ENG_DIR` (parallelink.so 위치), `LD_LIBRARY_PATH` (libnvm.so 위치) 포함.

#### 4. 실행 환경
- [ ] libnvm 커널 모듈 로드: `sudo insmod dist/libnvm.ko`.
- [ ] 대상 NVMe를 libnvm 드라이버에 bind (`nvme` 드라이버에서 unbind 후 `libnvm`에 bind).
- [ ] `/dev/libnvm0` 존재 확인.

#### 5. 디버깅 사용법 (증상별 대응)

**`-var-create: unable to create variable object` 에러**
- 브레이크포인트에서 **실제로 멈춘 뒤** Watch/Variables 재시도. `.so`는 dlopen 이후에만 심볼이 보인다.
- 디버그 콘솔에서 `-exec info locals`로 gdb가 변수를 아는지 1차 확인.
  - `No symbol "..."` → 디버그 심볼 부재 → 해당 모듈 `-g` 재빌드.
  - `value optimized out` → `-O0` 빌드 확인.
  - `cannot access memory` → PC가 아직 변수 초기화 전 → 한 줄 step 후 재시도.
- GPU 변수는 **cuda-gdb 포커스가 device thread**여야 함:
  ```
  -exec info cuda kernels
  -exec cuda thread
  ```

**호스트 측 변수만 보고 싶을 때**
- `parallelink: host only (gdb)` 구성 (cppdbg) 사용. cuda-gdb 오버헤드 없이 빠르게 왕복 가능. 단 GPU 커널 내부는 못 들어감.

### 시행착오 요약 (겪었던 것)
1. **Release 바이너리를 Debug로 착각** — `BUILD_TYPE` 미지정 시 Release가 기본이었음. → `build.sh` 기본값을 Debug로 변경 (커밋 `a464875`).
2. **`-ccbin=g++11` 에러로 빌드 자체 실패** — 사용자 쉘 `CUDAHOSTCXX=g++11` 오타. → `build.sh`에서 `export CUDAHOSTCXX`로 덮어쓰기 (커밋 `d8e4d43`).
3. **`-var-create` 에러** — 브레이크포인트 정지 전 Watch 시도, 또는 host 포커스에서 device 변수 조회 등 타이밍/포커스 이슈. → 위 5번 체크리스트로 대응법 정리.

### 교훈
- CUDA 디버깅은 **호스트 `-g`와 device `-G`가 별개**다. 한쪽만 있어도 반대쪽 변수는 안 보인다.
- **fio 서브모듈은 parallelink 빌드 시스템 바깥**이라 CMAKE_BUILD_TYPE의 영향을 안 받는다. 호스트 ioengine 코드를 디버깅하려면 fio를 따로 `--extra-cflags="-g -O0"`로 빌드해야 한다. 이걸 까먹으면 "Debug 빌드했는데 왜 변수가 안 보이지?" 로 한 번 더 삽질.
- `-var-create` 에러는 GDB MI의 포괄적 실패 — 원인은 항상 **"gdb가 그 이름의 심볼을 못 찾는다"** 로 귀결. 디버그 콘솔에서 `-exec info locals`로 gdb 자체 응답을 먼저 본 뒤 상위 IDE 레이어를 의심하는 순서가 빠르다.
- 브레이크포인트가 pending 상태(파란 동그라미)일 때 Watch 시도하면 실패한다. **실제 히트 후**에만 변수 객체가 만들어진다.

### 관련
- 파일: `.vscode/launch.json`, `CMakeLists.txt`, `build.sh`, `extern/fio/configure`
- 커밋: 미커밋 (디버깅 셋업 가이드 정리만, 코드 변경 없음)
- 참고: NVIDIA cuda-gdb 매뉴얼 (focus/thread 명령), CMake `-G` vs `-g` 의미.
