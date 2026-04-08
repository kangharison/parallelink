# parallelink

GPU-direct NVMe I/O engine for [fio](https://github.com/axboe/fio).

GPU persistent kernel이 CPU 개입 없이 자율적으로 NVMe I/O를 submit/complete하는 fio ioengine입니다.
[libnvm](https://github.com/ZaidQureshi/bam) userspace NVMe 드라이버를 활용하여 GPU에서 PCIe P2P로 SSD에 직접 접근합니다.

## Prerequisites

- NVIDIA GPU (Ampere 이상, PCIe P2P 지원)
- CUDA Toolkit 13.x+
- GCC 11 (GCC 13은 libnvm의 freestanding libcxx와 비호환)
- CMake 3.18+
- libnvm kernel module (extern/bam/module)
- Linux kernel 5.x+
- IOMMU disabled

## Build

```bash
# 1. submodule 초기화 (freestanding libcxx 포함)
git submodule update --init --recursive

# 2. fio 빌드 (config-host.h 생성 필요)
cd extern/fio
./configure
make -j$(nproc)
cd ../..

# 3. libnvm(BAM) 빌드
cd extern/bam
mkdir -p build && cd build
cmake .. \
    -Dno_smartio=true \
    -Dno_module=true \
    -Dno_fio=true \
    -DCMAKE_CUDA_ARCHITECTURES="80;90" \
    -DCMAKE_CXX_COMPILER=g++-11 \
    -DCMAKE_C_COMPILER=gcc-11 \
    -DCMAKE_CUDA_HOST_COMPILER=g++-11 \
    -DCMAKE_CXX_FLAGS="-I${PWD}/../include/freestanding/include" \
    -DCMAKE_C_FLAGS="-I${PWD}/../include/freestanding/include"
make -j$(nproc)
cd ../../..

# 4. (선택) libnvm 커널 모듈 빌드 및 로드
cd extern/bam/module
make
sudo insmod libnvm.ko
cd ../../..

# 5. parallelink 빌드
mkdir -p build && cd build
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="80;90" \
    -DCMAKE_CXX_COMPILER=g++-11 \
    -DCMAKE_C_COMPILER=gcc-11 \
    -DCMAKE_CUDA_HOST_COMPILER=g++-11
make -j$(nproc)
```

### Build Notes

- **GCC 11 필수**: libnvm의 freestanding libcxx(`simt/atomic`)가 GCC 13의 엄격한 `chrono::duration` 규칙과 비호환. GCC 11 사용 시 정상 빌드.
- **CUDA arch 80/90**: `compute_70`(Volta)은 CUDA 13.x에서 제거됨. Ampere(80) 이상 지정.
- **freestanding include**: libnvm이 `simt/atomic` 헤더를 사용하며, 이는 `extern/bam/include/freestanding/include`에 위치.

빌드 산출물:
- `parallelink.so` : fio external ioengine
- `plink-holder` : GPU 메모리 상주 데몬

## Usage

### NVMe 디바이스 준비

```bash
# 기존 nvme 드라이버 unbind
echo "0000:xx:xx.x" > /sys/bus/pci/drivers/nvme/unbind

# libnvm에 bind
echo "0000:xx:xx.x" > /sys/bus/pci/drivers/libnvm/bind

# /dev/libnvm0 캐릭터 디바이스 생성 확인
ls -l /dev/libnvm*
```

### (선택) Holder 데몬 실행

반복 테스트 시 NVMe 컨트롤러 리셋 비용을 제거합니다.

```bash
./plink-holder --nvme=/dev/libnvm0 --gpu=0 --queues=32
```

### fio 실행

```bash
# 엔진 라이브러리 경로 지정
export FIO_EXT_ENG_DIR=$(pwd)/build

# fio 실행
./extern/fio/fio --ioengine=external:parallelink.so \
    --gpu_warps=128 \
    --gpu_id=0 \
    --n_queues=32 \
    --nvme_dev=/dev/libnvm0 \
    --rw=randread \
    --bs=4k \
    --size=10G \
    --numjobs=1 \
    --runtime=60 \
    --time_based
```

또는 job 파일:

```ini
; randread_gpu.fio
[global]
ioengine=external:parallelink.so
numjobs=1
time_based
runtime=60

[gpu-randread]
gpu_warps=128
gpu_id=0
n_queues=32
nvme_dev=/dev/libnvm0
rw=randread
bs=4k
size=10G
```

```bash
./extern/fio/fio randread_gpu.fio
```

## Engine Options

| Option | Default | Description |
|--------|---------|-------------|
| `gpu_warps` | 32 | GPU warp 수 (총 스레드 = gpu_warps x 32) |
| `gpu_id` | 0 | CUDA GPU device ID |
| `n_queues` | 16 | NVMe submission/completion queue pair 수 |
| `nvme_dev` | /dev/libnvm0 | libnvm 캐릭터 디바이스 경로 |

## Project Structure

```
parallelink/
├── extern/
│   ├── fio/              fio (git submodule)
│   └── bam/              libnvm + GPU NVMe library (git submodule)
├── include/
│   └── gpu_engine.h      CPU-GPU 공유 구조체 및 인터페이스
├── src/
│   ├── gpu_engine.c      fio ioengine_ops 구현
│   ├── gpu_worker.cu     GPU persistent kernel
│   └── holder.cu         GPU 메모리 상주 데몬
└── CMakeLists.txt
```

## License

TBD
