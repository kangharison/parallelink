/*
 * compare_random.cu — compare old stride-based vs curand() LBA generation
 *
 * Runs both methods on the GPU and writes raw LBA arrays to binary files.
 * A companion Python script (compare_random.py) reads them and draws histograms.
 *
 * Build:
 *   nvcc -o compare_random compare_random.cu -lcurand -O2
 *
 * Run:
 *   ./compare_random
 *   # produces: old_lbas.bin  new_lbas.bin  (uint64_t arrays)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>

/* --- Drive / workload parameters --- */
constexpr uint64_t kDriveSize     = 8ULL * 1000000000000ULL;  /* 8 TB */
constexpr uint64_t kLbaSize       = 512ULL;
constexpr uint64_t kLbaMax        = kDriveSize / kLbaSize;    /* 15,625,000,000 */
constexpr uint32_t kNBlocks       = 8U;                       /* 4 KB I/O */
constexpr int      kTotalThreads  = 2048;
constexpr int      kIosPerThread  = 500;
constexpr int      kTotalIos      = kTotalThreads * kIosPerThread;

inline void chk(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess) {
		std::cerr << file << ":" << line << "  "
			  << cudaGetErrorString(err) << std::endl;
		std::exit(1);
	}
}
#define CHK(call) chk((call), __FILE__, __LINE__)

/* ------------------------------------------------------------------ */
/*  Old logic: tid-derived odd stride (verbatim from old gpu_worker)   */
/* ------------------------------------------------------------------ */
__global__ void gen_old(uint64_t *out, uint64_t lba_max,
			uint32_t n_blocks, int ios_per_thread)
{
	uint64_t tid = threadIdx.x + blockIdx.x * static_cast<uint64_t>(blockDim.x);
	if (tid >= kTotalThreads)
		return;

	uint64_t slba     = tid * static_cast<uint64_t>(n_blocks);
	uint64_t lba_step = ((tid * 1315423911ULL) | 1ULL) *
			    static_cast<uint64_t>(n_blocks);

	if (lba_max && slba + n_blocks >= lba_max)
		slba -= (slba / lba_max) * lba_max;
	if (lba_max && lba_step >= lba_max)
		lba_step -= (lba_step / lba_max) * lba_max;
	if (lba_step == 0)
		lba_step = n_blocks ? n_blocks : 1;

	uint64_t *base = out + tid * ios_per_thread;
	for (int i = 0; i < ios_per_thread; i++) {
		base[i] = slba;
		slba += lba_step;
		if (lba_max && slba >= lba_max)
			slba -= lba_max;
	}
}

/* ------------------------------------------------------------------ */
/*  New logic: curand() uniform random                                 */
/* ------------------------------------------------------------------ */
__global__ void gen_new(uint64_t *out, uint64_t lba_max,
			uint32_t n_blocks, int ios_per_thread)
{
	uint64_t tid = threadIdx.x + blockIdx.x * static_cast<uint64_t>(blockDim.x);
	if (tid >= kTotalThreads)
		return;

	curandState rng;
	curand_init(1234ULL, tid, 0, &rng);

	uint64_t bound = lba_max - static_cast<uint64_t>(n_blocks);
	uint64_t *base = out + tid * ios_per_thread;

	for (int i = 0; i < ios_per_thread; i++) {
		uint64_t r = (static_cast<uint64_t>(curand(&rng)) << 32) |
			     static_cast<uint64_t>(curand(&rng));
		base[i] = r % bound;
	}
}

/* ------------------------------------------------------------------ */
/*  Main: launch both kernels, copy results, dump to binary files      */
/* ------------------------------------------------------------------ */
static void dump_bin(const char *path, const std::vector<uint64_t> &data)
{
	std::ofstream out(path, std::ios::binary);
	if (!out) {
		std::cerr << "failed to open " << path << std::endl;
		std::exit(1);
	}
	out.write(reinterpret_cast<const char *>(data.data()),
		  data.size() * sizeof(uint64_t));
	std::cout << "  wrote " << path
		  << "  (" << data.size() << " samples)" << std::endl;
}

int main()
{
	std::cout << "lba_max        = " << kLbaMax        << "\n"
		  << "n_blocks       = " << kNBlocks       << "\n"
		  << "total_threads  = " << kTotalThreads  << "\n"
		  << "ios_per_thread = " << kIosPerThread  << "\n"
		  << "total I/Os     = " << kTotalIos      << std::endl;

	constexpr size_t nbytes = static_cast<size_t>(kTotalIos) * sizeof(uint64_t);

	uint64_t *d_old, *d_new;
	CHK(cudaMalloc(&d_old, nbytes));
	CHK(cudaMalloc(&d_new, nbytes));

	constexpr int tpb  = 256;
	constexpr int blks = (kTotalThreads + tpb - 1) / tpb;

	std::cout << "\nRunning old (stride) kernel..." << std::endl;
	gen_old<<<blks, tpb>>>(d_old, kLbaMax, kNBlocks, kIosPerThread);
	CHK(cudaDeviceSynchronize());

	std::cout << "Running new (curand) kernel..." << std::endl;
	gen_new<<<blks, tpb>>>(d_new, kLbaMax, kNBlocks, kIosPerThread);
	CHK(cudaDeviceSynchronize());

	std::vector<uint64_t> h_old(kTotalIos);
	std::vector<uint64_t> h_new(kTotalIos);
	CHK(cudaMemcpy(h_old.data(), d_old, nbytes, cudaMemcpyDeviceToHost));
	CHK(cudaMemcpy(h_new.data(), d_new, nbytes, cudaMemcpyDeviceToHost));

	std::cout << "\nDumping results:" << std::endl;
	dump_bin("old_lbas.bin", h_old);
	dump_bin("new_lbas.bin", h_new);

	cudaFree(d_old);
	cudaFree(d_new);

	std::cout << "\nDone.  Run compare_random.py to visualize." << std::endl;
	return 0;
}
