/*
 * compare_random.cu — compare old stride-based vs curand() LBA generation
 *
 * Runs both methods on the GPU and writes raw LBA arrays to binary files.
 * A companion Python script (plot_random.py) reads them and draws histograms.
 *
 * Build:
 *   nvcc -o compare_random compare_random.cu -lcurand -O2
 *
 * Run:
 *   ./compare_random
 *   # produces: old_lbas.bin  new_lbas.bin  (uint64_t arrays)
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>

/* --- Drive / workload parameters --- */
#define DRIVE_SIZE      (8ULL * 1000000000000ULL)  /* 8 TB */
#define LBA_SIZE        512ULL
#define LBA_MAX         (DRIVE_SIZE / LBA_SIZE)    /* 15,625,000,000 */
#define N_BLOCKS        8U                         /* 4 KB I/O */
#define TOTAL_THREADS   2048
#define IOS_PER_THREAD  500
#define TOTAL_IOS       (TOTAL_THREADS * IOS_PER_THREAD)

#define CHK(call) do {                                          \
    cudaError_t e = (call);                                     \
    if (e != cudaSuccess) {                                     \
        fprintf(stderr, "%s:%d  %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(e));                          \
        exit(1);                                                \
    }                                                           \
} while (0)

/* ------------------------------------------------------------------ */
/*  Old logic: tid-derived odd stride (verbatim from old gpu_worker)   */
/* ------------------------------------------------------------------ */
__global__ void gen_old(uint64_t *out, uint64_t lba_max,
                        uint32_t n_blocks, int ios_per_thread)
{
    uint64_t tid = threadIdx.x + blockIdx.x * (uint64_t)blockDim.x;
    if (tid >= TOTAL_THREADS)
        return;

    uint64_t slba     = tid * (uint64_t)n_blocks;
    uint64_t lba_step = ((tid * 1315423911ULL) | 1ULL) * (uint64_t)n_blocks;

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
    uint64_t tid = threadIdx.x + blockIdx.x * (uint64_t)blockDim.x;
    if (tid >= TOTAL_THREADS)
        return;

    curandState rng;
    curand_init(1234ULL, tid, 0, &rng);

    uint64_t bound = lba_max - (uint64_t)n_blocks;
    uint64_t *base = out + tid * ios_per_thread;

    for (int i = 0; i < ios_per_thread; i++) {
        uint64_t r = ((uint64_t)curand(&rng) << 32) |
                     (uint64_t)curand(&rng);
        base[i] = r % bound;
    }
}

/* ------------------------------------------------------------------ */
/*  Main: launch both kernels, copy results, dump to binary files      */
/* ------------------------------------------------------------------ */
static void dump_bin(const char *path, const uint64_t *data, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    fwrite(data, sizeof(uint64_t), n, f);
    fclose(f);
    printf("  wrote %s  (%zu samples)\n", path, n);
}

int main(void)
{
    printf("lba_max        = %llu\n", (unsigned long long)LBA_MAX);
    printf("n_blocks       = %u\n", N_BLOCKS);
    printf("total_threads  = %d\n", TOTAL_THREADS);
    printf("ios_per_thread = %d\n", IOS_PER_THREAD);
    printf("total I/Os     = %d\n", TOTAL_IOS);

    const size_t nbytes = (size_t)TOTAL_IOS * sizeof(uint64_t);

    uint64_t *d_old, *d_new;
    CHK(cudaMalloc(&d_old, nbytes));
    CHK(cudaMalloc(&d_new, nbytes));

    const int tpb = 256;
    const int blks = (TOTAL_THREADS + tpb - 1) / tpb;

    printf("\nRunning old (stride) kernel...\n");
    gen_old<<<blks, tpb>>>(d_old, LBA_MAX, N_BLOCKS, IOS_PER_THREAD);
    CHK(cudaDeviceSynchronize());

    printf("Running new (curand) kernel...\n");
    gen_new<<<blks, tpb>>>(d_new, LBA_MAX, N_BLOCKS, IOS_PER_THREAD);
    CHK(cudaDeviceSynchronize());

    uint64_t *h_old = (uint64_t *)malloc(nbytes);
    uint64_t *h_new = (uint64_t *)malloc(nbytes);
    CHK(cudaMemcpy(h_old, d_old, nbytes, cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(h_new, d_new, nbytes, cudaMemcpyDeviceToHost));

    printf("\nDumping results:\n");
    dump_bin("old_lbas.bin", h_old, TOTAL_IOS);
    dump_bin("new_lbas.bin", h_new, TOTAL_IOS);

    free(h_old);
    free(h_new);
    cudaFree(d_old);
    cudaFree(d_new);

    printf("\nDone.  Run plot_random.py to visualize.\n");
    return 0;
}
