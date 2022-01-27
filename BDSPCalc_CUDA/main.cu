#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <memory>
#include <windows.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Defines & macros
#define BILLION				(1024ULL * 1024ULL * 1024ULL)

// Compile-time configuration
#define MAKE_SHINY
#define PERFECT_IVS			6UL
#define THREAD_COUNT		2560UL
#define THREADS_PER_BLOCK	256UL
#define STATE_SEED			0xDEADBEEFUL
#define ITERATIONS_KERNEL	(4UL * 1024UL * 1024UL)
#define ITERATIONS			BILLION

enum output_index_t
{
	oi_encryption_constant,
	oi_random_TID,
	oi_random_PID,
	oi_IV_HP,
	oi_IV_Atk,
	oi_IV_Def,
	oi_IV_SpAtk,
	oi_IV_SpDef,
	oi_IV_Agi,
	oi_last_entry = 15,
	oi_count,
};

__device__ __host__ uint32_t xorshift128_unity(uint32_t *state)
{
	uint32_t t1 = state[0];
	uint32_t t2 = state[3];
	t1 ^= t1 << 11;
	t1 ^= t1 >> 8;
	uint32_t v = t1 ^ t2 ^ (t2 >> 19);
	state[0] = state[1];
	state[1] = state[2];
	state[2] = t2;
	state[3] = v;
	return 0x80000000UL + (v % 0xFFFFFFFFUL);
}

__global__ void find_pokemon(uint32_t *ivs, uint32_t *results)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t prev_state[4], state[4];
	memcpy(state, ivs + 4 * index, sizeof(state));
	memcpy(prev_state, state, sizeof(prev_state));
	uint32_t outputs[oi_count];
	for (int i = 0; i < _countof(outputs); ++i)
		outputs[i] = xorshift128_unity(state);
	uint64_t index_base = 0;
	results[index] = ITERATIONS_KERNEL;
	for (uint32_t i = 0; i < ITERATIONS_KERNEL; ++i)
	{
		uint32_t IVs =
			(outputs[(index_base + oi_IV_HP) % oi_count] & 31) +
			(outputs[(index_base + oi_IV_Atk) % oi_count] & 31) +
			(outputs[(index_base + oi_IV_Def) % oi_count] & 31) +
			(outputs[(index_base + oi_IV_SpAtk) % oi_count] & 31) +
			(outputs[(index_base + oi_IV_SpDef) % oi_count] & 31) +
			(outputs[(index_base + oi_IV_Agi) % oi_count] & 31);
		if (IVs == 31UL * PERFECT_IVS)
		{
			#ifdef MAKE_SHINY
			uint32_t random_TID = outputs[(index_base + oi_random_TID) % oi_count];
			uint32_t random_PID = outputs[(index_base + oi_random_PID) % oi_count];
			if (((random_TID >> 16) ^ (random_TID & 0xFFF0) ^ (random_PID >> 16) ^ (random_PID & 0xFFF0)) < 16)
			#endif
			{
				results[index] = i;
			}
		}
		index_base++;
		xorshift128_unity(prev_state);
		outputs[(index_base + oi_last_entry) % oi_count] = xorshift128_unity(state);
	}
	memcpy(ivs + 4 * index, state, sizeof(state));
}

void *allocate_cuda(size_t size)
{
	void *data = NULL;
	cudaError_t err = cudaMalloc(&data, size);
	if (err != cudaSuccess)
	{
		printf("Failed to allocate device memory! Error: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	return data;
}

void memcpy_cuda(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
	cudaError_t err = cudaMemcpy(dst, src, count, kind);
	if (err != cudaSuccess)
	{
		printf("Failed to copy memory! Error: %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char *argv[])
{
	std::mt19937 mt(STATE_SEED);
	auto h_ivs = std::make_unique<uint32_t[]>(4 * THREAD_COUNT);
	auto h_results = std::make_unique<uint32_t[]>(THREAD_COUNT);
	for (uint32_t i = 0; i < 4 * THREAD_COUNT; ++i)
		h_ivs[i] = mt();
	
	uint32_t *d_ivs = (uint32_t *)allocate_cuda(4 * THREAD_COUNT * sizeof(uint32_t));
	uint32_t *d_results = (uint32_t *)allocate_cuda(THREAD_COUNT * sizeof(uint32_t));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	memcpy_cuda(d_ivs, h_ivs.get(), 4 * THREAD_COUNT * sizeof(uint32_t), cudaMemcpyHostToDevice);

	LARGE_INTEGER frequency, start_time, end_time, elapsed_microseconds;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start_time);

	float total_kernel_time = 0;
	for (uint64_t i = 0; i < ITERATIONS / ITERATIONS_KERNEL; ++i)
	{
		cudaEventRecord(start);
		find_pokemon<<<THREAD_COUNT / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_ivs, d_results);
		cudaEventRecord(stop);
		memcpy_cuda(h_results.get(), d_results, THREAD_COUNT * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		for (uint32_t j = 0; j < THREAD_COUNT; ++j)
		{
			if (h_results[j] != ITERATIONS_KERNEL)
			{
				uint32_t state[4];
				memcpy(state, h_ivs.get() + 4 * j, sizeof(state));
				for (uint32_t k = 0; k < h_results[j]; ++k)
					xorshift128_unity(state);
				printf("%llu-%u) Found pokemon with state { %08X, %08X, %08X, %08X }\n", i, j, state[0], state[1], state[2], state[3]);
			}
		}
		memcpy_cuda(h_ivs.get(), d_ivs, 4 * THREAD_COUNT * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		float kernel_time = 0;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&kernel_time, start, stop);
		total_kernel_time += kernel_time;
	}

	QueryPerformanceCounter(&end_time);
	elapsed_microseconds.QuadPart = end_time.QuadPart - start_time.QuadPart;
	elapsed_microseconds.QuadPart *= 1000000;
	elapsed_microseconds.QuadPart /= frequency.QuadPart;
	printf("CUDA kernels took %fms\n", total_kernel_time);
	printf("Everything took %llu.%llums\n", elapsed_microseconds.QuadPart / 1000, elapsed_microseconds.QuadPart % 1000);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_ivs);
	cudaFree(d_results);
	return 0;
}
