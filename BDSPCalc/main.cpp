#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <memory>
#include <atomic>
#include <immintrin.h>
#include <windows.h>

// Defines & macros
#define BILLION				(1024ULL * 1024ULL * 1024ULL)
#define _mm256_const_vec(x) _mm256_set_epi32(x, x, x, x, x, x, x, x)
#define _mm256_not_si256(x) _mm256_xor_si256(x, _mm256_const_vec(0xFFFFFFFFUL))

// Compile-time configuration
#define MAKE_SHINY
#define PERFECT_IVS			6UL
#define THREAD_COUNT		4UL
#define STATE_SEED			0xDEADBEEFUL
#define ITERATIONS			UINT64_MAX

CRITICAL_SECTION stdout_section;
__declspec(thread) uint32_t tid = 0;
std::atomic<bool> stop_search = false;

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

struct thread_data_t
{
	uint32_t tid;
	uint32_t *ivs;
};

uint32_t xorshift128_unity(uint32_t *state)
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

__m256i xorshift128_unity_avx2(__m256i *state)
{
	__m256i t1 = state[0];
	__m256i t2 = state[3];
	t1 = _mm256_xor_si256(t1, _mm256_slli_epi32(t1, 11));
	t1 = _mm256_xor_si256(t1, _mm256_srli_epi32(t1, 8));
	__m256i v = _mm256_xor_si256(_mm256_xor_si256(t1, t2), _mm256_srli_epi32(t2, 19));
	state[0] = state[1];
	state[1] = state[2];
	state[2] = t2;
	state[3] = v;
	return _mm256_add_epi32(_mm256_const_vec(0x80000000UL), _mm256_and_si256(v, _mm256_not_si256(_mm256_cmpeq_epi32(v, _mm256_const_vec(0xFFFFFFFFUL)))));
}

int prints(const char *format, ...)
{
	EnterCriticalSection(&stdout_section);
	va_list args;
	va_start(args, format);
	int result = vprintf(format, args);
	va_end(args);
	LeaveCriticalSection(&stdout_section);
	return result;
}

int printsi(const char *format, ...)
{
	EnterCriticalSection(&stdout_section);
	va_list args;
	va_start(args, format);
	printf("[%u] ", tid);
	int result = vprintf(format, args);
	va_end(args);
	LeaveCriticalSection(&stdout_section);
	return result;
}

bool find_pokemon(uint32_t *iv)
{
	uint32_t prev_state[4], state[4];
	memcpy(prev_state, iv, sizeof(prev_state));
	memcpy(state, iv, sizeof(state));
	printsi("Initial state { %08X, %08X, %08X, %08X }\n\n", state[0], state[1], state[2], state[3]);
	uint32_t outputs[oi_count];
	for (int i = 0; i < _countof(outputs); ++i)
		outputs[i] = xorshift128_unity(state);
	uint64_t index_base = 0;
	for (uint64_t i = 0; i < ITERATIONS && !stop_search.load(std::memory_order::memory_order_relaxed); ++i)
	{
		if (i % BILLION == 0)
			printsi("%llu) Completed %llu iterations\n", i / BILLION, i);

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
				printsi("%llu) Found pokemon with state { %08X, %08X, %08X, %08X }\n", i, prev_state[0], prev_state[1], prev_state[2], prev_state[3]);
				return true;
			}
		}
		index_base++;
		xorshift128_unity(prev_state);
		outputs[(index_base + oi_last_entry) % oi_count] = xorshift128_unity(state);
	}

	return false;
}

bool find_pokemon(uint32_t seed)
{
	std::mt19937 mt(seed);
	uint32_t state[] = { mt(), mt(), mt(), mt() };
	return find_pokemon(state);
}

DWORD WINAPI th_find_pokemon(LPVOID lpParameter)
{
	thread_data_t *data = (thread_data_t *)lpParameter;
	tid = data->tid;
	find_pokemon(data->ivs);
	stop_search.store(true, std::memory_order::memory_order_relaxed);
	return 0;
}

bool find_pokemon_avx2(uint32_t *iv)
{
	__m256i prev_state[4];
	__m256i state[4] =
	{
		_mm256_set_epi32(iv[28], iv[24], iv[20], iv[16], iv[12], iv[8],  iv[4], iv[0]),
		_mm256_set_epi32(iv[29], iv[25], iv[21], iv[17], iv[13], iv[9],  iv[5], iv[1]),
		_mm256_set_epi32(iv[30], iv[26], iv[22], iv[18], iv[14], iv[10], iv[6], iv[2]),
		_mm256_set_epi32(iv[31], iv[27], iv[23], iv[19], iv[15], iv[11], iv[7], iv[3]),
	};
	memcpy(prev_state, state, sizeof(prev_state));
	printsi("Initial states:\n");
	for (int i = 0; i < 8; ++i)
	{
		printsi("%d) { %08X, %08X, %08X, %08X }\n",
			i,
			((uint32_t *)&state[0])[i],
			((uint32_t *)&state[1])[i],
			((uint32_t *)&state[2])[i],
			((uint32_t *)&state[3])[i]
		);
	}
	printsi("\n");
	__m256i outputs[oi_count];
	for (int i = 0; i < _countof(outputs); ++i)
		outputs[i] = xorshift128_unity_avx2(state);
	uint64_t index_base = 0;
	__m256i IV_mask = _mm256_const_vec(31UL);
	#ifdef MAKE_SHINY
	__m256i shiny_mask = _mm256_const_vec(0xFFF0UL);
	__m256i shiny_threshold = _mm256_const_vec(15UL);
	#endif
	for (uint64_t i = 0; i < ITERATIONS && !stop_search.load(std::memory_order::memory_order_relaxed); ++i)
	{
		if (i % BILLION == 0)
			printsi("%llu) Completed %llu iterations\n", i / BILLION, i);

		__m256i IV_HP = _mm256_and_si256(outputs[(index_base + oi_IV_HP) % oi_count], IV_mask);
		__m256i IV_Atk = _mm256_and_si256(outputs[(index_base + oi_IV_Atk) % oi_count], IV_mask);
		__m256i IV_Def = _mm256_and_si256(outputs[(index_base + oi_IV_Def) % oi_count], IV_mask);
		__m256i IV_SpAtk = _mm256_and_si256(outputs[(index_base + oi_IV_SpAtk) % oi_count], IV_mask);
		__m256i IV_SpDef = _mm256_and_si256(outputs[(index_base + oi_IV_SpDef) % oi_count], IV_mask);
		__m256i IV_Agi = _mm256_and_si256(outputs[(index_base + oi_IV_Agi) % oi_count], IV_mask);
		__m256i IVs = _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(IV_HP, IV_Atk), IV_Def), IV_SpAtk), IV_SpDef), IV_Agi);
		__m256i iv_cmp = _mm256_cmpeq_epi32(IVs, _mm256_const_vec(31UL * PERFECT_IVS));
		if (!_mm256_testz_si256(iv_cmp, iv_cmp))
		{
			#ifdef MAKE_SHINY
			__m256i random_TID = outputs[(index_base + oi_random_TID) % oi_count];
			__m256i random_PID = outputs[(index_base + oi_random_PID) % oi_count];
			__m256i TID_shifted = _mm256_srli_epi32(random_TID, 16);
			__m256i TID_masked = _mm256_and_si256(random_TID, shiny_mask);
			__m256i PID_shifed = _mm256_srli_epi32(random_PID, 16);
			__m256i PID_masked = _mm256_and_si256(random_PID, shiny_mask);
			__m256i shiny_value = _mm256_xor_si256(_mm256_xor_si256(_mm256_xor_si256(TID_shifted, TID_masked), PID_shifed), PID_masked);
			__m256i shiny_cmp = _mm256_not_si256(_mm256_cmpgt_epi32(shiny_value, shiny_threshold));
			__m256i final_cmp = _mm256_and_si256(iv_cmp, shiny_cmp);
			if (!_mm256_testz_si256(final_cmp, final_cmp))
			#endif
			{
				for (int j = 0; j < 8; ++j)
				{
					if (!((uint32_t *)&iv_cmp)[j])
						continue;
					#ifdef MAKE_SHINY
					if (!((uint32_t *)&shiny_cmp)[j])
						continue;
					#endif
					printsi("%d-%llu) Found pokemon with state { %08X, %08X, %08X, %08X }\n",
						j, i,
						((uint32_t *)&prev_state[0])[j],
						((uint32_t *)&prev_state[1])[j],
						((uint32_t *)&prev_state[2])[j],
						((uint32_t *)&prev_state[3])[j]
					);
				}
				return true;
			}
		}
		index_base++;
		xorshift128_unity_avx2(prev_state);
		outputs[(index_base + oi_last_entry) % oi_count] = xorshift128_unity_avx2(state);
	}

	return false;
}

bool find_pokemon_avx2(uint32_t seed)
{
	std::mt19937 mt(seed);
	uint32_t iv[32];
	for (size_t i = 0; i < _countof(iv); ++i)
		iv[i] = mt();
	return find_pokemon_avx2(iv);
}

DWORD WINAPI th_find_pokemon_avx2(LPVOID lpParameter)
{
	thread_data_t *data = (thread_data_t *)lpParameter;
	tid = data->tid;
	find_pokemon_avx2(data->ivs);
	stop_search.store(true, std::memory_order::memory_order_relaxed);
	return 0;
}

bool find_pokemon_threaded(uint32_t seed, uint32_t thread_count, LPTHREAD_START_ROUTINE thread_proc, uint32_t state_size)
{
	uint32_t iv_count = state_size * thread_count;
	auto ivs = std::make_unique<uint32_t[]>(iv_count);
	std::mt19937 mt(seed);
	for (uint32_t i = 0; i < iv_count; ++i)
		ivs[i] = mt();

	auto handles = std::make_unique<HANDLE[]>(thread_count);
	auto thread_data = std::make_unique<thread_data_t[]>(thread_count);
	stop_search.store(false, std::memory_order::memory_order_relaxed);
	for (uint32_t i = 0; i < thread_count; ++i)
	{
		auto &data = thread_data[i];
		data.tid = i + 1;
		data.ivs = ivs.get() + i * state_size;
		handles[i] = CreateThread(NULL, 0, thread_proc, &data, 0, NULL);
		if (!handles[i])
		{
			printsi("Failed to create thread #%u!\n", i);
			stop_search.store(true, std::memory_order::memory_order_relaxed);
			if (i > 0)
				WaitForMultipleObjects(i, handles.get(), TRUE, INFINITE);
			return false;
		}
	}

	DWORD result = WaitForMultipleObjects(thread_count, handles.get(), TRUE, INFINITE);
	bool success = result >= WAIT_OBJECT_0 && result < WAIT_OBJECT_0 + thread_count;
	if (success)
		printsi("Search finished\n");
	else
		printsi("WaitForMultipleObjects failed with error code %08X\n", result);

	return success;
}

int main(int argc, char *argv[])
{
	InitializeCriticalSection(&stdout_section);
	LARGE_INTEGER frequency, start_time, end_time, elapsed_microseconds;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&start_time);

	//find_pokemon(STATE_SEED);
	//find_pokemon_avx2(STATE_SEED);
	//find_pokemon_threaded(STATE_SEED, THREAD_COUNT, th_find_pokemon, 4);
	find_pokemon_threaded(STATE_SEED, THREAD_COUNT, th_find_pokemon_avx2, 32);

	QueryPerformanceCounter(&end_time);
	elapsed_microseconds.QuadPart = end_time.QuadPart - start_time.QuadPart;
	elapsed_microseconds.QuadPart *= 1000000;
	elapsed_microseconds.QuadPart /= frequency.QuadPart;
	printsi("Took %llu.%llums\n", elapsed_microseconds.QuadPart / 1000, elapsed_microseconds.QuadPart % 1000);
	DeleteCriticalSection(&stdout_section);
	return 0;
}
