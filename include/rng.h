#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Simple Philox-based PRNG for reproducible random numbers on GPU.
// Not JAX-Threefry compatible, but sufficient for equivalent statistical behavior.

__host__ __device__ inline uint64_t splitmix64(uint64_t key) {
    key += 0x9e3779b97f4a7c15ULL;
    key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ULL;
    key = (key ^ (key >> 27)) * 0x94d049bb133111ebULL;
    return key ^ (key >> 31);
}

// Split a key into two independent subkeys
__host__ __device__ inline void split_key(uint64_t key, uint64_t& key1, uint64_t& key2) {
    key1 = splitmix64(key);
    key2 = splitmix64(key1 + 1);
}

// Generate a uniform float in [0, 1)
__host__ __device__ inline float rand_uniform(uint64_t& key) {
    key = splitmix64(key);
    return (key >> 40) * (1.0f / 16777216.0f);  // 24-bit mantissa
}

// Generate a random integer in [0, n)
__host__ __device__ inline int rand_int(uint64_t& key, int n) {
    float u = rand_uniform(key);
    int r = (int)(u * n);
    return r >= n ? n - 1 : r;
}

// Bernoulli: returns true with probability p
__host__ __device__ inline bool rand_bernoulli(uint64_t& key, float p) {
    return rand_uniform(key) < p;
}

// Fisher-Yates shuffle (host-side for random_assignment)
inline void shuffle(int* arr, int n, uint64_t& key) {
    for (int i = n - 1; i > 0; --i) {
        int j = rand_int(key, i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

// Sample from a categorical distribution with unnormalized weights
__device__ inline int rand_categorical(uint64_t& key, const float* weights, int n) {
    float total = 0.0f;
    for (int i = 0; i < n; i++) total += weights[i];
    float u = rand_uniform(key) * total;
    float cum = 0.0f;
    for (int i = 0; i < n; i++) {
        cum += weights[i];
        if (u < cum) return i;
    }
    return n - 1;
}
