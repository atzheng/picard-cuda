#pragma once
#include <cuda_runtime.h>
#include <cmath>

// Small MLP forward pass, executed per-thread.
// Params layout in flat buffer: for each layer i (0..num_layers-1):
//   weights: layer_sizes[i+1] * layer_sizes[i] floats (row-major: out x in)
//   biases:  layer_sizes[i+1] floats
// Total: sum over i of layer_sizes[i+1] * (layer_sizes[i] + 1)

// Scratch space needed: max(layer_sizes) floats for input + max(layer_sizes) for output
// With typical sizes [30, 64, 64, 31], max is 64 => 128 floats scratch.

// Get offset into flat param buffer for layer i
__device__ inline int get_layer_offset(const int* layer_sizes, int num_layers, int layer) {
    int offset = 0;
    for (int i = 0; i < layer; i++) {
        offset += layer_sizes[i + 1] * layer_sizes[i];  // weights
        offset += layer_sizes[i + 1];                     // biases
    }
    return offset;
}

// Compute linear layer: out = W @ x + b
// W is [nout x nin] row-major, b is [nout], x is [nin], out is [nout]
__device__ inline void linear_forward(
    const float* W, const float* b,
    const float* x, float* out,
    int nin, int nout
) {
    for (int i = 0; i < nout; i++) {
        float sum = b[i];
        for (int j = 0; j < nin; j++) {
            sum += W[i * nin + j] * x[j];
        }
        out[i] = sum;
    }
}

// ReLU in-place
__device__ inline void relu_inplace(float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = fmaxf(0.0f, x[i]);
    }
}

// LogSumExp normalization: x[i] -= log(sum(exp(x[j])))
__device__ inline void logsumexp_normalize(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) max_val = fmaxf(max_val, x[i]);

    float sum_exp = 0.0f;
    for (int i = 0; i < n; i++) sum_exp += expf(x[i] - max_val);

    float lse = max_val + logf(sum_exp);
    for (int i = 0; i < n; i++) x[i] -= lse;
}

// Full MLP forward pass: predict(params, x)
// input: x of length layer_sizes[0]
// output: written to 'output' of length layer_sizes[num_layers]
// scratch: two buffers of length max(layer_sizes), caller-provided
__device__ inline void nn_predict(
    const float* params,
    const int* layer_sizes,
    int num_layers,
    const float* input,
    float* output,
    float* scratch_a,
    float* scratch_b
) {
    // Copy input to scratch_a
    int in_size = layer_sizes[0];
    for (int i = 0; i < in_size; i++) scratch_a[i] = input[i];

    int param_offset = 0;
    for (int layer = 0; layer < num_layers; layer++) {
        int nin = layer_sizes[layer];
        int nout = layer_sizes[layer + 1];
        const float* W = params + param_offset;
        const float* b = W + nout * nin;
        param_offset += nout * nin + nout;

        float* dst = (layer == num_layers - 1) ? output : scratch_b;
        linear_forward(W, b, scratch_a, dst, nin, nout);

        if (layer < num_layers - 1) {
            // Hidden layer: apply ReLU
            relu_inplace(dst, nout);
            // Swap buffers
            float* tmp = scratch_a;
            scratch_a = dst;
            scratch_b = tmp;
        }
    }

    // Apply logsumexp normalization to output (log-softmax)
    int out_size = layer_sizes[num_layers];
    logsumexp_normalize(output, out_size);
}
