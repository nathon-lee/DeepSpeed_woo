// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "custom_cuda_layers.h"

#define TRANSPOSE_TILE_DIM 32
#define TRANSPOSE_BLOCK_ROWS 8
#define TRANSPOSE_VEC4_THREADS_X 8
#define TRANSPOSE_VEC2_THREADS_X 16

#define TRANS0213_TILE_VEC 32
#define TRANS0213_TILE_HEAD 8

template <typename T>
__global__ void Transpose_Kernel(const T* inp, T* out, int row_width, int col_width)
{
    __shared__ T tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

    int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS) {
        int yj = y + j;
        if (x < row_width && yj < col_width) {
            tile[threadIdx.y + j][threadIdx.x] = inp[yj * row_width + x];
        }
    }

    __syncthreads();

    int out_x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x;
    int out_y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS) {
        int oyj = out_y + j;
        if (out_x < col_width && oyj < row_width) {
            out[oyj * col_width + out_x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void Transpose_Kernel_Vec4(const float* inp, float* out, int row_width, int col_width)
{
    __shared__ float tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

    int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x * 4;
    int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS) {
        int yj = y + j;
        if (yj < col_width) {
            if (x + 3 < row_width) {
                float4 v = *reinterpret_cast<const float4*>(inp + yj * row_width + x);
                tile[threadIdx.y + j][threadIdx.x * 4 + 0] = v.x;
                tile[threadIdx.y + j][threadIdx.x * 4 + 1] = v.y;
                tile[threadIdx.y + j][threadIdx.x * 4 + 2] = v.z;
                tile[threadIdx.y + j][threadIdx.x * 4 + 3] = v.w;
            } else {
                for (int k = 0; k < 4; k++) {
                    if (x + k < row_width) {
                        tile[threadIdx.y + j][threadIdx.x * 4 + k] = inp[yj * row_width + x + k];
                    }
                }
            }
        }
    }

    __syncthreads();

    int out_x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x * 4;
    int out_y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS) {
        int oyj = out_y + j;
        if (oyj < row_width) {
            if (out_x + 3 < col_width) {
                float4 v;
                v.x = tile[threadIdx.x * 4 + 0][threadIdx.y + j];
                v.y = tile[threadIdx.x * 4 + 1][threadIdx.y + j];
                v.z = tile[threadIdx.x * 4 + 2][threadIdx.y + j];
                v.w = tile[threadIdx.x * 4 + 3][threadIdx.y + j];
                *reinterpret_cast<float4*>(out + oyj * col_width + out_x) = v;
            } else {
                for (int k = 0; k < 4; k++) {
                    if (out_x + k < col_width) {
                        out[oyj * col_width + out_x + k] =
                            tile[threadIdx.x * 4 + k][threadIdx.y + j];
                    }
                }
            }
        }
    }
}

__global__ void Transpose_Kernel_Half2(const __half* inp, __half* out, int row_width, int col_width)
{
#ifdef HALF_PRECISION_AVAILABLE
    __shared__ __half tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];

    int x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x * 2;
    int y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS) {
        int yj = y + j;
        if (yj < col_width) {
            if (x + 1 < row_width) {
                __half2 v = *reinterpret_cast<const __half2*>(inp + yj * row_width + x);
                tile[threadIdx.y + j][threadIdx.x * 2 + 0] = __low2half(v);
                tile[threadIdx.y + j][threadIdx.x * 2 + 1] = __high2half(v);
            } else if (x < row_width) {
                tile[threadIdx.y + j][threadIdx.x * 2] = inp[yj * row_width + x];
            }
        }
    }

    __syncthreads();

    int out_x = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.x * 2;
    int out_y = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TRANSPOSE_TILE_DIM; j += TRANSPOSE_BLOCK_ROWS) {
        int oyj = out_y + j;
        if (oyj < row_width) {
            if (out_x + 1 < col_width) {
                __half2 v = __halves2half2(tile[threadIdx.x * 2 + 0][threadIdx.y + j],
                                           tile[threadIdx.x * 2 + 1][threadIdx.y + j]);
                *reinterpret_cast<__half2*>(out + oyj * col_width + out_x) = v;
            } else if (out_x < col_width) {
                out[oyj * col_width + out_x] = tile[threadIdx.x * 2][threadIdx.y + j];
            }
        }
    }
#endif
}

template <>
void Transpose<__half>(const __half* inp_mat,
                       __half* out_mat,
                       int rows,
                       int cols,
                       cudaStream_t stream)
{
    dim3 grid_dim((cols + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM,
                  (rows + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM);

    if ((cols % 2 == 0) && ((reinterpret_cast<uintptr_t>(inp_mat) % sizeof(__half2)) == 0) &&
        ((reinterpret_cast<uintptr_t>(out_mat) % sizeof(__half2)) == 0)) {
        dim3 block_dim(TRANSPOSE_VEC2_THREADS_X, TRANSPOSE_BLOCK_ROWS);
        Transpose_Kernel_Half2<<<grid_dim, block_dim, 0, stream>>>(inp_mat, out_mat, cols, rows);
    } else {
        dim3 block_dim(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
        Transpose_Kernel<__half><<<grid_dim, block_dim, 0, stream>>>(inp_mat, out_mat, cols, rows);
    }
}

template <>
void Transpose<float>(const float* inp_mat, float* out_mat, int rows, int cols, cudaStream_t stream)
{
    dim3 grid_dim((cols + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM,
                  (rows + TRANSPOSE_TILE_DIM - 1) / TRANSPOSE_TILE_DIM);

    if ((cols % 4 == 0) && ((reinterpret_cast<uintptr_t>(inp_mat) % sizeof(float4)) == 0) &&
        ((reinterpret_cast<uintptr_t>(out_mat) % sizeof(float4)) == 0)) {
        dim3 block_dim(TRANSPOSE_VEC4_THREADS_X, TRANSPOSE_BLOCK_ROWS);
        Transpose_Kernel_Vec4<<<grid_dim, block_dim, 0, stream>>>(inp_mat, out_mat, cols, rows);
    } else {
        dim3 block_dim(TRANSPOSE_TILE_DIM, TRANSPOSE_BLOCK_ROWS);
        Transpose_Kernel<float><<<grid_dim, block_dim, 0, stream>>>(inp_mat, out_mat, cols, rows);
    }
}

template <typename T>
__global__ void transform_0213(T* output,
                               const T* vals,
                               int hidden_dim,
                               int seq_length,
                               int heads,
                               int head_ext);

template <>
__global__ void transform_0213<float>(float* output,
                                      const float* vals,
                                      int hidden_dim,
                                      int seq_length,
                                      int heads,
                                      int head_ext)
{
    (void)head_ext;

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int tiles_per_head = (heads + blockDim.y - 1) / blockDim.y;
    int d0 = blockIdx.x;
    int d1 = blockIdx.y;
    int vec_tile = blockIdx.z / tiles_per_head;
    int head_tile = blockIdx.z % tiles_per_head;

    int d2 = head_tile * blockDim.y + threadIdx.y;
    int d3 = vec_tile * blockDim.x + threadIdx.x;

    if (d2 >= heads || d3 >= d2_stride) return;

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    float4* output_vec = reinterpret_cast<float4*>(output);

    float4 inputs = vals_vec[d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
    output_vec[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = inputs;
}

template <>
__global__ void transform_0213<__half>(__half* output,
                                       const __half* vals,
                                       int hidden_dim,
                                       int seq_length,
                                       int heads,
                                       int head_ext)
{
#ifdef HALF_PRECISION_AVAILABLE
    (void)head_ext;

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int tiles_per_head = (heads + blockDim.y - 1) / blockDim.y;
    int d0 = blockIdx.x;
    int d1 = blockIdx.y;
    int vec_tile = blockIdx.z / tiles_per_head;
    int head_tile = blockIdx.z % tiles_per_head;

    int d2 = head_tile * blockDim.y + threadIdx.y;
    int d3 = vec_tile * blockDim.x + threadIdx.x;

    if (d2 >= heads || d3 >= d2_stride) return;

    float4 vals_arr[1];

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    float4* output_vec = reinterpret_cast<float4*>(output);

    vals_arr[0] = vals_vec[d0 * d0_stride + d1 * d1_stride + d2 * d2_stride + d3];
    output_vec[d0 * d0_out_stride + d1 * d1_out_stride + d2 * d2_out_stride + d3] = vals_arr[0];
#endif
}

template <>
void launch_transform_0213<float>(float* output,
                                  const float* vals,
                                  int batch_size,
                                  int seq_length,
                                  int hidden_dim,
                                  int heads,
                                  cudaStream_t stream)
{
    hidden_dim >>= 2;
    int vec_per_head = hidden_dim / heads;

    dim3 block_dim(min(vec_per_head, TRANS0213_TILE_VEC), min(heads, TRANS0213_TILE_HEAD));
    dim3 grid_dim(batch_size,
                  seq_length,
                  ((vec_per_head + block_dim.x - 1) / block_dim.x) *
                      ((heads + block_dim.y - 1) / block_dim.y));

    transform_0213<float>
        <<<grid_dim, block_dim, 0, stream>>>(output, vals, hidden_dim, seq_length, heads, 1);
}

template <>
void launch_transform_0213<__half>(__half* output,
                                   const __half* vals,
                                   int batch_size,
                                   int seq_length,
                                   int hidden_dim,
                                   int heads,
                                   cudaStream_t stream)
{
    hidden_dim >>= 3;
    int vec_per_head = hidden_dim / heads;

    dim3 block_dim(min(vec_per_head, TRANS0213_TILE_VEC), min(heads, TRANS0213_TILE_HEAD));
    dim3 grid_dim(batch_size,
                  seq_length,
                  ((vec_per_head + block_dim.x - 1) / block_dim.x) *
                      ((heads + block_dim.y - 1) / block_dim.y));

    transform_0213<__half>
        <<<grid_dim, block_dim, 0, stream>>>(output, vals, hidden_dim, seq_length, heads, 1);
}

// Bias add
template <typename T>
__global__ void bias_add_transform_0213(T* output,
                                        const T* vals,
                                        const T* bias,
                                        int hidden_dim,
                                        int seq_length,
                                        int heads,
                                        int head_ext);

template <>
__global__ void bias_add_transform_0213<float>(float* output,
                                               const float* vals,
                                               const float* bias,
                                               int hidden_dim,
                                               int seq_length,
                                               int heads,
                                               int head_ext)
{
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y;                                                  // Sequence ID (0-127)
    int cnt = blockIdx.z / head_ext;                                      // Hidden count
    int d2 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output);

    float4 inputs = vals_vec[d0 * d0_stride * (gridDim.z / head_ext) + cnt * d1_stride +
                             d1 * d1_stride * (gridDim.z / head_ext) + d2 * d2_stride + d3];
    float4 biases = bias_vec[cnt * d1_stride + d2 * d2_stride + d3];

    float4 outputs;
    outputs.x = inputs.x + biases.x;
    outputs.y = inputs.y + biases.y;
    outputs.z = inputs.z + biases.z;
    outputs.w = inputs.w + biases.w;

    output_vec[cnt * d0_out_stride * gridDim.x + d0 * d0_out_stride + d1 * d1_out_stride +
               d2 * d2_out_stride + d3] = outputs;
}

#define ATTN_H 3
#define MAX_SEQ_LINE 10

template <>
__global__ void bias_add_transform_0213<__half>(__half* output,
                                                const __half* vals,
                                                const __half* bias,
                                                int hidden_dim,
                                                int seq_length,
                                                int heads,
                                                int head_ext)
{
#ifdef HALF_PRECISION_AVAILABLE

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y;                                                  // Sequence ID (0-127)
    int cnt = blockIdx.z / head_ext;                                      // Hidden count
    int d2 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    float4 vals_arr;
    float4 bias_arr;
    float4 output_arr;
    __half2* vals_half = reinterpret_cast<__half2*>(&vals_arr);
    __half2* bias_half = reinterpret_cast<__half2*>(&bias_arr);
    __half2* output_half = reinterpret_cast<__half2*>(&output_arr);

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output);

    vals_vec += (d0 * d0_stride * (gridDim.z / head_ext));
    vals_vec += (d1 * d1_stride * (gridDim.z / head_ext));
    vals_vec += (cnt * d1_stride);
    vals_vec += (d2 * d2_stride);

    bias_vec += (cnt * d1_stride);
    bias_vec += (d2 * d2_stride);

    output_vec += (cnt * d0_stride * gridDim.x);
    output_vec += (d1 * d2_stride);
    output_vec += (d0 * d0_stride);
    output_vec += (d2 * d2_out_stride);

    bias_arr = bias_vec[d3];
    vals_arr = vals_vec[d3];

#if defined(__ACC_HALF__)
    output_half[0] = vals_half[0] + bias_half[0];
    output_half[1] = vals_half[1] + bias_half[1];
    output_half[2] = vals_half[2] + bias_half[2];
    output_half[3] = vals_half[3] + bias_half[3];
#else
    float2 bias_arr_f[4];
    float2 vals_arr_f[4];
#pragma unroll
    for (int l = 0; l < 4; l++) {
        bias_arr_f[l] = __half22float2(bias_half[l]);
        vals_arr_f[l] = __half22float2(vals_half[l]);
        vals_arr_f[l].x += bias_arr_f[l].x;
        vals_arr_f[l].y += bias_arr_f[l].y;
        output_half[l] = __float22half2_rn(vals_arr_f[l]);
    }
#endif
    output_vec[d3] = output_arr;

#endif
}

__global__ void bias_add_transform_0213_v2(__half* output,
                                           const __half* vals,
                                           const __half* bias,
                                           int hidden_dim,
                                           int seq_length,
                                           int heads)
{
#ifdef HALF_PRECISION_AVAILABLE
    __shared__ float4 in_data[3072];

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;
    int iteration_stride = d1_stride * blockDim.z;  // Hidden * 3 / 8
    int batch_stride = d0_stride * blockDim.z;      // Hidden * S * 3 / 8

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;    // Batch
    int d1 = blockIdx.y;    // Sequence ID (0-127)
    int cnt = threadIdx.z;  // blockIdx.z; // Hidden count
    int d2 = threadIdx.y;   // Head (0-11)
    int d3 = threadIdx.x;   // Values (groups of 4)

    float4 vals_arr[1];
    float4 bias_arr[1];
    float4 output_arr[1];
    __half2* vals_half = reinterpret_cast<__half2*>(vals_arr);
    __half2* bias_half = reinterpret_cast<__half2*>(bias_arr);
    __half2* output_half = reinterpret_cast<__half2*>(output_arr);

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output);

    int iter_index = cnt * d1_stride + d2 * d2_stride + d3;
    int input_offset = d0 * batch_stride + d1 * (iteration_stride << 1);
    bias_arr[0] = bias_vec[iter_index];

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_id = iter * iteration_stride + iter_index;
        vals_arr[0] = vals_vec[input_offset + iter_id];

        output_half[0] = vals_half[0] + bias_half[0];
        output_half[1] = vals_half[1] + bias_half[1];
        output_half[2] = vals_half[2] + bias_half[2];
        output_half[3] = vals_half[3] + bias_half[3];

        in_data[iter_id] = output_arr[0];
    }
    __syncthreads();

    iteration_stride = blockDim.z * (blockDim.y >> 1);
    int matrix_stride = (d0_out_stride * gridDim.x);
    int head_count = (d2 >> 1) + cnt * (blockDim.y >> 1);

    int out_index = d0 * d0_out_stride + d1 * (d1_out_stride << 1) + d3 + (d2 % 2) * d2_stride;

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_row = (iter * iteration_stride) + head_count;
        int iter_offset =
            (iter_row % blockDim.y) * d2_out_stride + (iter_row / blockDim.y) * matrix_stride;
        output_vec[out_index + iter_offset] =
            in_data[iter_row * d2_stride + d3 + (d2 % 2) * (d1_stride * blockDim.z)];
    }
#endif
}

// [B S C*H] - > C * [B A S N]
template <>
void launch_bias_add_transform_0213<float>(float* output,
                                           const float* vals,
                                           const float* bias,
                                           int batch_size,
                                           int seq_length,
                                           int hidden_dim,
                                           int heads,
                                           cudaStream_t stream,
                                           int trans_count)
{
    hidden_dim >>= 2;
    int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;

    dim3 block_dim(hidden_dim / heads, (heads / head_ext));
    dim3 grid_dim(batch_size, seq_length, (trans_count * head_ext));

    bias_add_transform_0213<float><<<grid_dim, block_dim, 0, stream>>>(
        output, vals, bias, hidden_dim, seq_length, heads, head_ext);
}

template <>
void launch_bias_add_transform_0213<__half>(__half* output,
                                            const __half* vals,
                                            const __half* bias,
                                            int batch_size,
                                            int seq_length,
                                            int hidden_dim,
                                            int heads,
                                            cudaStream_t stream,
                                            int trans_count)
{
    hidden_dim >>= 3;
    if (hidden_dim > 128 || hidden_dim < 16) {
        int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;
        dim3 block_dim(hidden_dim / heads, (heads / head_ext));
        dim3 grid_dim(batch_size, seq_length, (trans_count * head_ext));
        bias_add_transform_0213<__half><<<grid_dim, block_dim, 0, stream>>>(
            output, vals, bias, hidden_dim, seq_length, heads, head_ext);
    } else {
        dim3 block_dim(hidden_dim / heads, heads, trans_count);
        dim3 grid_dim(batch_size, seq_length / 2);
        bias_add_transform_0213_v2<<<grid_dim, block_dim, 0, stream>>>(
            output, vals, bias, hidden_dim, seq_length, heads);
    }
}

template <typename T>
__global__ void transform4d_0213(T* out,
                                 const T* in,
                                 int heads,
                                 int seq_length,
                                 int hidden_dim,
                                 int head_ext);

template <>
__global__ void transform4d_0213<float>(float* out,
                                        const float* in,
                                        int heads,
                                        int seq_length,
                                        int hidden_dim,
                                        int head_ext)
{
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = d0_stride / heads;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = hidden_dim;

    int d0 = blockIdx.x;                                        // Batch
    int d1 = blockIdx.y / ((seq_length - 1) / blockDim.y + 1);  // Head
    int d2 = (threadIdx.y + blockDim.y * blockIdx.y) % seq_length;
    int cnt = blockIdx.z;
    int d3 = threadIdx.x;  // Values (groups of 8)

    if (d2 < seq_length) {
        const float4* in_vec = reinterpret_cast<const float4*>(in);
        float4* out_vec = reinterpret_cast<float4*>(out);

        float4 vals_vec = in_vec[cnt * d0_stride * gridDim.x + d0 * d0_stride + d1 * d1_stride +
                                 d2 * d2_stride + d3];
        out_vec[d0 * d0_out_stride * gridDim.z + cnt * d2_out_stride + d1 * d1_out_stride +
                d2 * d2_out_stride * gridDim.z + d3] = vals_vec;
    }
}

template <>
__global__ void transform4d_0213<__half>(__half* out,
                                         const __half* in,
                                         int heads,
                                         int seq_length,
                                         int hidden_dim,
                                         int head_ext)
{
#ifdef HALF_PRECISION_AVAILABLE

    int d0_stride = hidden_dim * (seq_length / head_ext);
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head
    int d2 = blockIdx.z / head_ext;                                       // Sequence
    int cnt = blockIdx.y;                                                 // Hidden count
    int d3 = threadIdx.x;                                                 // Values (groups of 8)

    const float4* in_vec = reinterpret_cast<const float4*>(in);
    float4* out_vec = reinterpret_cast<float4*>(out);

    in_vec += (cnt * d0_stride * gridDim.x);
    in_vec += (d0 * d0_stride);
    in_vec += (d2 * d2_stride);
    in_vec += (d1 * d2_stride * seq_length);

    out_vec += (cnt * d1_stride);
    out_vec += (d1 * d2_stride);
    out_vec += (d0 * d0_stride * gridDim.y);
    out_vec += (d2 * d1_stride * gridDim.y);

    out_vec[d3] = in_vec[d3];

#endif
}

__global__ void transform4d_0213_v2(__half* out,
                                    const __half* in,
                                    int heads,
                                    int seq_length,
                                    int hidden_dim)
{
#ifdef HALF_PRECISION_AVAILABLE
    __shared__ float4 in_data[3072];

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = blockIdx.x;    // Batch
    int d1 = threadIdx.y;   // Head
    int d2 = blockIdx.y;    // Sequence
    int cnt = threadIdx.z;  // Hidden count
    int d3 = threadIdx.x;   // Values (groups of 8)

    const float4* in_vec = reinterpret_cast<const float4*>(in);
    float4* out_vec = reinterpret_cast<float4*>(out);

    int input_offset = d0 * d0_stride + d2 * (d2_stride << 1) + d3 + (d1 % 2) * d2_stride;
    int head_count = (d1 >> 1) + cnt * (blockDim.y >> 1);
    int iteration_stride = blockDim.z * (blockDim.y >> 1);
    int matrix_stride = (d0_stride * gridDim.x);

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_row = iter * iteration_stride + head_count;
        int iter_offset = (iter_row % blockDim.y) * d2_stride;

        in_data[d3 + iter_offset + (iter_row / blockDim.y + (d1 % 2) * blockDim.z) * d1_stride] =
            in_vec[input_offset + iter_offset * seq_length +
                   (iter_row / blockDim.y) * matrix_stride];
    }
    __syncthreads();

    iteration_stride = d1_stride * blockDim.z;
    int iter_index = cnt * d1_stride + d1 * d2_stride + d3;
    int output_offset = d0 * d0_stride * blockDim.z + d2 * (iteration_stride << 1);

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_id = iter * iteration_stride + iter_index;
        out_vec[output_offset + iter_id] = in_data[iter_id];
    }
#endif
}

// 3 * [B A S N] - > [B S C*H]
template <>
void launch_transform4d_0213<float>(float* out,
                                    const float* in,
                                    int batch_size,
                                    int heads,
                                    int seq_length,
                                    int hidden_dim,
                                    cudaStream_t stream,
                                    int trans_count)
{
    hidden_dim >>= 2;
    dim3 grid_dims(batch_size, heads * ((seq_length - 1) / 8 + 1), trans_count);
    dim3 block_dims(hidden_dim / heads, 8);
    transform4d_0213<float>
        <<<grid_dims, block_dims, 0, stream>>>(out, in, heads, seq_length, hidden_dim, 1);
}

template <>
void launch_transform4d_0213<__half>(__half* out,
                                     const __half* in,
                                     int batch_size,
                                     int heads,
                                     int seq_length,
                                     int hidden_dim,
                                     cudaStream_t stream,
                                     int trans_count)
{
    hidden_dim >>= 3;
    if (hidden_dim > 128 || hidden_dim < 16) {
        int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;
        dim3 grid_dims(batch_size, trans_count, (seq_length * head_ext));
        dim3 block_dims(hidden_dim / heads, (heads / head_ext));
        transform4d_0213<__half><<<grid_dims, block_dims, 0, stream>>>(
            out, in, heads, seq_length, hidden_dim, head_ext);
    } else {
        dim3 grid_dims(batch_size, seq_length / 2);
        dim3 block_dims(hidden_dim / heads, heads, trans_count);
        transform4d_0213_v2<<<grid_dims, block_dims, 0, stream>>>(
            out, in, heads, seq_length, hidden_dim);
    }
}
