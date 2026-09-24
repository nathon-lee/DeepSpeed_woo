// SPDX-License-Identifier: Apache-2.0
// DeepSpeed Team

#include <torch/extension.h>
#include <climits>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "custom_cuda_layers.h"

void run(torch::Tensor input, torch::Tensor output, bool transform)
{
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "contiguous tensors required");
    TORCH_CHECK(input.device() == output.device() && input.scalar_type() == output.scalar_type(),
                "device and dtype must match");
    TORCH_CHECK(input.numel() == output.numel() && input.numel() > 0 && input.numel() < INT_MAX,
                "invalid tensor size");
    TORCH_CHECK(input.dim() == (transform ? 4 : 2), "invalid input rank");
    c10::cuda::CUDAGuard guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (input.scalar_type() == at::kHalf) {
        auto in = reinterpret_cast<const __half*>(input.data_ptr<at::Half>());
        auto out = reinterpret_cast<__half*>(output.data_ptr<at::Half>());
        if (transform) {
            launch_transform_0213<__half>(out, in, input.size(0), input.size(1),
                                         input.size(2) * input.size(3), input.size(2), stream);
        } else {
            Transpose<__half>(in, out, input.size(0), input.size(1), stream);
        }
    } else {
        TORCH_CHECK(input.scalar_type() == at::kFloat, "FP16 or FP32 required");
        if (transform) {
            launch_transform_0213<float>(output.data_ptr<float>(), input.data_ptr<float>(),
                                        input.size(0), input.size(1), input.size(2) * input.size(3),
                                        input.size(2), stream);
        } else {
            Transpose<float>(input.data_ptr<float>(), output.data_ptr<float>(),
                             input.size(0), input.size(1), stream);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
