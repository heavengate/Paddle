/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
__global__ void GatherPointKernel(int b, int n, int m,
                                  const T *__restrict__ inp,
                                  const int *__restrict__ idx,
                                  T *__restrict__ out) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int j = blockIdx.y * blockDim.x + threadIdx.x; j < m;
         j += blockDim.x * gridDim.y) {
      int a = idx[i * m + j];
      out[(i * m + j) * 3 + 0] = inp[(i * n + a) * 3 + 0];
      out[(i * m + j) * 3 + 1] = inp[(i * n + a) * 3 + 1];
      out[(i * m + j) * 3 + 2] = inp[(i * n + a) * 3 + 2];
    }
  }
}

template <typename T>
class GatherPointOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto *points = ctx.Input<Tensor>("X");
    auto *index = ctx.Input<Tensor>("Index");
    auto *output = ctx.Output<Tensor>("Output");
    if (points->numel() == 0) return;
    // allocate memory
    output->mutable_data<T>(ctx.GetPlace());

    int batch_size = points->dims()[0];
    int n_points = points->dims()[1];
    int m_points = index->dims()[1];

    // faltten
    auto in_points = framework::EigenVector<T>::Flatten(*points);
    const T *p_points = &(in_points(0));

    auto in_index = framework::EigenVector<int>::Flatten(*index);
    const int *p_index = &(in_index(0));

    auto out_points = framework::EigenVector<T>::Flatten(*output);
    T *p_out_points = &(out_points(0));

    GatherPointKernel<<<dim3(2, 8, 1), 512>>>(batch_size, n_points, m_points,
                                              p_points, p_index, p_out_points);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(gather_point, ops::GatherPointOpCUDAKernel<float>,
                        ops::GatherPointOpCUDAKernel<double>,
                        ops::GatherPointOpCUDAKernel<int>);
