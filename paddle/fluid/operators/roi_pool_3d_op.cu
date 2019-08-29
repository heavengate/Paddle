/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <math.h>
#include <stdio.h>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"

#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

__device__ inline int pt_in_box3d(float x, float y, float z, float cx,
                                  float bottom_y, float cz, float h, float w,
                                  float l, float angle, float max_dis) {
  float x_rot, z_rot, cosa, sina, cy;
  int in_flag;
  cy = bottom_y - h / 2.0;
  if ((fabsf(x - cx) > max_dis) || (fabsf(y - cy) > h / 2.0) ||
      (fabsf(z - cz) > max_dis)) {
    return 0;
  }
  cosa = cos(angle);
  sina = sin(angle);
  x_rot = (x - cx) * cosa + (z - cz) * (-sina);
  z_rot = (x - cx) * sina + (z - cz) * cosa;

  in_flag = (x_rot >= -l / 2.0) & (x_rot <= l / 2.0) & (z_rot >= -w / 2.0) &
            (z_rot <= w / 2.0);
  return in_flag;
}

template <typename T>
__global__ void assign_pts_to_box3d(int batch_size, int pts_num, int boxes_num,
                                    const T *xyz, const T *boxes3d,
                                    int *pts_assign) {
  // params xyz: (B, N, 3)
  // params boxes3d: (B, M, 7)
  // params pts_assign: (B, N, M): idx of the corresponding box3d, -1 means
  // background points
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int box_idx = blockIdx.y;
  int bs_idx = blockIdx.z;

  if (pt_idx >= pts_num || box_idx >= boxes_num || bs_idx >= batch_size) {
    return;
  }
  int assign_idx = bs_idx * pts_num * boxes_num + pt_idx * boxes_num + box_idx;
  pts_assign[assign_idx] = 0;

  int box_offset = bs_idx * boxes_num * 7 + box_idx * 7;
  int pt_offset = bs_idx * pts_num * 3 + pt_idx * 3;

  int cur_in_flag = pt_in_box3d(
      xyz[pt_offset], xyz[pt_offset + 1], xyz[pt_offset + 2],
      boxes3d[box_offset], boxes3d[box_offset + 1], boxes3d[box_offset + 2],
      boxes3d[box_offset + 3], boxes3d[box_offset + 4], boxes3d[box_offset + 5],
      boxes3d[box_offset + 6], 10.0);

  pts_assign[assign_idx] = cur_in_flag;
  // printf("bs=%d, pt=%d, in=%d\n", bs_idx, pt_idx, pts_assign[bs_idx * pts_num
  // + pt_idx]);
}

__global__ void get_pooled_idx(int batch_size, int pts_num, int boxes_num,
                               int sampled_pts_num, const int *pts_assign,
                               int *pts_idx, int *pooled_empty_flag) {
  // params xyz: (B, N, 3)
  // params pts_feature: (B, N, C)
  // params pts_assign: (B, N)
  // params pts_idx: (B, M, 512)
  // params pooled_empty_flag: (B, M)

  int boxes_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (boxes_idx >= boxes_num) {
    return;
  }

  int bs_idx = blockIdx.y;

  int cnt = 0;
  for (int k = 0; k < pts_num; k++) {
    if (pts_assign[bs_idx * pts_num * boxes_num + k * boxes_num + boxes_idx]) {
      if (cnt < sampled_pts_num) {
        pts_idx[bs_idx * boxes_num * sampled_pts_num +
                boxes_idx * sampled_pts_num + cnt] = k;
        cnt++;
      } else {
        break;
      }
    }
  }

  if (cnt == 0) {
    pooled_empty_flag[bs_idx * boxes_num + boxes_idx] = 1;
  } else if (cnt < sampled_pts_num) {
    // duplicate same points for sampling
    for (int k = cnt; k < sampled_pts_num; k++) {
      int duplicate_idx = k % cnt;
      int base_offset =
          bs_idx * boxes_num * sampled_pts_num + boxes_idx * sampled_pts_num;
      pts_idx[base_offset + k] = pts_idx[base_offset + duplicate_idx];
    }
  }
}

template <typename T>
__global__ void roipool3d_forward(int batch_size, int pts_num, int boxes_num,
                                  int feature_in_len, int sampled_pts_num,
                                  const T *xyz, const int *pts_idx,
                                  const T *pts_feature, T *pooled_features,
                                  int *pooled_empty_flag) {
  // params xyz: (B, N, 3)
  // params pts_idx: (B, M, 512)
  // params pts_feature: (B, N, C)
  // params pooled_features: (B, M, 512, 3+C)
  // params pooled_empty_flag: (B, M)

  int sample_pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int box_idx = blockIdx.y;
  int bs_idx = blockIdx.z;

  if (sample_pt_idx >= sampled_pts_num || box_idx >= boxes_num ||
      bs_idx >= batch_size) {
    return;
  }

  if (pooled_empty_flag[bs_idx * boxes_num + box_idx]) {
    return;
  }

  int temp_idx = bs_idx * boxes_num * sampled_pts_num +
                 box_idx * sampled_pts_num + sample_pt_idx;
  int src_pt_idx = pts_idx[temp_idx];
  int dst_feature_offset = temp_idx * (3 + feature_in_len);

  for (int j = 0; j < 3; j++)
    pooled_features[dst_feature_offset + j] =
        xyz[bs_idx * pts_num * 3 + src_pt_idx * 3 + j];

  int src_feature_offset =
      bs_idx * pts_num * feature_in_len + src_pt_idx * feature_in_len;
  for (int j = 0; j < feature_in_len; j++)
    pooled_features[dst_feature_offset + 3 + j] =
        pts_feature[src_feature_offset + j];
}

template <typename T>
class ROIPool3dOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto *pts = ctx.Input<Tensor>("pts");
    auto *pts_feature = ctx.Input<Tensor>("pts_feature");
    auto *boxes3d = ctx.Input<Tensor>("boxes3d");
    // auto *sample_num = ctx.Input<int>("sampled_pt_num");
    auto *output = ctx.Output<Tensor>("Out");
    auto *pooled_empty_flag = ctx.Output<Tensor>("pooled_empty_flag");

    if (pts->numel() == 0) return;

    const T *p_xyz = pts->data<T>();
    const T *p_boxes3d = boxes3d->data<T>();
    const T *p_pts_feature = pts_feature->data<T>();
    T *p_pooled_features = output->mutable_data<T>(ctx.GetPlace());
    int *p_pooled_empty_flag =
        pooled_empty_flag->mutable_data<int>(ctx.GetPlace());

    int batch_size = pts->dims()[0];
    int pts_num = pts->dims()[1];
    int boxes_num = boxes3d->dims()[1];
    int feature_in_len = pts_feature->dims()[2];
    int sampled_pts_num = output->dims()[2];

    // printf("batch_size=%d, pts_num=%d, boxes_num=%d\n", batch_size, pts_num,
    // boxes_num);
    int *pts_assign = NULL;
    cudaMalloc(&pts_assign, batch_size * pts_num * boxes_num *
                                sizeof(int));  // (batch_size, N, M)
    // cudaMemset(&pts_assign, -1, batch_size * pts_num * boxes_num *
    // sizeof(int));

    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num,
                batch_size);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    assign_pts_to_box3d<<<blocks, threads>>>(batch_size, pts_num, boxes_num,
                                             p_xyz, p_boxes3d, pts_assign);

    int *pts_idx = NULL;
    cudaMalloc(&pts_idx, batch_size * boxes_num * sampled_pts_num *
                             sizeof(int));  // (batch_size, M, sampled_pts_num)

    dim3 blocks2(DIVUP(boxes_num, THREADS_PER_BLOCK),
                 batch_size);  // blockIdx.x(col), blockIdx.y(row)
    get_pooled_idx<<<blocks2, threads>>>(batch_size, pts_num, boxes_num,
                                         sampled_pts_num, pts_assign, pts_idx,
                                         p_pooled_empty_flag);

    dim3 blocks_pool(DIVUP(sampled_pts_num, THREADS_PER_BLOCK), boxes_num,
                     batch_size);
    roipool3d_forward<<<blocks_pool, threads>>>(
        batch_size, pts_num, boxes_num, feature_in_len, sampled_pts_num, p_xyz,
        pts_idx, p_pts_feature, p_pooled_features, p_pooled_empty_flag);

    cudaFree(pts_assign);
    cudaFree(pts_idx);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(roi_pool_3d, ops::ROIPool3dOpCUDAKernel<float>,
                        ops::ROIPool3dOpCUDAKernel<double>);
