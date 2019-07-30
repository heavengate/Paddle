# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core


def farthest_point_sampling(xyz, npoint):
    B, N, C = xyz.shape
    S = npoint

    centroids = np.zeros((B, S))
    distance = np.ones((B, N)) * 1e10
    farthest = 0
    batch_indices = np.arange(B).astype('int32')
    for i in range(S):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].reshape((B, 1, 3))
        dist = np.sum((xyz - centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids.astype('int32')


class TestFarthestPointSamplingOp(OpTest):
    def setUp(self):
        self.op_type = 'farthest_point_sampling'
        self.config()
        x = np.random.randint(1, 100,
                              (self.x_shape[0] * self.x_shape[1] *
                               3, )).reshape(self.x_shape).astype(self.x_type)
        m = self.sampled_point_num
        out_np = farthest_point_sampling(x, m)
        self.inputs = {'X': x, }
        self.attrs = {'sampled_point_num': m, }
        self.outputs = {'Output': out_np, }

    def config(self):
        self.x_shape = (1, 512, 3)
        self.x_type = 'float32'
        self.sampled_point_num = 256

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
