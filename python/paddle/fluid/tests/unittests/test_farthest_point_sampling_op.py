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


def farthest_point_sampling(xyz, npoint):
    B, N, C = xyz.shape
    S = npoint

    centroids = np.zeros((B, S))
    distance = np.ones((B, N)) * 1e10
    # randomly select first point
    farthest = 0  #np.random.randint(0, N, (B,))
    batch_indices = np.arange(B)

    for i in range(S):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].reshape((B, 1, 3))
        dist = np.sum((xyz - centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)

    return centroids


class TestFarthestPointSamplingOp(OpTest):
    def setUp(self):
        x = np.random.randint(1, 100, (300)).reshape((1, 100, 3))
        self.op_type = 'farthest_point_sampling'
        self.inputs = {'X': x}
        self.attrs = {'sampled_point_num', 50}
        self.outputs = {'Output', farthest_point_sampling(x, 50)}

    def test_check_output(self):
        self.check_output(atol=1e-3)

    #def test_check_grad_normal(self):
    #    self.check_grad(['X'], 'Output', max_relative_error=0.61) 


if __name__ == "__main__":
    unittest.main()
