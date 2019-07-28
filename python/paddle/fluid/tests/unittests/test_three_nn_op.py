#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core


def three_nn(x, known, eps=1e-10):
    distance = np.ones_like(x).astype('float32') * 1e40
    idx = np.zeros_like(x).astype('int32')

    b, n, _ = x.shape
    m = known.shape[1]
    for i in range(b):
        for j in range(n):
            for k in range(m):
                sub = x[i, j, :] - known[i, k, :]
                d = float(np.sum(sub * sub))
                valid_d = max(d, eps)
                if d < distance[i, j, 0]:
                    distance[i, j, 2] = distance[i, j, 1]
                    idx[i, j, 2] = idx[i, j, 1]
                    distance[i, j, 1] = distance[i, j, 0]
                    idx[i, j, 1] = idx[i, j, 0]
                    distance[i, j, 0] = valid_d
                    idx[i, j, 0] = k
                elif d < distance[i, j, 1]:
                    distance[i, j, 2] = distance[i, j, 1]
                    idx[i, j, 2] = idx[i, j, 1]
                    distance[i, j, 1] = valid_d
                    idx[i, j, 1] = k
                elif d < distance[i, j, 2]:
                    distance[i, j, 2] = valid_d
                    idx[i, j, 2] = k
    return distance, idx


class TestThreeNNOp(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "three_nn"
        input_np = np.random.random(self.input_shape).astype("float32")
        known_np = np.random.random(self.known_shape).astype("float32")

        distance, idx = three_nn(input_np, known_np, self.eps)

        self.inputs = {
            'X': input_np,
            'Known': known_np,
        }
        self.attrs = {'eps': self.eps, }
        self.outputs = {
            'Distance': distance,
            'Idx': idx,
        }

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)

    def init_test_case(self):
        self.input_shape = [8, 16, 3]
        self.known_shape = [8, 32, 3]
        self.eps = 1e-10


class TestThreeNNOpCase1(TestThreeNNOp):
    def init_test_case(self):
        self.input_shape = [16, 32, 3]
        self.known_shape = [16, 8, 3]
        self.eps = 1e-10


if __name__ == "__main__":
    unittest.main()
