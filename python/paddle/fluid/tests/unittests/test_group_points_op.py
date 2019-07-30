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


def group_points(x, idx):
    b, m, s = idx.shape
    _, n, c = x.shape

    output = np.zeros((b, m, s, c)).astype(x.dtype)
    for i in range(b):
        for j in range(m):
            for k in range(s):
                output[i, j, k, :] = x[i, idx[i, j, k], :]
    return output


class TestGroupPointsOp(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "group_points"
        input_np = np.random.random(self.input_shape).astype("float32")
        idx_np = np.random.uniform(0, self.input_shape[1],
                                   self.idx_shape).astype("int32")

        output = group_points(input_np, idx_np)

        self.inputs = {
            'X': input_np,
            'Idx': idx_np,
        }
        self.outputs = {'Out': output, }

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, {'X'},
                'Out',
                max_relative_error=0.05,
                no_grad_set=['Idx'])

    def init_test_case(self):
        self.input_shape = [8, 43, 29]
        self.idx_shape = [8, 37, 41]


class TestGroupPointsOpCase1(TestGroupPointsOp):
    def init_test_case(self):
        self.input_shape = [8, 33, 37]
        self.idx_shape = [8, 23, 13]


if __name__ == "__main__":
    unittest.main()
