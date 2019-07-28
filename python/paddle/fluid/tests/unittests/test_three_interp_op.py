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


def three_interp(x, weight, idx):
    b, m, c = x.shape
    n = weight.shape[1]

    output = np.zeros((b, n, c)).astype('float32')
    for i in range(b):
        for j in range(n):
            w1, w2, w3 = weight[i, j, :]
            i1, i2, i3 = idx[i, j, :]
            output[i, j, :] = w1 * x[i, i1, :] \
                            + w2 * x[i, i2, :] \
                            + w3 * x[i, i3, :]
    return output


class TestThreeInterpOp(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "three_interp"
        input_np = np.random.random(self.input_shape).astype("float32")
        weight_np = np.random.random(self.weight_shape).astype("float32")
        idx_np = np.random.uniform(0, self.input_shape[1],
                                   self.weight_shape).astype("int32")

        output = three_interp(input_np, weight_np, idx_np)

        self.inputs = {
            'X': input_np,
            'Weight': weight_np,
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
                no_grad_set=['Weight', 'Idx'])

    def init_test_case(self):
        self.input_shape = [8, 21, 29]
        self.weight_shape = [8, 37, 3]


class TestThreeInterpOpCase1(TestThreeInterpOp):
    def init_test_case(self):
        self.input_shape = [8, 33, 37]
        self.weight_shape = [8, 23, 3]


if __name__ == "__main__":
    unittest.main()
