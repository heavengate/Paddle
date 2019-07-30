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


def gather_point_np(points, index):
    result = []
    for i in range(len(index)):
        a = points[i][index[i]]
        result.append(a.tolist())
    return result


class TestGatherPointOp(OpTest):
    def setUp(self):
        self.op_type = "gather_point"
        self.config()
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.inputs = {
            'X': xnp,
            'Index': np.array(self.index).astype(self.index_type)
        }
        out = gather_point_np(xnp, self.index)
        self.outputs = {'Output': np.array(out)}

    def config(self):
        self.x_shape = (2, 7, 3)
        self.x_type = "float32"
        self.index = [[1, 3], [0, 5]]
        self.index_type = "int32"

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-1)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, {'X'},
                'Output',
                max_relative_error=0.05,
                no_grad_set=['Index'])


class TestCase1(TestGatherPointOp):
    def config(self):
        """
        For one dimension input
        """
        self.x_shape = (1, 7, 3)
        self.x_type = "float32"
        self.index = [[1]]
        self.index_type = "int32"


class TestCase2(TestGatherPointOp):
    def config(self):
        """
        For input tyep
        """
        self.x_shape = (1, 7, 3)
        self.x_type = "float32"
        self.index = [[1, 2, 3, 4]]
        self.index_type = "int32"


if __name__ == "__main__":
    unittest.main()
