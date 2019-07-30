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


def query_ball_point(points, new_points, radius, nsample):
    b, n, c = points.shape
    _, m, _ = new_points.shape
    out = np.zeros(shape=(b, m, nsample)).astype('int32')
    radius_2 = radius * radius
    for i in range(b):
        for j in range(m):
            cnt = 0
            for k in range(n):
                if (cnt == nsample):
                    break
                dist = np.sum(np.square(points[i][k] - new_points[i][j]))
                if (dist < radius_2):
                    if cnt == 0:
                        out[i][j] = np.ones(shape=(nsample)) * k
                    out[i][j][cnt] = k
                    cnt += 1
    return out


class TestQueryBallOp(OpTest):
    def setUp(self):
        self.op_type = 'query_ball'
        self.config()
        points = np.array(np.random.randint(
            1, 5, size=self.points_shape)).astype(self.points_type)
        new_points = np.array(
            np.random.randint(
                1, 5, size=self.new_points_shape)).astype(self.new_points_type)
        self.inputs = {
            'Points': points,
            'New_Points': new_points,
        }
        self.attrs = {
            'Radius': np.array(self.radius).astype('float32'),
            'N_sample': np.array(self.nsample).astype('int32')
        }
        out = query_ball_point(points, new_points, self.radius, self.nsample)
        self.outputs = {'Output': np.array(out)}

    def config(self):
        self.points_shape = (2, 5, 3)
        self.points_type = 'float32'
        self.new_points_shape = (2, 3, 3)
        self.new_points_type = 'float32'
        self.radius = 6
        self.nsample = 5

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=0)


class TestCase1(TestQueryBallOp):
    def config(self):
        """
	For nsample = 0
	"""
        self.points_shape = (2, 5, 3)
        self.points_type = 'float32'
        self.new_points_shape = (2, 5, 3)
        self.new_points_type = 'float32'
        self.radius = 6
        self.nsample = 0


class TestCase2(TestQueryBallOp):
    def config(self):
        """
	For radius = 0
	"""
        self.points_shape = (2, 5, 3)
        self.points_type = 'float32'
        self.new_points_shape = (2, 5, 3)
        self.new_points_type = 'float32'
        self.radius = 0
        self.nsample = 4


class TestCase3(TestQueryBallOp):
    def config(self):
        """
	For nsample bigger than points
	"""
        self.points_shape = (2, 5, 3)
        self.points_type = 'float32'
        self.new_points_shape = (2, 4, 3)
        self.new_points_type = 'float32'
        self.radius = 5
        self.nsample = 20


if __name__ == "__main__":
    unittest.main()
