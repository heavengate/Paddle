#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import math
import sys
import paddle.fluid.core as core
from op_test import OpTest


class TestROIPoolOp(OpTest):
    def set_data(self):
        self.init_test_case()
        self.make_boxes()
        self.calc_roi_pool()

        self.inputs = {
            'pts': self.points[:, :, :3],
            'pts_feature': self.points[:, :, 3:9],
            'boxes3d': self.boxes3d[:, :, 1:8]
        }
        self.attrs = {
            'pool_extra_width': self.pool_extra_width,
            'sampled_pt_num': self.sample_pt_num,
        }

        self.outputs = {
            'Out': self.outs,
            'pooled_empty_flag': self.pooled_empty_flag
        }

    def init_test_case(self):
        self.batch_size = 2
        self.pts_num = 3
        self.boxes_num = 2
        self.channels = 9
        self.pool_extra_width = 1
        self.sample_pt_num = 5

        # n, c, d, h, w
        self.points_dim = (self.batch_size, self.pts_num, self.channels)

        self.points = np.random.randint(0, 5, self.points_dim).astype('float32')

    def calc_roi_pool(self):
        def pt_in_box(x, y, z, cx, bottom_y, cz, h, w, l, angle):
            max_ids = 10.0
            cy = bottom_y - h / 2.0
            if ((abs(x - cx) > max_ids) or (abs(y - cy) > h / 2.0) or
                (abs(z - cz) > max_ids)):
                return 0
            cosa = math.cos(angle)
            sina = math.sin(angle)
            x_rot = (x - cx) * cosa + (z - cz) * (-sina)

            z_rot = (x - cx) * sina + (z - cz) * cosa

            flag = (x_rot >= -l / 2.0) and (x_rot <= l / 2.0) and (
                z_rot >= -w / 2.0) and (z_rot <= w / 2.0)
            return flag

        out_data = np.zeros((self.batch_size, self.boxes_num,
                             self.sample_pt_num, self.channels))
        empty_flag_data = np.zeros((self.batch_size, self.boxes_num))

        for bs in range(self.batch_size):
            for i in range(self.boxes_num):
                cnt = 0
                box = self.boxes3d[bs][i]
                box_batch_id = int(box[0])
                cx = int(box[1])
                bottom_y = int(box[2])
                cz = int(box[3])
                h = int(box[4])
                w = int(box[5])
                l = int(box[6])
                ry = box[7]
                x_i = self.points[bs]  # (batch_id=0, pts_num, channle=9)
                for j in range(self.pts_num):
                    x = x_i[j][0]
                    y = x_i[j][1]
                    z = x_i[j][2]

                    cur_in_flag = pt_in_box(x, y, z, cx, bottom_y, cz, h, w, l,
                                            ry)

                    if cur_in_flag:
                        if cnt < self.sample_pt_num:
                            out_data[bs][i][cnt] = x_i[j]
                            cnt += 1
                        else:
                            break
                if cnt == 0:
                    empty_flag_data[bs][i] = 1

                elif (cnt < self.sample_pt_num):
                    for k in range(cnt, self.sample_pt_num):
                        out_data[bs][i][k] = out_data[bs][i][k % cnt]

        self.outs = out_data.astype('float32')
        self.pooled_empty_flag = empty_flag_data.astype('int64')

    def make_boxes(self):
        boxes = []
        #self.boxes_num = 4

        for i in range(self.batch_size):
            tmp = []
            for box_num in range(self.boxes_num):
                x = np.random.random_integers(
                    0, int(max(self.points[i][:, 0]) + 0.5)) / 2.0
                y = np.random.random_integers(
                    0, int(max(self.points[i][:, 1]) + 0.5)) / 2.0
                z = np.random.random_integers(
                    0, int(max(self.points[i][:, 2]) + 0.5)) / 2.0

                h = np.random.random_integers(3, 6)
                w = np.random.random_integers(3, 6)
                l = np.random.random_integers(3, 6)
                ry = np.random.random(1)

                box = [i, x, y, z, h, w, l, ry]
                tmp.append(box)
            boxes.append(tmp)
        self.boxes3d = np.array(boxes).astype("float32")

    def setUp(self):
        self.op_type = "roi_pool_3d"
        self.set_data()

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place, atol=0)


if __name__ == '__main__':
    unittest.main()
