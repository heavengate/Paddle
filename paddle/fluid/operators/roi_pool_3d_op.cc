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

#include <memory>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
// using LoDTensor = framework::LoDTensor;

class ROIPool3DOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("pts"),
                   "Input(pts) of ROIPool3DOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("pts_feature"),
                   "Input(pts_feature) of ROIPool3DOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("boxes3d"),
                   "Input(boxes3d) of ROIPool3DOp should not be null.");
    // PADDLE_ENFORCE(ctx->HasInput("pool_extra_width"),
    //               "Input(pool_extra_width) of ROIPool3DOp should not be
    //               null.");
    // PADDLE_ENFORCE(ctx->HasInput("sampled_pt_num"),
    //               "Input(sampled_pt_num) of ROIPool3DOp should not be
    //               null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ROIPool3DOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("pooled_empty_flag"),
        "Output(pooled_empty_flag) of ROIPool3DOp should not be null.");

    auto pts_dims = ctx->GetInputDim("pts");
    PADDLE_ENFORCE(pts_dims.size() == 3 && pts_dims[2] == 3,
                   "The format of pts tensor is (B,N,3).");

    auto pts_feature_dims = ctx->GetInputDim("pts_feature");
    PADDLE_ENFORCE(pts_feature_dims.size() == 3,
                   "The format of pts_feature tensor is (B, N, C).");

    auto boxes3d_dims = ctx->GetInputDim("boxes3d");
    PADDLE_ENFORCE(boxes3d_dims.size() == 3 && boxes3d_dims[2] == 7,
                   "The format of boxes3d tensor is (B, M, 7");

    int sampled_pt_num = ctx->Attrs().Get<int>("sampled_pt_num");
    ctx->SetOutputDim("Out", {pts_dims[0], boxes3d_dims[1], sampled_pt_num,
                              3 + pts_feature_dims[2]});
    ctx->SetOutputDim("pooled_empty_flag", {pts_dims[0], boxes3d_dims[1]});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::Tensor>("pts")->type(),
                                   ctx.device_context());
  }
};

class ROIPool3DOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("pts",
             "(Tensor), "
             "The input points of ROI3DPoolOp. "
             "The format of points is (B, N, 3). Where B is batch size, "
             "N is number of points"
             "3 is the points channels.");
    AddInput("pts_feature",
             "(Tensor), "
             "The points feature of ROI3DPoolOp. "
             "The format of points feature is (B, N, C). Where B is batch size,"
             "N is number of points feature"
             "C is the points feature channels.");
    AddInput("boxes3d",
             "(Tensor)"
             "The input boxes3d of ROI3DPoolOp."
             "The format of boxes3d is (B, M, 7).Where B is batch size,"
             "M is number of boxes3d,"
             "7 is the boxes3d channels.");
    AddOutput("Out",
              "(Tensor), "
              "The output of ROIPoolOp is a 5-D tensor with shape "
              "(B, M, sampled_pt_num, 3+C).");
    AddOutput("pooled_empty_flag",
              "(Tensor), "
              "pooled_empty_flag")  // todo:add annotation by yxt
        .AsIntermediate();
    AddAttr<float>("pool_extra_width",
                   "(float, default 1.0), "
                   "what is pool extra width")
        .SetDefault(1.0);
    AddAttr<int>("sampled_pt_num",
                 "(int, default 512),"
                 "The number of sampled points.")
        .SetDefault(512);

    AddComment(R"DOC(
**ROIPool3D Operator**

Region of interest pooling (also known as RoI pooling) is to perform
is to perform max pooling on inputs of nonuniform sizes to obtain
fixed-size feature maps (e.g. 7*7).

The operator has three steps:

1. Dividing each region proposal into equal-sized sections with
   the pooled_width and pooled_height

2. Finding the largest value in each section

3. Copying these max values to the output buffer

ROI Pooling for Faster-RCNN. The link below is a further introduction: 
https://stackoverflow.com/questions/43430056/what-is-roi-layer-in-fast-rcnn
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(roi_pool_3d, ops::ROIPool3DOp, ops::ROIPool3DOpMaker);
