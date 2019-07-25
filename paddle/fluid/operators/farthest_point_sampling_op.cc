/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class FarthestPointSamplingOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor)input points with shape (batch, n, 3), n is input "
             "points's num");
    AddOutput("Output",
              "(Tensor)output points's index with shape (batch, m), m is "
              "output points's num");
    AddAttr<int>("sampled_point_num",
                 R"Doc(sampling points's num)Doc")
        .SetDefault(0)
        .EqualGreaterThan(0);
    AddComment(
        R"Doc(Fathest Point Sampling)Doc");
  }
};

class FarthestPointSamplingOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) shoud not be null");
    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE(x_dims.size() == 3,
                   "Input(X) of FathestPointSamplingOp should be 3-D Tensor");
    const int m = ctx->Attrs().Get<int>("sampled_point_num");
    ctx->SetOutputDim("Output", {x_dims[0], m});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = ctx.Input<Tensor>("X")->type();
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(farthest_point_sampling, ops::FarthestPointSamplingOp,
                  ops::FarthestPointSamplingOpMaker);
