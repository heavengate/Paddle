/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class GroupPointsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of GroupPointsOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Idx"),
                   "Input(Idx) of GroupPointsOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of GroupPointsOp should not be null.");

    auto dim_x = ctx->GetInputDim("X");  // [B, N, C]
    PADDLE_ENFORCE_EQ(dim_x.size(), 3, "X's dimension must be 3");

    auto dim_idx = ctx->GetInputDim("Idx");  // [B, N, S]
    PADDLE_ENFORCE_EQ(dim_idx.size(), 3, "Idx's dimension must be 3");

    PADDLE_ENFORCE_EQ(dim_x[0], dim_idx[0],
                      "X and Idx dim[0] should be equal.");

    // output: [B, M, S, C]
    std::vector<int64_t> dim_out({dim_x[0], dim_idx[1], dim_idx[2], dim_x[2]});
    ctx->SetOutputDim("Out", framework::make_ddim(dim_out));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.GetPlace());
  }
};

class GroupPointsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of group_points operator. "
             "This is a 3-D tensor with shape of [B, N, C].");
    AddInput("Idx",
             "The input tensor of nearest neighbor index of group_points "
             "operator. This is a 3-D tensor with shape of [B, M, S].");
    AddOutput("Out",
              "The output tensor of group_points operator. "
              "This is a 4-D tensor with shape of [B, M, S, C].");

    AddComment(R"DOC(
          This operator group input points with index.
         )DOC");
  }
};

class GroupPointsOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Idx"), "Input(Idx) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto dim_x = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), dim_x);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<Tensor>(framework::GradVarName("Out"))->type(),
        ctx.GetPlace());
  }
};

class GroupPointsGradDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType(ForwardOp().Type() + "_grad");
    op->SetInput("X", Input("X"));
    op->SetInput("Idx", Input("Idx"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(group_points, ops::GroupPointsOp, ops::GroupPointsOpMaker,
                  ops::GroupPointsGradDescMaker);
REGISTER_OPERATOR(group_points_grad, ops::GroupPointsOpGrad);
