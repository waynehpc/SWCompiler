/*************************************************************************
        > File Name: dlOp.h
        > Author: cryinlaugh
        > Mail: cryinlaugh@gmail.com
        > Created Time: 二 12/ 4 15:57:29 2018
 ************************************************************************/

#ifndef _DLOP_H
#define _DLOP_H

#include "op/basicOp/basicOps.h"

namespace swc {
namespace op {

//=====================================================
// Definition of 2-D deep learning specific operations.
// Version v0.1: ops for simple-MLP listed below
//--MatrixMatrixFC
//--Activation:
//----MatrixTanh
//--MatrixSoftmax
//--Loss:
//----MatrixLogNegLoss
//=====================================================

class MatrixMatrixFCOp : public Op {
    // input, weights
    // output
  public:
    MatrixMatrixFCOp() : Op(DL_OP, 2, 1, std::string("MatrixMatrixFC")) {
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);

        this->_einOp = 1;
        this->_einRep.push_back("ik");
        this->_einRep.push_back("kj");
        this->_einRep.push_back("ij");
    }
    ~MatrixMatrixFCOp() {}
    void destroy(){};

    void checkValid(OpNode *node);
    void outTensorTypeGen(OpNode *node, size_t index, Tensor *tensor) override;
    // for lowering
    void autoDiff(IRGraph *graph, IRNode *opNode,
                  std::unordered_map<IRNode *, IRNode *> &gradNodeMap);

    void einsumLowering(IRGraph *graph, IRNode *node);

    // for common lowering
    void lowering(IRGraph *graph, IRNode *node);
    void paralleling(IRGraph *graph, IRNode *node);
};

// update FCGrad do not need output
// with output will bring none-use communication cost
// op should keep with 0. audodiff 1. lowering 2. codegen
class MatrixMatrixFCGradOp : public Op {
    // input, wieght, bias, orig_output, orig_output_grad
    // input_grad, weight_grad, bias_grad
  public:
    MatrixMatrixFCGradOp()
        : Op(DL_OP, 3, 2, std::string("MatrixMatrixFCGrad")) {

        this->_einOp = 1;
        this->_einRep.push_back("ik"); // input
        this->_einRep.push_back("kj"); // weight
        // this->_einRep.push_back("ij"); // orig_output
        this->_einRep.push_back("ij"); // outputGrad
        this->_einRep.push_back("ik"); // inputGrad
        this->_einRep.push_back("kj"); // weightGrad
    }
    ~MatrixMatrixFCGradOp() {}
    void destroy() {}

    // for lowering
    void einsumLowering(IRGraph *graph, IRNode *node);
    // for common lowering
    void lowering(IRGraph *graph, IRNode *node);
};

class MatrixMatrixFCBiasOp : public Op {
    // input, weight, bias
    // output
  public:
    MatrixMatrixFCBiasOp()
        : Op(DL_OP, 3, 1, std::string("MatrixMatrixFCBias")) {
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(2);

        this->_einOp = 1;
        this->_einRep.push_back("ik"); // input
        this->_einRep.push_back("kj"); // weight
        this->_einRep.push_back("j");  // bias
        this->_einRep.push_back("ij"); // out
    }
    ~MatrixMatrixFCBiasOp() {}
    void destroy(){};

    void checkValid(OpNode *node);
    void outTensorTypeGen(OpNode *node, size_t index, Tensor *tensor) override;
    // for lowering
    void autoDiff(IRGraph *graph, IRNode *opNode,
                  std::unordered_map<IRNode *, IRNode *> &gradNodeMap);
    void einsumLowering(IRGraph *graph, IRNode *node);
    // for common lowering
    void lowering(IRGraph *graph, IRNode *node);
};

class MatrixMatrixFCBiasGradOp : public Op {
    // input, wieght, bias, orig_output, orig_output_grad
    // input_grad, weight_grad, bias_grad
  public:
    MatrixMatrixFCBiasGradOp()
        : Op(DL_OP, 4, 3, std::string("MatrixMatrixFCBiasGrad")) {

        this->_einOp = 1;
        this->_einRep.push_back("ik"); // input
        this->_einRep.push_back("kj"); // weight
        this->_einRep.push_back("j");  // bias
        // this->_einRep.push_back("ij"); // orig_output
        this->_einRep.push_back("ij"); // outputGrad

        this->_einRep.push_back("ik"); // inputGrad
        this->_einRep.push_back("kj"); // weightGrad
        this->_einRep.push_back("j");  // biasGrad
    }
    ~MatrixMatrixFCBiasGradOp() {}
    void destroy() {}

    // for lowering
    void einsumLowering(IRGraph *graph, IRNode *node);
    // for common lowering
    void lowering(IRGraph *graph, IRNode *node);
    void paralleling();
};

class ReshapeOp : public Op {
    std::vector<size_t> oshape_;

  public:
    ReshapeOp()
        : Op(DL_OP, 1, 1, std::string("Reshape")){
              /* unknown
              this->_inputNDims.push_back(4);
              this->_outputNDims.push_back(4);
              */
          };
    ReshapeOp(std::vector<size_t> &shape)
        : Op(DL_OP, 1, 1, std::string("Reshape")) {
        oshape_.assign(shape.begin(), shape.end());
    }
    std::vector<size_t> getOutShape() { return oshape_; }
    ~ReshapeOp();
    void destroy() {}
};

class MatrixTanhOp : public Op {
  public:
    MatrixTanhOp() : Op(DL_OP, 1, 1, std::string("MatrixTanh")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
        this->_einOp = 1;
        this->_einRep.push_back("ij");
        this->_einRep.push_back("ij");
    };
    ~MatrixTanhOp();
    void destroy(){};
    void autoDiff(IRGraph *graph, IRNode *opNode,
                  std::unordered_map<IRNode *, IRNode *> &gradNodeMap);
};

class MatrixTanhGradOp : public Op {
  public:
    MatrixTanhGradOp() : Op(DL_OP, 3, 1, std::string("MatrixTanhGrad")) {
        this->_einOp = 1;
        this->_einRep.push_back("ij");
        this->_einRep.push_back("ij");
        this->_einRep.push_back("ij");
        this->_einRep.push_back("ij");
    }
    ~MatrixTanhGradOp();
    void destroy() {}
};

class MatrixSoftmaxOp : public Op {
  public:
    MatrixSoftmaxOp() : Op(DL_OP, 1, 1, std::string("MatrixSoftmax")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);

        this->_einOp = 1;
        this->_einRep.push_back("i_");
        this->_einRep.push_back("i_");
    };
    ~MatrixSoftmaxOp();
    void checkValid(OpNode *node);
    void destroy(){};
    void autoDiff(IRGraph *graph, IRNode *opNode,
                  std::unordered_map<IRNode *, IRNode *> &gradNodeMap);
};

class MatrixSoftmaxGradOp : public Op {
  public:
    MatrixSoftmaxGradOp() : Op(DL_OP, 3, 1, std::string("MatrixSoftmaxGrad")){};
    ~MatrixSoftmaxGradOp();
    void destroy() {}
};

class MatrixSoftmaxWithLossOp : public Op {
  public:
    MatrixSoftmaxWithLossOp()
        : Op(DL_OP, 2, 2, std::string("MatrixSoftmaxWithLoss")) {
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
        this->_outputNDims.push_back(1);

        this->_einOp = 1;
        this->_einRep.push_back("i_"); // input
        this->_einRep.push_back("i");  // label
        this->_einRep.push_back("i_"); // output
        this->_einRep.push_back(
            "_"); // loss scalar // error, shoudl not reduce, but mean...
    };
    ~MatrixSoftmaxWithLossOp();
    void checkValid(OpNode *node);
    void destroy(){};
    void autoDiff(IRGraph *graph, IRNode *opNode,
                  std::unordered_map<IRNode *, IRNode *> &gradNodeMap);
};

class MatrixSoftmaxWithLossGradOp : public Op {
  public:
    MatrixSoftmaxWithLossGradOp()
        : Op(DL_OP, 2, 1, std::string("MatrixSoftmaxWithLossGrad")) {
        this->_einOp = 1;
        this->_einRep.push_back("n");  // label
        this->_einRep.push_back("n_"); // origin out
        this->_einRep.push_back("n_"); // grad of input
    };
    ~MatrixSoftmaxWithLossGradOp();
    void destroy() {}
};

class DropoutOp : public Op {
    float ratio_;

  public:
    DropoutOp(float ratio) : Op(DL_OP, 2, 1, std::string("Dropout")) {
        ratio_ = ratio;
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(2); // _mask
        this->_outputNDims.push_back(2);

        this->_einOp = 1;
        this->_einRep.push_back("jk"); // label
        this->_einRep.push_back("jk"); // origin out
        this->_einRep.push_back("jk"); // grad of input
    };
    ~DropoutOp();
    float getRatio() { return ratio_; }
    void destroy() {}
    void autoDiff(IRGraph *graph, IRNode *opNode,
                  std::unordered_map<IRNode *, IRNode *> &gradNodeMap);
};

class SGDOp : public Op {
    // weight weight_grad momentum
    // weight
    float lr_{0.001};
    float decay_{0.001};
    float momentum_{0.9};
    size_t batch_{1};

  public:
    SGDOp() : Op(DL_OP, 3, 1, std::string("SGD")) {}
    SGDOp(float lr, float decay, float momentum, size_t batch)
        : Op(DL_OP, 3, 1, std::string("SGD")), lr_(lr), decay_(decay),
          momentum_(momentum), batch_(batch) {

        this->_einOp = 0;
        this->_einRep.push_back("ijkm"); // w{t}
        this->_einRep.push_back("ijkm"); // dw
        this->_einRep.push_back("ijkm"); // momentum

        this->_einRep.push_back("ijkm"); // w_{t+1}
    }
    ~SGDOp();
    float getLR() { return lr_; }
    float getDecay() { return decay_; }
    float getMomentum() { return momentum_; }
    size_t getBatch() { return batch_; }
    void destroy() {}
};

class MatrixLogNegLossOp : public Op {
  public:
    MatrixLogNegLossOp() : Op(DL_OP, 1, 1, std::string("MatrixLogNegLoss")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(0);
    };
    ~MatrixLogNegLossOp();
    void destroy(){};
};

class MatrixTransOp : public Op {
  public:
    MatrixTransOp() : Op(DL_OP, 1, 1, std::string("MatrixTrans")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
        this->_einOp = 0;
    };
    ~MatrixTransOp();
    void destroy(){};
};

class MatrixAddOp : public Op {
  public:
    MatrixAddOp() : Op(DL_OP, 2, 1, std::string("MatrixAdd")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    };
    ~MatrixAddOp();
    void destroy(){};
};

class ElementAddOp : public Op {
  public:
    ElementAddOp() : Op(DL_OP, 2, 1, std::string("ElementAdd")) {}
    ~ElementAddOp();
    void destroy(){};
    std::string getOpInfo() override;
    void setAttr(int ndim) override {
        switch (ndim) {
        case 1:
            this->setIONDims({1, 1}, {1});
            this->setEinReps({"a", "a", "a"});
            break;
        case 2:
            this->setIONDims({2, 2}, {2});
            this->setEinReps({"ab", "ab", "ab"});
            break;
        case 4:
            this->setIONDims({4, 4}, {4});
            this->setEinReps({"abcd", "abcd", "abcd"});
            break;
        default:
            SWLOG_ERROR << "error, unimplemented io idims\n";
            exit(0);
            break;
        }
    }
    void setIONDims(std::initializer_list<int> indims,
                    std::initializer_list<int> ondims) override {
        for (auto ndim : indims) {
            this->_inputNDims.push_back(ndim);
        }
        for (auto ndim : ondims) {
            this->_outputNDims.push_back(ndim);
        }
    }
    void setEinReps(std::initializer_list<std::string> reps) override {
        this->_einOp = 1;
        for (auto rep : reps) {
            this->_einRep.push_back(rep);
        }
    }

    void autoDiff(IRGraph *graph, IRNode *opNode,
                  std::unordered_map<IRNode *, IRNode *> &gradNodeMap) override;
};

class ElementSubOp : public Op {
  public:
    ElementSubOp() : Op(DL_OP, 2, 1, std::string("ElementSub")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    };
    ~ElementSubOp();
    void destroy(){};
};

class ElementMulOp : public Op {
  public:
    ElementMulOp() : Op(DL_OP, 2, 1, std::string("ElementMul")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);

        this->_einOp = 1;
        this->_einRep.push_back("bc"); // label
        this->_einRep.push_back("bc"); // origin out
        this->_einRep.push_back("bc"); // grad of input
    }
    ~ElementMulOp();
    void destroy() {}
};

class ElementDivOp : public Op {
  public:
    ElementDivOp() : Op(DL_OP, 2, 1, std::string("ElementDiv")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(2);
    };
    ~ElementDivOp();
    void destroy(){};
};

class PrintMatrixOp : public Op {
    PrintStreamType type_;
    std::string outfile_;

  public:
    PrintMatrixOp() : Op(DL_OP, 1, 0, std::string("PrintMatrix")) {
        this->_inputNDims.push_back(2);
    };
    ~PrintMatrixOp();
    void destroy(){};
    void setPrintStream(PrintStreamType type, std::string file = "") {
        type_ = type;
        outfile_ = file;
    }
    PrintStreamType getPrintStreamType() { return type_; }
    std::string getOutFile() { return outfile_; }
};

class ScatterOp : public Op {
    size_t offset_{0};

    int axis_{-1};
    int degree_{1};

  public:
    ScatterOp() : Op(DL_OP, 0, 0, "Scatter"), offset_(0) {}
    ScatterOp(size_t offset) : Op(DL_OP, 0, 0, "Scatter"), offset_(offset) {}
    ScatterOp(int axis, int degree)
        : Op(DL_OP, 0, 0, "Scatter"), axis_(axis), degree_(degree) {}
    ~ScatterOp();

    std::string getOpInfo() override;
    size_t getCost(OpNode *, Config &config) override;
    std::string getCostTrace(OpNode *, Config &config) override;

    // bytes origin tensor
    static size_t getSimCost(size_t bytes, Config &config, int strategy);

    void setOffset(size_t offset) { offset_ = offset; }
    size_t getOffset() { return offset_; }

    void setAxis(int axis) { axis_ = axis; }
    int getAxis() { return axis_; }

    void setDegree(int degree) { degree_ = degree; }
    int getDegree() { return degree_; }
};

class GatherOp : public Op {
    size_t offset_{0};

    int axis_{-1};
    int degree_{1};

  public:
    GatherOp() : Op(DL_OP, 0, 0, "Gather"), offset_(0) {}
    GatherOp(size_t offset) : Op(DL_OP, 0, 0, "Gather"), offset_(offset) {}
    GatherOp(int axis, int degree)
        : Op(DL_OP, 0, 0, "Gather"), axis_(axis), degree_(degree) {}
    ~GatherOp();

    std::string getOpInfo() override;
    size_t getCost(OpNode *, Config &config) override;
    std::string getCostTrace(OpNode *, Config &config) override;
    static size_t getSimCost(size_t bytes, Config &config, int strategy);

    void setOffset(size_t offset) { offset_ = offset; }
    size_t getOffset() { return offset_; }

    void setAxis(int axis) { axis_ = axis; }
    int getAxis() { return axis_; }

    void setDegree(int degree) { degree_ = degree; }
    int getDegree() { return degree_; }
};

class ReduceOp : public Op {

  public:
    ReduceOp() : Op(DL_OP, 0, 0, "Reduce") {}
    ~ReduceOp();
    // std::string getOpInfo() override;
    size_t getCost(OpNode *, Config &config) override;
    std::string getCostTrace(OpNode *, Config &config) override;
    static size_t getSimCost(size_t bytes, Config &config, int strategy);
};

class BroadcastOp : public Op {

  public:
    BroadcastOp() : Op(DL_OP, 0, 0, "Broadcast") {}
    ~BroadcastOp();
    size_t getCost(OpNode *, Config &config) override;
    std::string getCostTrace(OpNode *, Config &config) override;
    static size_t getSimCost(size_t bytes, Config &config, int strategy);
};

class TransformOp : public Op {
    int preAxis_{-1};
    int postAxis_{-1};
    int degree_{1};

  public:
    TransformOp() : Op(DL_OP, 0, 0, "Transform") {}
    TransformOp(int pre_axis, int post_axis, int degree)
        : Op(DL_OP, 0, 0, "Transform"), preAxis_(pre_axis),
          postAxis_(post_axis), degree_(degree) {}
    ~TransformOp() {}

    std::string getOpInfo() override;
    size_t getCost(OpNode *, Config &config) override;
    std::string getCostTrace(OpNode *, Config &config) override;

    static size_t getSimCost(size_t bytes, Config &config, int pre_strategy,
                             int post_strategy);

    void setPreAxis(int axis) { preAxis_ = axis; }
    int getPreAxis() { return preAxis_; }
    void setPostAxis(int axis) { postAxis_ = axis; }
    int getPostAxis() { return postAxis_; }

    void setDegree(int degree) { degree_ = degree; }
    int getDegree() { return degree_; }
};

class SubGraphOp : public Op {
    IRGraph *graph_;

  public:
    SubGraphOp() : Op(DL_OP, 0, 0, "SubGraph") {}
    ~SubGraphOp();
    void setGraph(IRGraph *graph) { graph_ = graph; }
    IRGraph *getGraph() { return graph_; }
};

class Conv2dOp : public Op {
    std::vector<size_t> kernels_;
    std::vector<size_t> strides_;
    std::vector<size_t> pads_;
    int group_{1};

  public:
    Conv2dOp() : Op(DL_OP, 3, 1, std::string("Conv2d")) {
        this->_inputNDims.push_back(4);
        this->_inputNDims.push_back(4);
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(4);

        this->_einOp = 1;
        // warning, strategy will be orderd by char
        // so data parallel
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("o__c"); // w
        this->_einRep.push_back("o");    // b
        this->_einRep.push_back("b__o"); // out
    };
    Conv2dOp(std::vector<size_t> &kernels, std::vector<size_t> &strides,
             std::vector<size_t> &pads)
        : Op(DL_OP, 3, 1, std::string("Conv2d")) {
        kernels_.assign(kernels.begin(), kernels.end());
        strides_.assign(strides.begin(), strides.end());
        pads_.assign(pads.begin(), pads.end());
        this->_inputNDims.push_back(4);
        this->_inputNDims.push_back(4);
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(4);

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("o__c"); // w
        this->_einRep.push_back("o");    // b
        this->_einRep.push_back("b__o"); // out
    }
    std::vector<size_t> getPads() { return pads_; }
    std::vector<size_t> getKernels() { return kernels_; }
    std::vector<size_t> getStrides() { return strides_; }
    size_t getGroup() { return group_; }
    ~Conv2dOp();
    void destroy() override {}

    std::string getOpInfo() override;

    void outTensorTypeGen(OpNode *node, size_t index, Tensor *tensor) override;

    void autoDiff(IRGraph *graph, IRNode *opNode,
                  std::unordered_map<IRNode *, IRNode *> &gradNodeMap) override;
};

class Conv2dGradOp : public Op {
    std::vector<size_t> kernels_;
    std::vector<size_t> strides_;
    std::vector<size_t> pads_;
    int group_{1};

  public:
    // infered input/output maybe (5, 3), bias no need ,so we use (4, 3)
    Conv2dGradOp() : Op(DL_OP, 3, 3, std::string("Conv2dGrad")) {
        this->_inputNDims.push_back(4); // input
        this->_inputNDims.push_back(4); // weight
        this->_inputNDims.push_back(4); // outputG

        this->_outputNDims.push_back(4); // inputG
        this->_outputNDims.push_back(4); // weightG
        this->_outputNDims.push_back(1); // biasG

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("o__c"); // w
        this->_einRep.push_back("b__o"); // outG

        this->_einRep.push_back("b__c"); // inG
        this->_einRep.push_back("o__c"); // wG
        this->_einRep.push_back("o");    // bG
    };
    Conv2dGradOp(std::vector<size_t> &kernels, std::vector<size_t> &strides,
                 std::vector<size_t> &pads)
        : Op(DL_OP, 3, 3, std::string("Conv2dGrad")) {
        kernels_.assign(kernels.begin(), kernels.end());
        strides_.assign(strides.begin(), strides.end());
        pads_.assign(pads.begin(), pads.end());

        this->_inputNDims.push_back(4); // input
        this->_inputNDims.push_back(4); // weight
        this->_inputNDims.push_back(4); // outputG

        this->_outputNDims.push_back(4); // inputG
        this->_outputNDims.push_back(4); // weightG
        this->_outputNDims.push_back(1); // biasG

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("o__c"); // w
        this->_einRep.push_back("b__o"); // outG

        this->_einRep.push_back("b__c"); // inG
        this->_einRep.push_back("o__c"); // wG
        this->_einRep.push_back("o");    // bG
    }

    std::vector<size_t> getPads() { return pads_; }
    std::vector<size_t> getKernels() { return kernels_; }
    std::vector<size_t> getStrides() { return strides_; }
    size_t getGroup() { return group_; }
    ~Conv2dGradOp();
    void destroy() {}
};
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/conv_ops_fused_impl.h
// https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnActivationDescriptor_t
// inherit from Conv2d may be simple
// but current implementation is not suitable, e.g.initialization of
// _opClassName
class Conv2dWithActivationOp : public Op {
    std::vector<size_t> kernels_;
    std::vector<size_t> strides_;
    std::vector<size_t> pads_;
    int group_{1};
    activation_type activation_{SWC_ACTIVATION_RELU};

  public:
    Conv2dWithActivationOp(std::vector<size_t> &kernels,
                           std::vector<size_t> &strides,
                           std::vector<size_t> &pads,
                           activation_type activation)
        : Op(DL_OP, 3, 1, std::string("Conv2dWithActivation")) {
        kernels_.assign(kernels.begin(), kernels.end());
        strides_.assign(strides.begin(), strides.end());
        pads_.assign(pads.begin(), pads.end());
        activation_ = activation;

        this->_inputNDims.push_back(4);
        this->_inputNDims.push_back(4);
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(4);

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("o__c"); // w
        this->_einRep.push_back("o");    // b
        this->_einRep.push_back("b__o"); // out
    }
    std::vector<size_t> getPads() { return pads_; }
    std::vector<size_t> getKernels() { return kernels_; }
    std::vector<size_t> getStrides() { return strides_; }
    size_t getGroup() { return group_; }
    activation_type getActivationType() { return activation_; }

    ~Conv2dWithActivationOp();

    void autoDiff(IRGraph *graph, IRNode *opNode,
                  std::unordered_map<IRNode *, IRNode *> &gradNodeMap) override;
};

class Conv2dWithActivationGradOp : public Op {
    std::vector<size_t> kernels_;
    std::vector<size_t> strides_;
    std::vector<size_t> pads_;
    int group_{1};
    activation_type activation_{SWC_ACTIVATION_RELU};

  public:
    Conv2dWithActivationGradOp(std::vector<size_t> &kernels,
                               std::vector<size_t> &strides,
                               std::vector<size_t> &pads,
                               activation_type activation)
        : Op(DL_OP, 3, 3, std::string("Conv2dWithActivationGrad")) {
        kernels_.assign(kernels.begin(), kernels.end());
        strides_.assign(strides.begin(), strides.end());
        pads_.assign(pads.begin(), pads.end());
        activation_ = activation;

        this->_inputNDims.push_back(4); // input
        this->_inputNDims.push_back(4); // weight
        this->_inputNDims.push_back(4); // outputG

        this->_outputNDims.push_back(4); // inputG
        this->_outputNDims.push_back(4); // weightG
        this->_outputNDims.push_back(1); // biasG

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("o__c"); // w
        this->_einRep.push_back("b__o"); // outG

        this->_einRep.push_back("b__c"); // inG
        this->_einRep.push_back("o__c"); // wG
        this->_einRep.push_back("c");    // bG
    }

    std::vector<size_t> getPads() { return pads_; }
    std::vector<size_t> getKernels() { return kernels_; }
    std::vector<size_t> getStrides() { return strides_; }
    size_t getGroup() { return group_; }
    activation_type getActivationType() { return activation_; }

    ~Conv2dWithActivationGradOp();
    void destroy() {}
};

class BatchNorm2dOp : public Op {
    float epsilon_;
    float momentum_;

  public:
    BatchNorm2dOp(float eps = 1e-5, float momentum = 0.1)
        : Op(DL_OP, 5, 1, std::string("BatchNorm2d")), epsilon_(eps),
          momentum_(momentum) {

        this->_inputNDims.push_back(4); // input
        this->_inputNDims.push_back(1); // mean
        this->_inputNDims.push_back(1); // var
        this->_inputNDims.push_back(1); // scale
        this->_inputNDims.push_back(1); // shift

        this->_outputNDims.push_back(4); // output

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // input
        this->_einRep.push_back("c");    // mean
        this->_einRep.push_back("c");    // var
        this->_einRep.push_back("c");    // scale
        this->_einRep.push_back("c");    // shift

        this->_einRep.push_back("b__c"); // output
    }
    float getEpsilon() { return epsilon_; }
    float getMomentum() { return momentum_; }
    ~BatchNorm2dOp();
    void destroy() {}
    void autoDiff(IRGraph *graph, IRNode *opNode,
                  std::unordered_map<IRNode *, IRNode *> &gradNodeMap) override;
};

class BatchNorm2dGrad : public Op {
    float epsilon_;
    float momentum_;

  public:
    BatchNorm2dGrad(float eps = 1e-5, float momentum = 0.1)
        : Op(DL_OP, 6, 3, std::string("BatchNorm2dGrad")), epsilon_(eps),
          momentum_(momentum) {
        epsilon_ = eps;

        this->_inputNDims.push_back(4); // input
        this->_inputNDims.push_back(1); // mean
        this->_inputNDims.push_back(1); // var
        this->_inputNDims.push_back(1); // scale
        this->_inputNDims.push_back(1); // shift
        this->_inputNDims.push_back(4); // outputG

        this->_outputNDims.push_back(4); // inputG
        this->_outputNDims.push_back(1); // scaleG
        this->_outputNDims.push_back(1); // shiftG

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // input
        this->_einRep.push_back("c");    // mean
        this->_einRep.push_back("c");    // var
        this->_einRep.push_back("c");    // scale
        this->_einRep.push_back("c");    // shift
        this->_einRep.push_back("b__c"); // outputG

        this->_einRep.push_back("b__c"); // inputG
        this->_einRep.push_back("c");    // scaleG
        this->_einRep.push_back("c");    // shiftG
    }

    float getEpsilon() { return epsilon_; }
    float getMomentum() { return momentum_; }

    ~BatchNorm2dGrad() = default;
    void destroy() {}
};

// TODO: LRN use scale_cache
class LRNOp : public Op {
    float alpha_{1e-4};
    float beta_{0.75};
    float k_{1.0};
    size_t windowSize_{2};

  public:
    LRNOp() : Op(DL_OP, 1, 1, std::string("LRN")) {

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("b__c"); // out
    }

    LRNOp(float alpha, float beta, float k, size_t n)
        : Op(DL_OP, 1, 1, std::string("LRN")) {
        alpha_ = alpha;
        beta_ = beta;
        k_ = k;
        windowSize_ = n;

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("b__c"); // out
    }

    float getAlpha() const { return alpha_; }
    float getBeta() const { return beta_; }
    float getK() const { return k_; }
    float getN() const { return windowSize_; }

    void autoDiff(IRGraph *graph, IRNode *opNode,
                  std::unordered_map<IRNode *, IRNode *> &gradNodeMap) override;
};

class LRNGradOp : public Op {
    float alpha_{1e-4};
    float beta_{0.75};
    float k_{1.0};
    size_t windowSize_{2};

  public:
    LRNGradOp() : Op(DL_OP, 3, 1, std::string("LRNGrad")) {

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("b__c"); // out
        this->_einRep.push_back("b__c"); // outGrad

        this->_einRep.push_back("b__c"); // inGrad
    }

    LRNGradOp(float alpha, float beta, float k, size_t n)
        : Op(DL_OP, 1, 1, std::string("LRNGrad")) {
        alpha_ = alpha;
        beta_ = beta;
        k_ = k;
        windowSize_ = n;

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("b__c"); // out
    }

    float getAlpha() const { return alpha_; }
    float getBeta() const { return beta_; }
    float getK() const { return k_; }
    float getN() const { return windowSize_; }
};

class ReluOp : public Op {
  public:
    ReluOp() : Op(DL_OP, 1, 1, std::string("Relu")) {
        this->_inputNDims.push_back(4);
        this->_outputNDims.push_back(4);

        this->_einOp = 1;
        this->_einRep.push_back("bhwc"); // in
        this->_einRep.push_back("bhwc"); // out
    }
    ~ReluOp();
    void checkValid(OpNode *node);
    void destroy() {}
    void autoDiff(IRGraph *graph, IRNode *opNode,
                  std::unordered_map<IRNode *, IRNode *> &gradNodeMap);
};

class ReluGradOp : public Op {
  public:
    ReluGradOp() : Op(DL_OP, 3, 1, std::string("ReluGrad")) {
        this->_inputNDims.push_back(4);
        this->_inputNDims.push_back(4);
        this->_inputNDims.push_back(4);
        this->_outputNDims.push_back(4);

        this->_einOp = 1;
        this->_einRep.push_back("bhwc"); // in
        this->_einRep.push_back("bhwc"); // out
        this->_einRep.push_back("bhwc"); // outGrad

        this->_einRep.push_back("bhwc");
    }
    ~ReluGradOp();
    void destroy() {}
};

class MaxPoolOp : public Op {
    std::vector<size_t> kernels_;
    std::vector<size_t> strides_;
    std::vector<size_t> pads_;

  public:
    MaxPoolOp() : Op(DL_OP, 1, 1, std::string("MaxPool")) {
        this->_inputNDims.push_back(4);
        this->_outputNDims.push_back(4);

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("b__c"); // out
    }
    MaxPoolOp(std::vector<size_t> &kernels, std::vector<size_t> &strides,
              std::vector<size_t> &pads)
        : Op(DL_OP, 1, 1, std::string("MaxPool")) {
        kernels_.assign(kernels.begin(), kernels.end());
        strides_.assign(strides.begin(), strides.end());
        pads_.assign(pads.begin(), pads.end());
        this->_inputNDims.push_back(4);
        this->_outputNDims.push_back(4);

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("b__c"); // out
    }
    ~MaxPoolOp();
    std::vector<size_t> getPads() { return pads_; }
    std::vector<size_t> getKernels() { return kernels_; }
    std::vector<size_t> getStrides() { return strides_; }
    void destroy() override {}

    std::string getOpInfo() override;

    void outTensorTypeGen(OpNode *node, size_t index, Tensor *tensor) override;

    void autoDiff(IRGraph *graph, IRNode *opNode,
                  std::unordered_map<IRNode *, IRNode *> &gradNodeMap) override;
};

class MaxPoolGradOp : public Op {
    std::vector<size_t> kernels_;
    std::vector<size_t> strides_;
    std::vector<size_t> pads_;

  public:
    MaxPoolGradOp() : Op(DL_OP, 3, 1, std::string("MaxPoolGrad")) {
        this->_inputNDims.push_back(4); // input
        this->_inputNDims.push_back(4); // output
        this->_inputNDims.push_back(4); // outputG

        this->_outputNDims.push_back(4); // inputG

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("b__c"); // out
        this->_einRep.push_back("b__c"); // outG

        this->_einRep.push_back("b__c"); // inG
    }
    MaxPoolGradOp(std::vector<size_t> &kernels, std::vector<size_t> &strides,
                  std::vector<size_t> &pads)
        : Op(DL_OP, 3, 1, std::string("MaxPoolGrad")) {
        kernels_.assign(kernels.begin(), kernels.end());
        strides_.assign(strides.begin(), strides.end());
        pads_.assign(pads.begin(), pads.end());

        this->_inputNDims.push_back(4); // input
        this->_inputNDims.push_back(4); // output
        this->_inputNDims.push_back(4); // outputG

        this->_outputNDims.push_back(4); // inputG

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("b__c"); // out
        this->_einRep.push_back("b__c"); // outG

        this->_einRep.push_back("b__c"); // inG
    }
    ~MaxPoolGradOp();
    std::vector<size_t> getPads() { return pads_; }
    std::vector<size_t> getKernels() { return kernels_; }
    std::vector<size_t> getStrides() { return strides_; }
    void destroy() {}
};

class AvgPoolOp : public Op {
    std::vector<size_t> kernels_;
    std::vector<size_t> strides_;
    std::vector<size_t> pads_;

  public:
    AvgPoolOp() : Op(DL_OP, 1, 1, std::string("AveragePool")) {
        this->_inputNDims.push_back(4);
        this->_outputNDims.push_back(4);

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("b__c"); // out
    }
    AvgPoolOp(std::vector<size_t> &kernels, std::vector<size_t> &strides,
              std::vector<size_t> &pads)
        : Op(DL_OP, 1, 1, std::string("AveragePool")) {
        kernels_.assign(kernels.begin(), kernels.end());
        strides_.assign(strides.begin(), strides.end());
        pads_.assign(pads.begin(), pads.end());
        this->_inputNDims.push_back(4);
        this->_outputNDims.push_back(4);

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("b__c"); // out
    }
    ~AvgPoolOp();
    std::vector<size_t> getPads() { return pads_; }
    std::vector<size_t> getKernels() { return kernels_; }
    std::vector<size_t> getStrides() { return strides_; }
    void destroy() {}
    void autoDiff(IRGraph *graph, IRNode *opNode,
                  std::unordered_map<IRNode *, IRNode *> &gradNodeMap) override;
};
class AvgPoolGradOp : public Op {
    std::vector<size_t> kernels_;
    std::vector<size_t> strides_;
    std::vector<size_t> pads_;

  public:
    AvgPoolGradOp() : Op(DL_OP, 3, 1, std::string("AveragePoolGrad")) {
        this->_inputNDims.push_back(4); // input
        this->_inputNDims.push_back(4); // output
        this->_inputNDims.push_back(4); // outputG

        this->_outputNDims.push_back(4); // inputG

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("b__c"); // out
        this->_einRep.push_back("b__c"); // outG

        this->_einRep.push_back("b__c"); // inG
    }
    AvgPoolGradOp(std::vector<size_t> &kernels, std::vector<size_t> &strides,
                  std::vector<size_t> &pads)
        : Op(DL_OP, 3, 1, std::string("AveragePoolGrad")) {
        kernels_.assign(kernels.begin(), kernels.end());
        strides_.assign(strides.begin(), strides.end());
        pads_.assign(pads.begin(), pads.end());

        this->_inputNDims.push_back(4); // input
        this->_inputNDims.push_back(4); // output
        this->_inputNDims.push_back(4); // outputG

        this->_outputNDims.push_back(4); // inputG

        this->_einOp = 1;
        this->_einRep.push_back("b__c"); // in
        this->_einRep.push_back("b__c"); // out
        this->_einRep.push_back("b__c"); // outG

        this->_einRep.push_back("b__c"); // inG
    }
    ~AvgPoolGradOp();
    std::vector<size_t> getPads() { return pads_; }
    std::vector<size_t> getKernels() { return kernels_; }
    std::vector<size_t> getStrides() { return strides_; }
    void destroy() {}
};
/*
class BatchedAddOp : public Op {
public:
    BatchedAddOp() : Op(DL_OP, 2, 1, std::string("BatchedAdd")) {
        this->_inputNDims.push_back(2);
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(2);
    }
    ~BatchedAddOp();
    void destroy() {}
};

BatchedAddOp should be MatrixVectorAddOp, remove
*/

class BatchedReduceAddOp : public Op {
  public:
    BatchedReduceAddOp() : Op(DL_OP, 1, 1, std::string("BatchedReduceAdd")) {
        this->_inputNDims.push_back(2);
        this->_outputNDims.push_back(1);

        this->_einOp = 1;
        this->_einRep.push_back("ij");
        this->_einRep.push_back("j");
    }
    ~BatchedReduceAddOp();
    void destroy() {}
};

class TransposeOp : public Op {
    std::vector<size_t> shuffle_;

  public:
    TransposeOp(const std::initializer_list<size_t> &shuffle)
        : Op(DL_OP, 1, 1, std::string("Transpose")) {
        this->_inputNDims.push_back(4);
        this->_outputNDims.push_back(4);
        for (auto i : shuffle)
            shuffle_.push_back(i);
        this->_einOp = 0;
    }
    ~TransposeOp();
    std::vector<size_t> getShuffle() { return shuffle_; }
    void destroy() {}
};
// this may be a tensor op;

class ArgMaxOp : public Op {
    int topK_;

  public:
    ArgMaxOp(int topK) : Op(DL_OP, 1, 1, std::string("ArgMax")) {
        topK_ = topK;
        this->_einOp = 1;
        // currently, parallel pass can only support float type tensor
        // if lower this, int label will cause error
        this->_einRep.push_back("i_");
        this->_einRep.push_back("i_");
    }
    ~ArgMaxOp() {}
    int getTopK() { return topK_; }
    void destroy() {}
};

// currently we let AccuracyOp link to ArgMaxOp.out and label
// then we get topk accuracy
// TODO: let accuray link to prob and label
class AccuracyOp : public Op {

  public:
    AccuracyOp() : Op(DL_OP, 2, 1, std::string("Accuracy")) {
        this->_einOp = 1;
        this->_einRep.push_back("i_");
        this->_einRep.push_back("i_");
        this->_einRep.push_back("_"); // cnt right_cnt
    }
    ~AccuracyOp() {}
    void destroy() {}
};

/**
 *  \brief currently deubg means print 2D Tensor
 */
class DebugOp : public Op {
  public:
    DebugOp() : Op(DL_OP, 1, 1, std::string("Debug")) {
        this->_einOp = 1;
        this->_einRep.push_back("__");
        this->_einRep.push_back("__");
    }
    ~DebugOp() {}
    void destroy() {}
};

//=====================================================
// Definition of 1-D deep learning specific operations.
// Version v0.1: ops for simple-MLP-nobias-fw listed below
//--Activation:
//----VectorTanh
//--VectorSoftmax
//--Loss:
//----VectorLogNegLoss
//=====================================================

class VectorTanhOp : public Op {
  public:
    VectorTanhOp() : Op(DL_OP, 1, 1, std::string("VectorTanh")) {
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(1);
    };
    ~VectorTanhOp();
    void destroy(){};
};

class VectorSoftmaxOp : public Op {
  public:
    VectorSoftmaxOp() : Op(DL_OP, 1, 1, std::string("VectorSoftmax")) {
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(1);
    };
    ~VectorSoftmaxOp();
    void destroy(){};
};

class VectorLogNegLossOp : public Op {
  public:
    VectorLogNegLossOp() : Op(DL_OP, 1, 1, std::string("VectorLogNegLoss")) {
        this->_inputNDims.push_back(1);
        this->_outputNDims.push_back(0);
    };
    ~VectorLogNegLossOp();
    void destroy(){};
};

//=====================================================
// Definition of 0-D deep learning specific operations.
// Version v0.1: ops for simple-MLP-nobias-fw listed below
//--Activation:
//----ScalarTanh
//=====================================================

class ScalarTanhOp : public Op {
  public:
    ScalarTanhOp() : Op(DL_OP, 1, 1, std::string("ScalarTanh")) {
        this->_inputNDims.push_back(0);
        this->_outputNDims.push_back(0);
    };
    ~ScalarTanhOp();
    void destroy(){};
};
} // namespace op
} // namespace swc

#endif
