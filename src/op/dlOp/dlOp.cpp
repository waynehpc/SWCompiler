/*
 * dlOp.cpp
 * Copyright © 2019 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2019-07-16
 */

#include "dlOp.h"

#include "SWDSL.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
#include "op/tensorOp/tensorOps.h"

#define COP(token, name, method, para...)                                      \
    OpNode *token = new OpNode(name, new method(para));

#define CTENSOR(token, name, shape, parent)                                    \
    TensorNode *token = new TensorNode(name, new Tensor(shape), parent)

void ReluOp::checkValid(OpNode *node) {

    SWLOG_DEBUG(4) << "Checking connect validation for " << node->name()
                   << std::endl;
    assert(node->parentNum() == 1 && "Relu input should be 1: data");

    TensorNode *parent = (TensorNode *)(node->getParentNode(0));

    if (parent->getTensor()->getNDim() != 4) {

        SWLOG_DEBUG(5) << "Customized the relu Op"
                       << " with input and output dimensions:"
                       << parent->getTensor()->getNDim() << std::endl;
        this->_inputNDims[0] = parent->getTensor()->getNDim();
        this->_outputNDims[0] = this->_inputNDims[0];
    }
}

/***************************************
 *
 *      This check will add descend op
 *
 ***************************************/

void MatrixMatrixFCOp::checkValid(OpNode *node) {

    SWLOG_DEBUG(4) << "Checking connect validation for " << node->name()
                   << std::endl;
    assert(node->parentNum() == 2 && "MMFC input should be 2: data and weight");

    TensorNode *data = (TensorNode *)(node->getParentNode(0));
    TensorNode *weight = (TensorNode *)(node->getParentNode(1));

    if (data->getTensor()->getNDim() != 2) {

        auto dim2 = data->getTensor()->viewAs2D(1);

        SWLOG_DEBUG(5) << "Decend the high dimension tensor data"
                       << " from " << data->getTensor()->getNDim() << " to 2 {"
                       << dim2.first << ", " << dim2.second << "}" << std::endl;
    }
    if (weight->getTensor()->getNDim() != 2) {
        std::cout << "FATAL ERROR: the FC weight dimension is not 2"
                  << std::endl;
        abort();
    }
}

/***************************************
 *
 *      This check will add descend op
 *
 ***************************************/
void MatrixMatrixFCBiasOp::checkValid(OpNode *node) {

    SWLOG_DEBUG(4) << "Checking connect validation for " << node->name()
                   << std::endl;
    assert(node->parentNum() == 3 &&
           "MMFCBias input should be 3: data and weight, bias");

    TensorNode *data = (TensorNode *)(node->getParentNode(0));
    TensorNode *weight = (TensorNode *)(node->getParentNode(1));

    if (data->getTensor()->getNDim() != 2) {

        auto dim2 = data->getTensor()->viewAs2D(1);

        SWLOG_DEBUG(5) << "Decend the high dimension tensor data"
                       << " from " << data->getTensor()->getNDim() << " to 2 {"
                       << dim2.first << ", " << dim2.second << "}" << std::endl;
    }
    if (weight->getTensor()->getNDim() != 2) {
        std::cout << "FATAL ERROR: the FC weight dimension is not 2"
                  << std::endl;
        abort();
    }
}

/********************************************
 *
 *      This check will reshape label tensor
 *
 *********************************************/
void MatrixSoftmaxOp::checkValid(OpNode *node) {

    SWLOG_DEBUG(4) << "Checking connect validation for " << node->name()
                   << std::endl;
    assert(node->parentNum() == 1 && "Softmax input should be 1: data");

    TensorNode *data = (TensorNode *)(node->getParentNode(0));

    if (data->getTensor()->getNDim() != 2) {
        std::cout << "FATAL ERROR: the SoftMax data dimension is not 2"
                  << std::endl;
        abort();
    }
}

void MatrixSoftmaxWithLossOp::checkValid(OpNode *node) {

    SWLOG_DEBUG(4) << "Checking connect validation for " << node->name()
                   << std::endl;
    assert(node->parentNum() == 2 &&
           "SoftmaxWithLoss input should be 2: data and label");

    TensorNode *data = (TensorNode *)(node->getParentNode(0));
    TensorNode *label = (TensorNode *)(node->getParentNode(1));

    if (data->getTensor()->getNDim() != 2) {
        std::cout << "FATAL ERROR: the SoftMax data dimension is not 2"
                  << std::endl;
        abort();
    }
    if (label->getTensor()->getNDim() != 2) {
        auto dim0 = data->getTensor()->getDim(0);
        auto dim1 = data->getTensor()->getDim(1);
        SWLOG_DEBUG(5) << "Reshape  label tensor dimension"
                       << " to data input dimension{" << dim0 << ", " << dim1
                       << "}" << std::endl;
        label->getTensor()->reset({dim0, dim1});
    }
}

void Conv2dOp::outTensorTypeGen(OpNode *node, size_t index, Tensor *tensor) {
    std::vector<size_t> idims =
        ((TensorNode *)node->getParentNode(0))->getDims();
    std::vector<size_t> wdims =
        ((TensorNode *)node->getParentNode(1))->getDims();
    std::vector<size_t> kernels = ((Conv2dOp *)node->getOp())->getKernels();
    std::vector<size_t> strides = ((Conv2dOp *)node->getOp())->getStrides();
    std::vector<size_t> pads = ((Conv2dOp *)node->getOp())->getPads();

    assert(kernels.size() == 2);
    assert(strides.size() == 2);
    assert(pads.size() == 4);

    size_t oh = ((idims[1] + pads[0] + pads[2] - kernels[0]) / strides[0] + 1);
    size_t ow = ((idims[2] + pads[1] + pads[3] - kernels[1]) / strides[1] + 1);

    std::vector<size_t> shape;
    shape.push_back(idims[0]);
    shape.push_back(oh);
    shape.push_back(ow);
    shape.push_back(wdims[0]);

    tensor->reset(TensorType(shape));
}

void MaxPoolOp::outTensorTypeGen(OpNode *node, size_t index, Tensor *tensor) {
    std::vector<size_t> idims =
        ((TensorNode *)node->getParentNode(0))->getDims();
    std::vector<size_t> kernels = ((MaxPoolOp *)node->getOp())->getKernels();
    std::vector<size_t> strides = ((MaxPoolOp *)node->getOp())->getStrides();
    std::vector<size_t> pads = ((MaxPoolOp *)node->getOp())->getPads();

    assert(kernels.size() == 2);
    assert(strides.size() == 2);
    assert(pads.size() == 4);

    size_t oh = ((idims[1] + pads[0] + pads[2] - kernels[0]) / strides[0] + 1);
    size_t ow = ((idims[2] + pads[1] + pads[3] - kernels[1]) / strides[1] + 1);

    std::vector<size_t> shape;
    shape.push_back(idims[0]);
    shape.push_back(oh);
    shape.push_back(ow);
    shape.push_back(idims[3]);

    tensor->reset(TensorType(shape));
}

void MatrixMatrixFCOp::outTensorTypeGen(OpNode *node, size_t index,
                                        Tensor *tensor) {
    std::vector<size_t> idims =
        ((TensorNode *)node->getParentNode(0))->getDims();
    std::vector<size_t> wdims =
        ((TensorNode *)node->getParentNode(1))->getDims();

    std::vector<size_t> shape;
    shape.push_back(idims[0]);
    shape.push_back(wdims[1]);

    tensor->reset(TensorType(shape));
}

void MatrixMatrixFCBiasOp::outTensorTypeGen(OpNode *node, size_t index,
                                            Tensor *tensor) {
    std::vector<size_t> idims =
        ((TensorNode *)node->getParentNode(0))->getDims();
    std::vector<size_t> wdims =
        ((TensorNode *)node->getParentNode(1))->getDims();

    std::vector<size_t> shape;
    shape.push_back(idims[0]);
    shape.push_back(wdims[1]);

    tensor->reset(TensorType(shape));
}
