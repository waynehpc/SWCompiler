/*************************************************************************
	> File Name: dlOp.cpp
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: 二 12/ 4 15:57:35 2018
 ************************************************************************/

#include "dlOp.h"

#include <cassert>

#include "SWDSL.h"
#include "graphIR/IRGraph.h"
#include "graphIR/IRNode.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"

using namespace swc::op;

/*--------------------------------Auto Diff ------------------------
 *
 *
 *
 * ----------------------------------------------------------------*/
void MatrixMatrixFCBiasOp::autoDiff(
    IRGraph *graph, IRNode *opNode,
    std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *weight = opNode->getParentNode(1);
    auto *bias = opNode->getParentNode(2);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) && "grad of FC output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    // SWLOG_DEBUG(6) << opNode->name() << " weight " <<
    // ((TensorNode*)weight)->getTraining() << "\n"; SWLOG_DEBUG(6) <<
    // opNode->name() << " bias " << ((TensorNode*)bias)->getTraining() << "\n";

    auto *N =
        new OpNode(opNode->name() + "_grad", new MatrixMatrixFCBiasGradOp());

    // in current implementation, redundant link to output
    // N->exlinkUpperNode(input, weight, bias, output, outputGrad);
    N->exlinkUpperNode(input, weight, bias, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
}
void MatrixMatrixFCOp::autoDiff(
    IRGraph *graph, IRNode *opNode,
    std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *weight = opNode->getParentNode(1);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) && "grad of FC output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N = new OpNode(opNode->name() + "_grad", new MatrixMatrixFCGradOp());
    // N->exlinkUpperNode(input, weight, output, outputGrad);
    N->exlinkUpperNode(input, weight, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
}

void ReluOp::autoDiff(IRGraph *graph, IRNode *opNode,
                      std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) && "grad of Relu output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N = new OpNode(opNode->name() + "_grad", new ReluGradOp());
    N->exlinkUpperNode(input, output, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
}

void MatrixTanhOp::autoDiff(
    IRGraph *graph, IRNode *opNode,
    std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) && "grad of Tanh output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N = new OpNode(opNode->name() + "_grad", new MatrixTanhGradOp());
    N->exlinkUpperNode(input, output, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
}

void MaxPoolOp::autoDiff(IRGraph *graph, IRNode *opNode,
                         std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *output = opNode->getChildNode(0);

    auto *pool_op = (MaxPoolOp *)((OpNode *)opNode)->getOp();
    auto kernels = pool_op->getKernels();
    auto strides = pool_op->getStrides();
    auto pads = pool_op->getPads();

    assert(gradNodeMap.count(output) && "grad of MaxPool output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N = new OpNode(opNode->name() + "_grad",
                         new MaxPoolGradOp(kernels, strides, pads));
    N->exlinkUpperNode(input, output, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
}

void AvgPoolOp::autoDiff(IRGraph *graph, IRNode *opNode,
                         std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *output = opNode->getChildNode(0);

    auto *pool_op = (AvgPoolOp *)((OpNode *)opNode)->getOp();
    auto kernels = pool_op->getKernels();
    auto strides = pool_op->getStrides();
    auto pads = pool_op->getPads();

    assert(gradNodeMap.count(output) && "grad of AvgPool output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N = new OpNode(opNode->name() + "_grad",
                         new AvgPoolGradOp(kernels, strides, pads));
    N->exlinkUpperNode(input, output, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
}

void MatrixSoftmaxOp::autoDiff(
    IRGraph *graph, IRNode *opNode,
    std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *label = opNode->getParentNode(1);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) && "grad of Softmax output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N = new OpNode(opNode->name() + "_grad", new MatrixSoftmaxGradOp());
    N->exlinkUpperNode(input, label, output, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
}

void MatrixSoftmaxWithLossOp::autoDiff(
    IRGraph *graph, IRNode *opNode,
    std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName << std::endl;
    // auto *input = opNode->getParentNode(0);
    auto *label = opNode->getParentNode(1);
    auto *prob = opNode->getChildNode(0);
    // auto *loss = opNode->getChildNode(1);
    assert(gradNodeMap.count(prob) && "grad of Softmax output unfound\n");
    // auto *outputGrad = gradNodeMap[prob];

    auto *N =
        new OpNode(opNode->name() + "_grad", new MatrixSoftmaxWithLossGradOp());
    // N->exlinkUpperNode(input, label, prob, loss, outputGrad);
    // we do not need input and loss for grad computation
    // actually outputGrad == loss because loss is the start of autoDiff
    N->exlinkUpperNode(label, prob);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
}

void Conv2dOp::autoDiff(IRGraph *graph, IRNode *opNode,
                        std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *weight = opNode->getParentNode(1);
    // auto *bias = opNode->getParentNode(2);
    auto *output = opNode->getChildNode(0);

    auto *conv_op = (Conv2dOp *)((OpNode *)opNode)->getOp();
    auto kernels = conv_op->getKernels();
    auto strides = conv_op->getStrides();
    auto pads = conv_op->getPads();

    assert(gradNodeMap.count(output) && "grad of Conv2d output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N = new OpNode(opNode->name() + "_grad",
                         new Conv2dGradOp(kernels, strides, pads));
    // N->exlinkUpperNode(input, weight, bias, output, outputGrad);
    N->exlinkUpperNode(input, weight, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
}

void Conv2dWithActivationOp::autoDiff(
    IRGraph *graph, IRNode *opNode,
    std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *weight = opNode->getParentNode(1);
    // auto *bias = opNode->getParentNode(2);
    auto *output = opNode->getChildNode(0);

    auto *conv_op = (Conv2dWithActivationOp *)((OpNode *)opNode)->getOp();
    auto kernels = conv_op->getKernels();
    auto strides = conv_op->getStrides();
    auto pads = conv_op->getPads();
    auto activation = conv_op->getActivationType();

    assert(gradNodeMap.count(output) &&
           "grad of Conv2dWithActivation output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N = new OpNode(
        opNode->name() + "_grad",
        new Conv2dWithActivationGradOp(kernels, strides, pads, activation));
    // N->exlinkUpperNode(input, weight, bias, output, outputGrad);
    N->exlinkUpperNode(input, weight, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
}

void BatchNormalizationOp::autoDiff(IRGraph *graph, IRNode *opNode,
                     std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) && "grad of BatchNormalization output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *bn = (BatchNormalizationOp *)((OpNode *)opNode)->getOp();
    auto eps = bn->getEpsilon();

    auto *N = new OpNode(opNode->name() + "_grad", new BNGradOp(eps));
    N->exlinkUpperNode(input, output, outputGrad);
    //     N->exlinkUpperNode(input, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
}

void LRNOp::autoDiff(IRGraph *graph, IRNode *opNode,
                     std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName << std::endl;
    auto *input = opNode->getParentNode(0);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) && "grad of LRN output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N = new OpNode(opNode->name() + "_grad", new LRNGradOp());
    N->exlinkUpperNode(input, output, outputGrad);
    //     N->exlinkUpperNode(input, outputGrad);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
}

void DropoutOp::autoDiff(IRGraph *graph, IRNode *opNode,
                         std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName << std::endl;
    // auto *x = opNode->getParentNode(0);
    auto *mask = opNode->getParentNode(1);
    auto *output = opNode->getChildNode(0);

    assert(gradNodeMap.count(output) && "grad of Dropout output unfound\n");
    auto *outputGrad = gradNodeMap[output];

    auto *N = new OpNode(opNode->name() + "_grad", new ElementMulOp());
    N->exlinkUpperNode(outputGrad, mask);

    gradNodeMap[opNode] = N;
    graph->pushOpNode(N);
}

void ElementAddOp::autoDiff(IRGraph *graph, IRNode *opNode,
                     std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    SWLOG_DEBUG(4) << "autoDiff: " << _opClassName << std::endl;
    auto *lhs = opNode->getParentNode(0);
    auto *rhs = opNode->getParentNode(1);
    auto *output = opNode->getChildNode(0);
    assert(gradNodeMap.count(output) && "grad of ElementAdd output unfound\n");
    auto *outputGrad = gradNodeMap[output];
    
    // TODO fix potential bug when lhs grad already exists
    gradNodeMap[lhs] = outputGrad;
    gradNodeMap[rhs] = outputGrad;
}
