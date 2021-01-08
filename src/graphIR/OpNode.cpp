/*
 * OpNode.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-11-23
 */

#include "OpNode.h"
#include "IRGraph.h"
#include "TensorNode.h"
#include "op/dlOp/dlOp.h"
// for StrategyLabel class
#include "parallel/TilingLabel.h"

namespace swc {
void OpNode::destroy() {
    SWLOG_DEBUG(4) << "Destroy OpNode: " << name() << "\n";
    getOp()->destroy();
    // getLabel()->destroy();
    delete _strategyLabel;
};

/// must clone op_ because destructed in ctor
OpNode *OpNode::clone() const {
    OpNode *opNode = new OpNode(name());
    opNode->setOp(op_);
    return opNode;
}

// we may want to clone scatter node
// but reset ScatterOp::Offset
// ConvOp::kernels...
// so we have to implement clone or copy ctor
// for every dlOp?
// TODO solve this problem
OpNode *OpNode::deepClone() const {
    OpNode *opNode = new OpNode(name());
    opNode->setOp(op_);
    return opNode;
}

std::string OpNode::toString() const {
    std::stringstream os;
    os << "OpNode " << name() << "\n"
       << "  op: " << op_->getOpName() << "\n"
       << "    nInput : " << op_->getnInput() << "\n"
       << "    nOutput: " << op_->getnOutput();
    return os.str();
}

void OpNode::setUpOpGradNode(
    IRGraph *graph, std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    op_->autoDiff(graph, this, gradNodeMap);
}

void OpNode::setUpInputsGradNode(
    IRGraph *graph, std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    if (!gradNodeMap.count(this)) {
        // e.g. optimization: ElementAddOp does not need ElementAddGradOp
        return;
    }

    int parentGradNum = parentNum();

    if (dynamic_cast<op::BatchNorm2dOp *>(op_)) {
        createOrAddInputGrad(graph, 0, gradNodeMap); // input
        createOrAddInputGrad(graph, 3, gradNodeMap); // scale
        createOrAddInputGrad(graph, 4, gradNodeMap); // shift

        return;
    }

    if (dynamic_cast<op::MatrixSoftmaxWithLossOp *>(op_) ||
        dynamic_cast<op::SigmoidCrossEntropyLossOp *>(op_) ||
        dynamic_cast<op::EuclideanLossOp *>(op_) ||
        dynamic_cast<op::DropoutOp *>(op_)) {
        /**
         * 1a. softmaxwithloss in(input, label) out(prob, loss)
         * 1b. softmaxwithlossgrad in(label, prob) out(inputGrad)
         * 2a. dropout in(input, mask) out(output)
         * 2b. dropoutgrad in(outputGrad, mask) out(inputGrad)
         */
        parentGradNum--;
    }

    for (int i = 0; i < parentGradNum /*parentNum()*/; i++) {
        createOrAddInputGrad(graph, i, gradNodeMap);
    }
}

void OpNode::createOrAddInputGrad(
    IRGraph *graph, int inputIdx,
    std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
    auto *tnode = (TensorNode *)(getParentNode(inputIdx));
    // TODO: this is dangerous e.g. user set data->require_grad=false
    // but we always support Conv2dGradOp has 3 outputs, which will cause error
    // if(tnode->require_grad == false) {
    //     continue;
    // }

    auto *tensor = tnode->getTensor();
    auto *N = new TensorNode(tnode->name() + "_grad",
                             new Tensor(tensor->getType()), gradNodeMap[this]);

    SWLOG_DEBUG(4) << "get Grad for " << name() << " input " << tnode->name()
                   << "\n";

    graph->pushTensorNode(N);
    auto iter = gradNodeMap.find(tnode);
    if (iter != gradNodeMap.end()) {
        auto *add = new OpNode("add", new ElementAddOp(),
                               {(TensorNode *)iter->second, N});
        // auto *add_out = new TensorNode(tnode->name() + "_grad",
        //                     new Tensor(tensor->getType()),
        //                     add);
        auto *add_out = new TensorNode(tnode->name() + "_grad",
                                       new Tensor(tensor->getType()), add);
        add->getOp()->setAttr(N->getNDim());

        gradNodeMap[tnode] = add_out;
        graph->pushOpNode(add);
        graph->pushTensorNode(add_out);

    } else {
        gradNodeMap[tnode] = N;
    }
}

} // namespace swc
