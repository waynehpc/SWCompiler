/*
 * IRGraph.cpp
 * Copyright © 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * @AUTHOR:      Hongkun Yu
 * @MAIL:        staryhk@gmail.com
 * @VERSION:     2018-12-04
 */
#include "IRGraph.h"

#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
#include "op/basicOp/basicOps.h"
#include "op/dlOp/dlOp.h"
#include "parallel/TilingLabel.h"

#include "common.h"
#include <cassert>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace swc {

IRGraph::~IRGraph() {
    // std::cout << "dtor of IRGraph\n";
    // not do clear of vector
    for (auto &tnode : _tensors)
        tnode->destroy();
    for (auto &onode : _ops)
        onode->destroy();
}

IRNode *IRGraph::getNodeByName(std::string name) const {
    for (auto &node : _tensors)
        if (node->name() == name)
            return node;

    for (auto &node : _ops)
        if (node->name() == name)
            return node;
    return nullptr;
}

bool IRGraph::buildSubGraphs(TensorNode *in, TensorNode *out,
                             ParallelStrategy strategy, int axis, int num) {
    assert(strategy == ParallelStrategy::SLICE && "only support SLICE ");
    assert(axis == 0 && "only herizonnal SLICE ");

    auto inDims = in->getDims();
    size_t dimPerSub = inDims[/*axis*/ 0] / num;
    assert((inDims[0] % dimPerSub) == 0 && "");

    OpNode *subGNode = extractSubGraph(in, out);
    if (!subGNode)
        return false;

    auto subInDims = inDims;
    subInDims[axis] = dimPerSub;
    std::vector<size_t> shape;
    for (auto dim : subInDims)
        shape.push_back(dim);

    IRGraph *subG = ((SubGraphOp *)subGNode->getOp())->getGraph();
    auto *inNodeOfSubG = (TensorNode *)subG->getNodeByName(in->name() + "_sub");
    if (!inNodeOfSubG)
        return false;
    inNodeOfSubG->setTensor(new Tensor(shape));
    subG->initTensorNodes();

    for (int i = 1; i < num; i++) {
        // TensorNode reference to the same Tensor
        // OpNode reference to the same Op
        auto *subG_cp = subG->clone();
        inNodeOfSubG =
            (TensorNode *)subG_cp->getNodeByName(in->name() + "_sub");
        if (!inNodeOfSubG)
            return false;

        inNodeOfSubG->setTensor(new Tensor(shape));
        subG_cp->initTensorNodes();

        auto *subG_Op = new SubGraphOp();
        subG_Op->setGraph(subG_cp);
        auto *subGNode_cp = new OpNode("subG", subG_Op);

        for (auto &p : subGNode->getParentNodes())
            subGNode_cp->exlinkUpperNode(p);
        for (auto &c : subGNode->getChildNodes())
            c->exlinkUpperNode(subGNode_cp);

        this->pushOpNode(subGNode_cp);
    }
    return true;
}

OpNode *IRGraph::extractSubGraph(TensorNode *in, TensorNode *out) {

    SWLOG_DEBUG(4) << "extract SubGraph from " << in->name() << " to "
                   << out->name() << "\n";

    std::unordered_set<IRNode *> found;
    std::queue<IRNode *> toVisit;
    found.insert(out);
    toVisit.push(out);

    while (!toVisit.empty()) {
        auto *cur = toVisit.front();
        toVisit.pop();

        if (cur == in)
            continue;

        for (auto child : cur->getParentNodes()) {
            if (!found.count(child)) {
                toVisit.push(child);
                found.insert(child);
            }
        }
    }

    if (!found.count(in)) {
        return nullptr;
    }

    IRGraph *subG = new IRGraph();
    SubGraphOp *subG_Op = new SubGraphOp();
    subG_Op->setGraph(subG);
    OpNode *subGNode = new OpNode("subG", subG_Op);

    for (auto irNode : found) {
        SWLOG_DEBUG(4) << "process node " << irNode->name() << "\n";
        if (irNode->nodeType() == OP_NODE) {
            auto *node = (OpNode *)irNode;
            subG->pushOpNode(node);
            this->delOpNode(node);
        } else if (irNode->nodeType() == TENSOR_NODE) {
            auto *node = (TensorNode *)irNode;
            if (node == in) {

                TensorNode *node_mirror = node->clone();
                node_mirror->setExternal(true);

                OpNode *scatter = new OpNode("scatter", new ScatterOp());
                scatter->exlinkUpperNode(node_mirror);

                TensorNode *node_sub = new TensorNode(
                    node->name() + "_sub",
                    new Tensor(node->getTensor()->getType()), scatter);
                node->replaceUseKeepOrder(node_sub);

                subG->pushTensorNode(node_mirror, node_sub);
                subG->pushOpNode(scatter);

                continue;
            }
            if (node == out) {
                // suppose TensorNode only have one ParentNode
                assert(node->parentNum() == 1 && "");

                TensorNode *node_sub =
                    new TensorNode(node->name() + "_sub",
                                   new Tensor(node->getTensor()->getType()));
                node_sub->exlinkUpperNode(node->getParentNode(0));

                TensorNode *node_mirror = node->clone();
                node_mirror->setExternal(true);
                OpNode *gather = new OpNode("gather", new GatherOp());
                gather->exlinkUpperNode(node_sub);
                node_mirror->exlinkUpperNode(gather);

                subG->pushTensorNode(node_mirror, node_sub);
                subG->pushOpNode(gather);
                continue;
            }
            if (node->parentNum() == 0) {
                // parameter of Op. e.g. weight and bias of FC;
                // filter and bias of Conv

                TensorNode *node_mirror = node->clone();
                node_mirror->setExternal(true);
                OpNode *scatter = new OpNode("scatter", new ScatterOp());
                scatter->setRunOnce();
                scatter->exlinkUpperNode(node_mirror);

                TensorNode *node_sub = new TensorNode(
                    node->name() + "_sub",
                    new Tensor(node->getTensor()->getType()), scatter);
                node->replaceUseKeepOrder(node_sub);

                subG->pushTensorNode(node_mirror, node_sub);
                subG->pushOpNode(scatter);
                subGNode->exlinkUpperNode(node);

                continue;
            }

            subG->pushTensorNode(node);
            this->delTensorNode(node);
        } // TENSOR_NODE
    }     // for irNode : found

    for (auto c : in->getChildNodes()) {
        if (found.count(c))
            c->destroyUpperNode(in);
    }

    for (auto p : out->getParentNodes()) {
        SWLOG_DEBUG(4) << "destroy " << out->name() << "->" << p->name()
                       << "\n";
        if (found.count(p))
            out->destroyUpperNode(p);
    }
    subGNode->exlinkUpperNode(in);
    out->exlinkUpperNode(subGNode);

    this->pushOpNode(subGNode);
    SWLOG_DEBUG(4) << "extract subGraph successfully\n";

    return subGNode;
}

//---------------------------------------------------------
std::vector<size_t> inferConvOutDims(size_t ih, size_t iw,
                                     std::vector<size_t> &kernels,
                                     std::vector<size_t> &strides,
                                     std::vector<size_t> &pads) {
    assert(kernels.size() == 2);
    assert(strides.size() == 2);
    assert(pads.size() == 4);

    size_t oh = ((ih + pads[0] + pads[2] - kernels[0]) / strides[0] + 1);
    size_t ow = ((iw + pads[1] + pads[3] - kernels[1]) / strides[1] + 1);
    return {oh, ow};
}

void IRGraph::initTensorNodes() {
    updateTopology();

    for (int i = 0; i < topologyNum(); i++) {
        for (int j = 0; j < getNumInTopoLevel(i); j++) {
            auto *irNode = getNodeInTopo(i, j);
            if (irNode->nodeType() == OP_NODE) {
                auto *node = (OpNode *)irNode;
                auto *op = node->getOp();

                SWLOG_DEBUG(1)
                    << "init tensornodes of op " << node->name() << " "
                    << "parent " << node->parentNum() << " child "
                    << node->childNum() << "\n";
                std::cout << op->getOpInfo() << "\n";
                std::cout << ((TensorNode *)node->getParentNode(0))->toString()
                          << "\n";

                if (dynamic_cast<ElementAddOp *>(op) ||
                    dynamic_cast<ElementSubOp *>(op) ||
                    dynamic_cast<ElementMulOp *>(op) ||
                    dynamic_cast<ElementDivOp *>(op)) {
                    auto *in = (TensorNode *)node->getParentNode(0);
                    auto *out = (TensorNode *)node->getChildNode(0);
                    // out->setTensor(
                    //     new Tensor(in->getTensor()->getType()));
                    out->setTensor(new Tensor(in->getTensor()->getType()));

                    op->setAttr(in->getNDim());
                    // switch (in->getNDim()) {
                    //     case 1:
                    //         op->setIONDims({1, 1}, {1});
                    //         op->setEinReps({"a", "a", "a"});
                    //         break;
                    //     case 2:
                    //         op->setIONDims({2, 2}, {2});
                    //         op->setEinReps({"ab", "ab", "ab"});
                    //         break;
                    //     case 4:
                    //         op->setIONDims({4, 4}, {4});
                    //         op->setEinReps({"abcd", "abcd", "abcd"});
                    //         break;
                    //     default:
                    //         SWLOG_ERROR << "error, unimplemented io idims\n";
                    //         exit(0);
                    //         break;
                    // }
                } else if (dynamic_cast<MatrixMatrixFCOp *>(op) ||
                           dynamic_cast<MatrixMatrixFCBiasOp *>(op) ||
                           dynamic_cast<MatrixMatrixMulOp *>(op)) {
                    auto *input = (TensorNode *)node->getParentNode(0);
                    auto idims =
                        ((TensorNode *)node->getParentNode(0))->getDims();
                    auto *weight = (TensorNode *)node->getParentNode(1);
                    auto wdims = weight->getDims();

                    /*
                     * wrong: this will cause tensor losing properties like
                     * training, initInfo_
                     */
                    // weight->setTensor(new Tensor({idims[1], wdims[1]}));

                    auto dim2 = input->getTensor()->viewAs2D(1);
                    SWLOG_DEBUG(2)
                        << input->name() << " ndims = " << idims.size()
                        << ", view as 2d " << dim2.first << " * " << dim2.second
                        << " to fit MatrixMatrixMulOp\n";
                    SWLOG_DEBUG(2) << node->name() << ", reset weight dim to "
                                   << dim2.second << ", " << wdims[1] << "\n";
                    weight->getTensor()->reset({dim2.second, wdims[1]});

                    auto *out = (TensorNode *)node->getChildNode(0);
                    out->setTensor(new Tensor({idims[0], wdims[1]}));
                } else if (dynamic_cast<MatrixTanhOp *>(op)) {
                    auto idims =
                        ((TensorNode *)node->getParentNode(0))->getDims();
                    auto *out = (TensorNode *)node->getChildNode(0);
                    out->setTensor(new Tensor({idims[0], idims[1]}));
                }

                else if (dynamic_cast<ReluOp *>(op) ||
                         dynamic_cast<LRNOp *>(op) ||
                         dynamic_cast<BatchNorm2dOp *>(op)) {

                    auto *in = (TensorNode *)node->getParentNode(0);
                    auto *out = (TensorNode *)node->getChildNode(0);
                    // out->setTensor(
                    //     new Tensor(in->getTensor()->getType()));
                    out->setTensor(new Tensor(in->getTensor()->getType()));
                } else if (dynamic_cast<DropoutOp *>(op)) {

                    auto *in = (TensorNode *)node->getParentNode(0);
                    auto *mask = (TensorNode *)node->getParentNode(1);
                    auto *out = (TensorNode *)node->getChildNode(0);

                    // mask->setTensor(
                    //     new Tensor(in->getTensor()->getType()));
                    // out->setTensor(
                    //     new Tensor(in->getTensor()->getType()));
                    mask->setTensor(new Tensor(in->getTensor()->getType()));
                    out->setTensor(new Tensor(in->getTensor()->getType()));
                }

                else if (dynamic_cast<MatrixSoftmaxOp *>(op)) {
                    auto idims =
                        ((TensorNode *)node->getParentNode(0))->getDims();
                    auto *out = (TensorNode *)node->getChildNode(0);
                    out->setTensor(new Tensor({idims[0], idims[1]}));
                }

                else if (dynamic_cast<MatrixSoftmaxWithLossOp *>(op)) {
                    auto idims =
                        ((TensorNode *)node->getParentNode(0))->getDims();
                    auto *prob = (TensorNode *)node->getChildNode(0);
                    auto *loss = (TensorNode *)node->getChildNode(1);
                    prob->setTensor(new Tensor({idims[0], idims[1]}));
                    loss->setTensor(new Tensor({1}));
                }

                else if (dynamic_cast<ScatterOp *>(op)) {
                    // child reinit
                    auto *out = (TensorNode *)node->getChildNode(0);
                    // auto odims = out->getDims();
                    // auto *shape = out->getTensor()->getType();
                    // out->setTensor(new Tensor(shape));
                    out->setTensor(new Tensor(out->getTensor()->getType()));
                } else if (auto *conv = dynamic_cast<Conv2dOp *>(op)) {
                    auto idims =
                        ((TensorNode *)node->getParentNode(0))->getDims();

                    auto kernels = conv->getKernels();
                    auto strides = conv->getStrides();
                    auto pads = conv->getPads();

                    auto *w = (TensorNode *)node->getParentNode(1);
                    auto *b = (TensorNode *)node->getParentNode(2);
                    auto co = w->getTensor()->getDim(0);

                    w->reset({co, kernels[0], kernels[1], idims[3]});
                    b->reset({co});
                    auto wdims =
                        ((TensorNode *)node->getParentNode(1))->getDims();

                    std::vector<size_t> ohw = inferConvOutDims(
                        idims[1], idims[2], kernels, strides, pads);

                    auto *out = (TensorNode *)node->getChildNode(0);
                    // out->setTensor(new Tensor({idims[0], idims[1]}));
                    out->setTensor(
                        new Tensor({idims[0], ohw[0], ohw[1], wdims[0]}));
                } else if (auto *conv =
                               dynamic_cast<Conv2dWithActivationOp *>(op)) {
                    auto idims =
                        ((TensorNode *)node->getParentNode(0))->getDims();
                    auto wdims = ((TensorNode *)node->getParentNode(1))
                                     ->getDims(); // OC K K IC
                    auto kernels = conv->getKernels();
                    auto strides = conv->getStrides();
                    auto pads = conv->getPads();
                    std::vector<size_t> ohw = inferConvOutDims(
                        idims[1], idims[2], kernels, strides, pads);

                    auto *out = (TensorNode *)node->getChildNode(0);
                    // out->setTensor(new Tensor({idims[0], idims[1]}));
                    out->setTensor(
                        new Tensor({idims[0], ohw[0], ohw[1], wdims[0]}));
                } else if (auto *pool = dynamic_cast<MaxPoolOp *>(op)) {
                    auto idims =
                        ((TensorNode *)node->getParentNode(0))->getDims();
                    auto kernels = pool->getKernels();
                    auto strides = pool->getStrides();
                    auto pads = pool->getPads();
                    std::vector<size_t> ohw = inferConvOutDims(
                        idims[1], idims[2], kernels, strides, pads);

                    auto *out = (TensorNode *)node->getChildNode(0);
                    // out->setTensor(new Tensor({idims[0], idims[1]}));
                    out->setTensor(
                        new Tensor({idims[0], ohw[0], ohw[1], idims[3]}));
                } else if (auto *pool = dynamic_cast<AvgPoolOp *>(op)) {
                    auto idims =
                        ((TensorNode *)node->getParentNode(0))->getDims();
                    auto kernels = pool->getKernels();
                    auto strides = pool->getStrides();
                    auto pads = pool->getPads();
                    std::vector<size_t> ohw = inferConvOutDims(
                        idims[1], idims[2], kernels, strides, pads);

                    auto *out = (TensorNode *)node->getChildNode(0);
                    // out->setTensor(new Tensor({idims[0], idims[1]}));
                    out->setTensor(
                        new Tensor({idims[0], ohw[0], ohw[1], idims[3]}));
                }

                else {
                    SWLOG_ERROR << "unsupported Op in initTensorNodes\n";
                    exit(0);
                }

                // std::cout << ((TensorNode*)node->getChildNode(0))->toString()
                // << "\n";
            }
        }
    }
}

void IRGraph::findInOut() {
    _inNodes.clear();
    _outNodes.clear();

    /*
    typename std::vector<TensorNode *>::iterator tnIter;

    for (tnIter = _tensors.begin(); tnIter != _tensors.end(); tnIter++) {
        if ((*tnIter)->parentNum() == 0)
            _inNodes.push_back(*tnIter);
    }
    for (tnIter = _tensors.begin(); tnIter != _tensors.end(); tnIter++) {
        if ((*tnIter)->childNum() == 0)
            _outNodes.push_back(*tnIter);
    }
    */
    // _inNodes could not be op
    for (auto &tnode : _tensors) {
        if (tnode->parentNum() == 0)
            _inNodes.push_back(tnode);
    }

    for (auto &tnode : _tensors) {
        if (tnode->childNum() == 0)
            _outNodes.push_back(tnode);
    }
    for (auto &opnode : _ops) {
        if (opnode->childNum() == 0)
            _outNodes.push_back(opnode);
    }

    SWLOG_DEBUG(4) << "findInOut innodes:" << _inNodes.size()
                   << " outnodes:" << _outNodes.size() << "\n";
    /*
    std::cout << "_inNodes\n";
    for(auto node : _inNodes)
        std::cout << node->name() << "\n";
    std::cout << "_outNodes\n";
    for(auto node : _outNodes)
        std::cout << node->name() << "\n";
    */
    // OutMark should be decied by other rules but not simple topology out
    // setOutMark();
}

template <typename T> void IRGraph::updateTopology(T node) {
    int currentTopoId = node->topologyId();
    /*std::cout << "Current Node: " << node->name()
      << " TopologyID: " << node->topologyId()
      << std::endl;*/
    for (int i = 0; i < node->childNum(); i++) {
        if (node->getChildNode(i)->topologyId() < currentTopoId + 1) {
            node->getChildNode(i)->setTopologyId(currentTopoId + 1);
            /*std::cout << "Update " << node->getChildNode(i)->name()
              << " TopologyID " << node->getChildNode(i)->topologyId()
              << " To " << currentTopoId + 1
              << std::endl;*/
            updateTopology(node->getChildNode(i));
        }
    }
}

void IRGraph::updateTopology() {
    /*
    typename std::vector<TensorNode *>::iterator tnIter;
    typename std::vector<OpNode *>::iterator opIter;

    for (tnIter = _tensors.begin(); tnIter != _tensors.end(); tnIter++)
        (*tnIter)->setTopologyId(0);
    for (opIter = _ops.begin(); opIter != _ops.end(); opIter++)
        (*opIter)->setTopologyId(0);

    for (tnIter = _inNodes.begin(); tnIter != _inNodes.end(); tnIter++)
        updateTopology(*tnIter);
    */

    for (auto &tnode : _tensors)
        tnode->setTopologyId(0);
    for (auto &opnode : _ops)
        opnode->setTopologyId(0);

    for (auto &node : _inNodes)
        updateTopology(node);

    updateTopoNodeList();
}

void IRGraph::updateTopoNodeList() {
    typename std::vector<TensorNode *>::iterator tnIter;
    typename std::vector<OpNode *>::iterator opIter;
    std::vector<std::vector<IRNode *>>::iterator ndIter;
    int topoId;
    std::vector<IRNode *> vtemp;

    for (ndIter = _nodesByTopology.begin(); ndIter != _nodesByTopology.end();
         ndIter++) {
        ndIter->clear();
    }
    _nodesByTopology.clear();

    for (tnIter = _tensors.begin(); tnIter != _tensors.end(); tnIter++) {
        topoId = (*tnIter)->topologyId();
        while (topoId >= (int)_nodesByTopology.size()) {
            _nodesByTopology.push_back(vtemp);
        }
        _nodesByTopology[topoId].push_back(*tnIter);
    }
    for (opIter = _ops.begin(); opIter != _ops.end(); opIter++) {
        topoId = (*opIter)->topologyId();
        while (topoId >= (int)_nodesByTopology.size()) {
            _nodesByTopology.push_back(vtemp);
        }
        _nodesByTopology[topoId].push_back(*opIter);
    }
}

void IRGraph::copyTo(IRGraph *graph) const {

    std::unordered_map<TensorNode *, TensorNode *> tensors_map;
    std::unordered_map<OpNode *, OpNode *> ops_map;

    for (auto &N : _tensors) {
        TensorNode *tn = (N->isExternal()) ? N->clone() : N->deepClone();
        tensors_map[N] = tn;
        graph->pushTensorNode(tn);
    }
    for (auto &N : _ops) {
        OpNode *opn = N->clone();
        ops_map[N] = opn;
        graph->pushOpNode(opn);
    }

    for (auto &N : _tensors) {
        auto tn = tensors_map[N];
        for (int i = 0; i < N->parentNum(); i++) {
            auto parent = ops_map[(OpNode *)N->getParentNode(i)];
            tn->exlinkUpperNode(parent);
        }
    }
    for (auto &N : _ops) {
        auto opn = ops_map[N];
        for (int i = 0; i < N->parentNum(); i++) {
            auto parent = tensors_map[(TensorNode *)N->getParentNode(i)];
            opn->exlinkUpperNode(parent);
        }
    }

    // graph->setConfig(_config);

    graph->setDeviceLabel(_dev);
    graph->findInOut();
    graph->updateTopology();
}

// 2019.10.1 modify clone as deepclone
IRGraph *IRGraph::clone() const {
    IRGraph *graph = new IRGraph();

    SWLOG_DEBUG(1) << "IRGraph::clone() clone tensornodes and opnodes\n";
    std::unordered_map<TensorNode *, TensorNode *> tensors_map;
    std::unordered_map<OpNode *, OpNode *> ops_map;
    for (auto &N : _tensors) {
        TensorNode *tn = N->deepClone();
        tensors_map[N] = tn;
        graph->pushTensorNode(tn);
    }
    for (auto &N : _ops) {
        OpNode *opn = N->clone();
        ops_map[N] = opn;
        graph->pushOpNode(opn);
    }

    SWLOG_DEBUG(1) << "IRGraph::clone() link tensornodes and opnodes\n";
    // it worked, but remind that
    // static_cast may cause offset
    for (auto &N : _tensors) {
        auto tn = tensors_map[N];
        for (int i = 0; i < N->parentNum(); i++) {
            auto parent = ops_map[(OpNode *)N->getParentNode(i)];
            tn->exlinkUpperNode(parent);
        }
    }
    SWLOG_DEBUG(1) << "IRGraph::clone() link ops to parent\n";
    for (auto &N : _ops) {
        auto opn = ops_map[N];
        for (int i = 0; i < N->parentNum(); i++) {
            SWLOG_DEBUG(1) << "IRGraph::clone() op->parentNum()= "
                           << N->parentNum() << " p" << i << ": "
                           << N->getParentNode(i)->name() << "\n";
            auto parent = tensors_map[(TensorNode *)N->getParentNode(i)];
            opn->exlinkUpperNode(parent);
        }
    }

    SWLOG_DEBUG(1) << "IRGraph::clone() addLogicalOutNodes\n";
    // clone _logicalOutNodes
    for (auto &node : _logicalOutNodes) {
        if (node->nodeType() == TENSOR_NODE) {
            graph->addLogicalOutNodes(tensors_map.at((TensorNode *)node));
        } else
            graph->addLogicalOutNodes(ops_map.at((OpNode *)node));
    }
    // clone _input_data_node and _input_label_node
    if (_input_data_node && _input_data_node)
        graph->setTrainDataNodes(tensors_map.at(_input_label_node),
                                 tensors_map.at(_input_data_node));

    // clone _dispaly_nodes
    for (auto &node : _display_nodes) {
        graph->addDisplayTensorNodes(node);
    }

    // clone _config
    graph->setConfig(_config);

    // clone _dev
    graph->setDeviceLabel(_dev);

    SWLOG_DEBUG(4) << "updateTopology of deepcloned graph\n";
    // updateTopoly
    graph->findInOut();
    graph->updateTopology();

    graph->resetParallelStrategy();

    return graph;
}

void IRGraph::setDeviceLabel(Device dev) {
    SWLOG_DEBUG(4) << "set Graph Device Label (Skip external node)\n";
    _dev = dev;
    for (auto tnode : _tensors) {
        // suppose Device Graph, all TensorNodes(in degree=0)
        // should be mirror of cpu TensorNodes
        if (tnode->isExternal()) {
            SWLOG_DEBUG(4) << tnode->name()
                           << " isExternal=" << tnode->isExternal()
                           << " skip\n";
        }

        if (!tnode->isExternal())
            tnode->getLabel()->setDeviceLabel(dev.rank, dev.type, dev.id);
    }
    for (auto opnode : _ops) {
        opnode->getLabel()->setDeviceLabel(dev.rank, dev.type, dev.id);
    }
}

void IRGraph::setOutMark() {
    for (unsigned int i = 0; i < _outNodes.size(); i++) {
        _outNodes[i]->getLabel()->setIsOut();
        SWLOG_DEBUG(4) << "set out mark for " << _outNodes[i]->name() << "\n";
    }
}

// if remove node from _outNodes, we need to clear its mark
void IRGraph::clearOutMark() {
    for (auto out : _outNodes) {
        out->getLabel()->setIsOut(0);
    }
}

void IRGraph::setLogicalOutMark() {
    for (unsigned int i = 0; i < _logicalOutNodes.size(); i++) {
        _logicalOutNodes[i]->getLabel()->setIsOut();
        SWLOG_DEBUG(4) << "set out mark for " << _logicalOutNodes[i]->name()
                       << "\n";
    }
}

size_t IRGraph::getCommCost() {
    size_t cost = 0;
    // _ops should keep with _nodesByTopology
    for (auto opnode : _ops) {
        // split these comm ops, because code may be different
        // in the future
        if (dynamic_cast<ScatterOp *>(opnode->getOp())) {
            cost += opnode->getCost(_config);
        } else if (dynamic_cast<GatherOp *>(opnode->getOp())) {
            cost += opnode->getCost(_config);
        } else if (dynamic_cast<ReduceOp *>(opnode->getOp())) {
            cost += opnode->getCost(_config);
        } else if (dynamic_cast<TransformOp *>(opnode->getOp())) {
            cost += opnode->getCost(_config);
        }
    }

    return cost;
}

std::string IRGraph::getCommTrace() {
    std::string trace;
    for (auto opnode : _ops) {
        // split these comm ops, because code may be different
        // in the future
        if (dynamic_cast<ScatterOp *>(opnode->getOp()) ||
            dynamic_cast<GatherOp *>(opnode->getOp()) ||
            dynamic_cast<ReduceOp *>(opnode->getOp()) ||
            dynamic_cast<TransformOp *>(opnode->getOp())) {

            trace += opnode->getCostTrace(_config);
        }
    }
    return trace;
}

void IRGraph::resetParallelStrategy() {
    for (auto &tnode : _tensors) {
        tnode->setTilingLabel(new TilingLabel());
    }
    for (auto &onode : _ops) {
        onode->setStrategyLabel(nullptr);
    }
}

void IRGraph::elimRedundantScatter() {
    for (auto &opnode : _ops) {
        auto *scatter = dynamic_cast<ScatterOp *>(opnode->getOp());
        if (!scatter) {
            continue;
        }
        /*
        if(scatter->getAxis() == -1)
            continue;
        */

        // parent tensornode
        auto *ptnode = (TensorNode *)opnode->getParentNode(0);
        // childnode
        auto *ctnode = (TensorNode *)opnode->getChildNode(0);

        // this means ptnode is an topoInTensornode, and this scatter is its
        // only child
        if (ptnode->parentNum() == 0 && ptnode->childNum() == 1) {
            SWLOG_DEBUG(6) << opnode->name() << " and its parent "
                           << ptnode->name()
                           << " is redundant during batch iterations\n";
            ctnode->destroyUpperNode(opnode);
            // !!! do not delOpNode because we are iterate on _ops
            // this->delOpNode(opnode);
            // this->delTensorNode(ptnode);
        }

    } // for loop
}

void IRGraph::setOpDevLabelByInput() {
    for (auto &opnode : _ops) {
        auto *op = opnode->getOp();
        if (dynamic_cast<ScatterOp *>(op) || dynamic_cast<GatherOp *>(op) ||
            dynamic_cast<TransformOp *>(op) ||
            dynamic_cast<BroadcastOp *>(op) || dynamic_cast<ReduceOp *>(op)) {
            continue;
        }

        auto *input0 = (TensorNode *)opnode->getChildNode(0);
        Device dev = input0->getLabel()->getDeviceLabel();
        opnode->getLabel()->setDeviceLabel(dev);

        SWLOG_DEBUG(1) << opnode->name() << " " << dev.toString() << "\n";
    }
}

} // namespace swc
