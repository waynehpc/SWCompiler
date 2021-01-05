/*
 * OpNode.h
 * Copyright (C) 2018 Hongkun Yu <staryhk@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef OPNODE_H_
#define OPNODE_H_

#include "IRNode.h"
#include "op/Op.h"
#include <sstream>

using namespace swc::op;

namespace swc {

// Forward declaration
class StrategyLabel;

class OpNode : public IRNode {
  public:
    OpNode() : op_(NULL){};
    explicit OpNode(std::string name) : IRNode(OP_NODE, name) {}
    explicit OpNode(std::string name, Op *op)
        : IRNode(OP_NODE, name), op_(op) {}

    explicit OpNode(std::string name, Op *op,
                    std::initializer_list<IRNode *> parents)
        : IRNode(OP_NODE, name), op_(op) {
        for (auto &it : parents)
            this->exlinkUpperNode(it);
    }

    ~OpNode(){};

    void destroy();

    void setOp(Op *op) { op_ = op; }

    Op *getOp() { return op_; }

    const std::string getOpName() { return op_->getOpName(); }

    OpNode *clone() const;
    OpNode *deepClone() const;
    std::string toString() const;
    void setRunOnce() { run_once_ = true; }
    bool runable() {
        bool run = run_;
        if (run_once_)
            run_ = false;
        return run;
    }

    void autoDiff(IRGraph *graph,
                  std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
        SWLOG_DEBUG(4) << "OpNode " << name() << " begin to autodiff"
                       << std::endl;
        setUpOpGradNode(graph, gradNodeMap);
        setUpInputsGradNode(graph, gradNodeMap);
    }

    void setUpOpGradNode(IRGraph *graph,
                         std::unordered_map<IRNode *, IRNode *> &gradNodeMap);
    void
    setUpInputsGradNode(IRGraph *graph,
                        std::unordered_map<IRNode *, IRNode *> &gradNodeMap);
    void
    createOrAddInputGrad(IRGraph *graph, int inputIdx,
                         std::unordered_map<IRNode *, IRNode *> &gradNodeMap);
    void checkValid() {
        Op *_op = op_;
        _op->checkValid(this);
        return;
    };

    void outTensorTypeGen(size_t index, Tensor *tensor) {
        Op *_op = op_;
        _op->outTensorTypeGen(this, index, tensor);
    };

    void genOutTensor() const;

    void setStrategyLabel(StrategyLabel *strategyLabel) {
        _strategyLabel = strategyLabel;
    }
    StrategyLabel *getStrategyLabel() { return _strategyLabel; }

    size_t getCost(Config &config) { return op_->getCost(this, config); }
    std::string getCostTrace(Config &config) {
        return op_->getCostTrace(this, config);
    }

  private:
    Op *op_;
    bool run_{true};
    bool run_once_{false};

    StrategyLabel *_strategyLabel{NULL};
};

} // namespace swc
#endif /* !OPNODE_H_ */
