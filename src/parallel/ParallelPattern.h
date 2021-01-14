/***********************************************
#
#      Filename: src/parallel/ParallelPattern.cpp
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-07-05 11:04:16
# Last Modified: 2019-07-05 11:04:16
***********************************************/
#ifndef _PARALLELPATTERN_H
#define _PARALLELPATTERN_H
#include "TilingLabel.h"
#include "common.h"
#include "graphIR/IRGraph.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
#include "op/Op.h"
#include "op/dlOp/dlOp.h"
#include "tensor/tensor.h"
#include <climits>

using namespace swc::op;
namespace swc {

class BasePattern {
  protected:
    TensorNode *_tensornode;
    int _num;
    Device _p_dev{INT_MAX, DeviceType::CPU, 0};

  public:
    BasePattern(TensorNode *node, int num) : _tensornode(node), _num(num) {}
    BasePattern(TensorNode *node, const Config &config) : _tensornode(node) {
        if (config.mpi) {
            _num = config.mpi_size;
        } else if (config.sproc_mgpu) {
            _num = config.ngpus_per_rank;
        }
    }

    virtual void apply(int strategy, IRGraph *graph) = 0;
};

class ForkPattern : public BasePattern {
  public:
    ForkPattern(TensorNode *tensornode, int num)
        : BasePattern(tensornode, num) {}
    ForkPattern(TensorNode *tensornode, const Config &config)
        : BasePattern(tensornode, config) {}
    ~ForkPattern() {}

    void apply(int strategy, IRGraph *irgraph) override {
        SWLOG_DEBUG(4) << "ForkPattern on tensor " << _tensornode->name()
                       << ", strategy= " << strategy << "\n";
        TilingLabel *tlabel = _tensornode->getTilingLabel();
        TensorType originType = _tensornode->getTensor()->getType();

        TensorNode *tilenode;
        if (strategy >= 0) {
            tilenode = new TensorNode(
                _tensornode->name() + "_tile",
                new Tensor(originType.getTiledTensorType(strategy, _num)));
        } else if (strategy == -1) {
            // tilenode = new TensorNode(_tensornode->name() + "_replicate",
            //                           new Tensor(originshape, dtype));
            tilenode = new TensorNode(_tensornode->name() + "_replicate",
                                      new Tensor(originType));
        } else
            // tilenode = new TensorNode(_tensornode->name() + "_unknown",
            //                           new Tensor(originshape, dtype));
            tilenode = new TensorNode(_tensornode->name() + "_unknown",
                                      new Tensor(originType));

        tilenode->getLabel()->setDeviceLabel(_p_dev);

        OpNode *opnode = new OpNode(_tensornode->name() + "_fork");
        opnode->setOp(new ScatterOp(strategy, _num));

        tilenode->exlinkUpperNode(opnode);
        opnode->exlinkUpperNode(_tensornode);

        irgraph->pushTensorNode(tilenode);
        irgraph->pushOpNode(opnode);
        irgraph->updateTopology();

        tlabel->setCurrentNode(tilenode);
        tlabel->setCurrentStrategy(strategy);
        tlabel->setApplied();
    }
};

class TransformPattern : public BasePattern {
  public:
    TransformPattern(TensorNode *tensornode, int num)
        : BasePattern(tensornode, num) {}
    TransformPattern(TensorNode *tensornode, const Config &config)
        : BasePattern(tensornode, config) {}
    ~TransformPattern() {}

    void apply(int, IRGraph *) override {}

    void apply(int pre_strategy, int strategy, IRGraph *irgraph) {
        SWLOG_DEBUG(4) << "TransformPattern on tensor " << _tensornode->name()
                       << ", strategy= " << strategy << "\n";
        TilingLabel *tlabel = _tensornode->getTilingLabel();
        TensorType originType = _tensornode->getTensor()->getType();

        TensorNode *tilenode;
        if (strategy >= 0) {
            tilenode = new TensorNode(
                _tensornode->name() + "_tile",
                new Tensor(originType.getTiledTensorType(strategy, _num)));
        } else if (strategy == -1) {
            // tilenode = new TensorNode(_tensornode->name() + "_replicate",
            //                           new Tensor(originshape, dtype));
            tilenode = new TensorNode(_tensornode->name() + "_replicate",
                                      new Tensor(originType));
        } else {
            // tilenode = new TensorNode(_tensornode->name() + "_unknown",
            //                           new Tensor(originshape, dtype));
            tilenode = new TensorNode(_tensornode->name() + "_unknown",
                                      new Tensor(originType));
        }

        tilenode->getLabel()->setDeviceLabel(_p_dev);

        OpNode *opnode = new OpNode(_tensornode->name() + "_transform");
        opnode->setOp(new TransformOp(pre_strategy, strategy, _num));

        TensorNode *pre_par_tnode;
        if (!tlabel->strategyExist(pre_strategy)) {
            // currently, we do not manage _strategy_parnode_map
            // when MEM_SAVING
            pre_par_tnode = tlabel->getCurrentNode();
        } else {
            pre_par_tnode = tlabel->getStrategyParNode(pre_strategy);
        }

        opnode->exlinkUpperNode(pre_par_tnode);
        tilenode->exlinkUpperNode(opnode);
        // opnode->exlinkUpperNode(tlabel->getCurrentNode());

        irgraph->pushTensorNode(tilenode);
        irgraph->pushOpNode(opnode);

        // TODO: this updateTopology seems redundency and frequent
        // wait for checks to rm
        irgraph->updateTopology();
        tlabel->setCurrentNode(tilenode);
        tlabel->setCurrentStrategy(strategy);
        tlabel->setApplied();
    }
};

class JoinPattern : public BasePattern {
  public:
    JoinPattern(TensorNode *tensornode, int num)
        : BasePattern(tensornode, num) {}
    JoinPattern(TensorNode *tensornode, const Config &config)
        : BasePattern(tensornode, config) {}
    ~JoinPattern() {}

    void apply(int strategy, IRGraph *irgraph) override {
        SWLOG_DEBUG(4) << "JoinPattern on tensor " << _tensornode->name()
                       << ", strategy= " << strategy << "\n";
        TilingLabel *tlabel = _tensornode->getTilingLabel();
        TensorType originType = _tensornode->getTensor()->getType();

        TensorNode *tilenode;
        if (strategy >= 0) {
            tilenode = new TensorNode(
                _tensornode->name() + "_t" + std::to_string(strategy),
                new Tensor(originType.getTiledTensorType(strategy, _num)));
        } else if (strategy == -2) {
            // tilenode = new TensorNode(_tensornode->name() + "_tr",
            //                           new Tensor(originshape, dtype));
            tilenode = new TensorNode(_tensornode->name() + "_tr",
                                      new Tensor(originType));
        } else {
            // tilenode = new TensorNode(_tensornode->name() + "_unknown",
            //                           new Tensor(originshape, dtype));
            tilenode = new TensorNode(_tensornode->name() + "_unknown",
                                      new Tensor(originType));
        }

        tilenode->getLabel()->setDeviceLabel(_p_dev);

        OpNode *opnode = new OpNode(_tensornode->name() + "_join");
        // if(strategy == -2) {
        //     opnode->setOp(new ReduceOp());
        // }else {
        //     opnode->setOp(new GatherOp(strategy, _num));
        // }
        opnode->setOp(new GatherOp(strategy, _num));

        opnode->exlinkUpperNode(tilenode);
        _tensornode->exlinkUpperNode(opnode);

        irgraph->pushTensorNode(tilenode);
        irgraph->pushOpNode(opnode);
        irgraph->updateTopology();
        tlabel->setCurrentNode(tilenode);
        tlabel->setCurrentStrategy(strategy);
        tlabel->setApplied();
        SWLOG_DEBUG(4) << "Finish JoinPattern on tensor " << _tensornode->name()
                       << ", strategy= " << strategy << "\n";
    }
};

} // namespace swc

#endif
