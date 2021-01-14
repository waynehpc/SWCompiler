/*************************************************************************
 *    > File Name: wide_resnet.cpp
 *    > Author:  wayne
 *    > mail:
 *    > Created Time: Sun 10 Jan 2021 01:25:34 PM UTC
 ************************************************************************/

#include "SWC.h"
#include <iostream>

using namespace std;
using namespace swc;
using namespace swc::op;

TensorNode *basicblock(IRGraph *graph, std::string scope,
                       /*"resblock"*/ TensorNode *input, int filters,
                       bool downsample) {

    TensorNode *identity = input;
    TensorNode *out;

    if (downsample) {
        out = graph->createConv2d(scope + "_conv0", input, filters, 3,
                                  /*stride*/ 2, 1);
        identity = graph->createConv2d(scope + "_conv_init", input, filters, 1,
                                       /*stride*/ 2, 0);
        identity = graph->createBatchNorm2d(scope + "_bn", identity, filters);
    } else {
        out = graph->createConv2d(scope + "_conv0", input, filters, 3, 1, 1);
    }

    out = graph->createBatchNorm2d(scope + "_bn0", out, filters);
    out = graph->createRelu(scope + "_relu0", out);

    out = graph->createConv2d(scope + "_conv1", out, filters, 3, 1, 1);
    out = graph->createBatchNorm2d(scope + "_bn1", out, filters);
    // out = graph->createRelu(out, scope+"_relu1");

    out = graph->createElementAdd(scope + "_add", out, identity);
    out = graph->createRelu(scope + "_relu", out);

    return out;
}

// https://zhuanlan.zhihu.com/p/54289848
// ResNet V2 e full pre-activation
TensorNode *basicblockV2E(IRGraph *graph, std::string scope,
                          /*"resblock"*/ TensorNode *input, int filters,
                          bool downsample) {
    auto *identity = input;
    TensorNode *out;

    out = graph->createBatchNorm2d(scope + "_bn0", input, filters);
    out = graph->createRelu(scope + "_relu0", out);

    if (downsample) {
        out = graph->createConv2d(scope + "_conv0", out, filters, 3, 2, 1);

        identity =
            graph->createConv2d(scope + "_conv_init", input, filters, 1, 2, 0);
        identity = graph->createBatchNorm2d(scope + "_bn", identity, filters);
    } else {
        out = graph->createConv2d(scope + "_conv0", out, filters, 3, 1, 1);
    }

    out = graph->createBatchNorm2d(scope + "_bn1", out, filters);
    out = graph->createRelu(scope + "_relu1", out);
    out = graph->createConv2d(scope + "_conv1", out, filters, 3, 1, 1);

    auto *add = graph->createElementAdd(scope + "_add", out, identity);

    return add;
}

typedef TensorNode *(*BlockFuncPtr)(IRGraph *, std::string, TensorNode *, int,
                                    bool);

#define WIDEN_FACTOR 4
TensorNode *network(IRGraph *graph, TensorNode *input, std::vector<int> res_n) {

    BlockFuncPtr resblock;

    // resblock = basicblockV2E;
    resblock = basicblock;

    // assert(res_n.size() == 4);
    int filters = 64 * WIDEN_FACTOR;
    int num_classes = 1000;

    auto *x = graph->createConv2d("conv", input, filters, 7, 2, 3);
    x = graph->createBatchNorm2d("bn", x, filters);
    x = graph->createRelu("relu", x);

    x = graph->createMaxPool("pool", x, 3, 2, 1);

    for (int i = 0; i < res_n[0]; i++) {
        x = resblock(graph, "blk0_" + to_string(i), x, filters, false);
    }

    x = resblock(graph, "blk1_0", x, filters * 2, true);
    for (int i = 1; i < res_n[1]; i++) {
        x = resblock(graph, "blk1_" + to_string(i), x, filters * 2, false);
    }

    x = resblock(graph, "blk2_0", x, filters * 4, true);
    for (int i = 1; i < res_n[2]; i++) {
        x = resblock(graph, "blk2_" + to_string(i), x, filters * 4, false);
    }

    x = resblock(graph, "blk3_0", x, filters * 8, true);
    for (int i = 1; i < res_n[3]; i++) {
        x = resblock(graph, "blk3_" + to_string(i), x, filters * 8, false);
    }

    x = graph->createBatchNorm2d("bn", x, filters * 8);
    x = graph->createRelu("relu", x);

    x = graph->createAvgPool("avgpool", x, 7, 1, 0);
    x = graph->createFC("fc", x, num_classes);

    return x;
}

#define MINIBATCH 8

int main() {
    IRGraph *resnet18 = new IRGraph();

    auto *data = resnet18->createTensor("data", {MINIBATCH, 224, 224, 3});
    auto *label =
        resnet18->createTensor("label", {MINIBATCH}, DataType::Int32_t);

    vector<int> res_n = {2, 2, 2, 2};
    auto *out = network(resnet18, data, res_n);

    auto *loss = resnet18->createSoftmaxWithLoss("softmax", out, label);

    resnet18->findInOut();
    resnet18->updateTopology();

    cout << "update topology ok" << endl;

    resnet18->initTensorNodes();

    cout << "init tensornodes ok" << endl;

    // svgGen(resnet18, "resnet18_orig.dot");

    label->require_grad = false;
    data->require_grad = false;
    resnet18->setTrainDataNodes(label, data);
    resnet18->addDisplayTensorNodes(loss);

    svgGen(resnet18, "resnet18.dot");

    Config config;
    config.train_mode = true;
    config.mpi = true;
    config.mpi_size = 4;
    config.train_config.optimizer = "sgd";
    config.train_config.train_data_file = "xxx";
    config.train_config.label_bytes = BytesProto::ONE_BYTE_AS_INT;
    config.train_config.data_bytes = BytesProto::FOUR_BYTES_AS_FLOAT;
    config.train_config.train_data_samples = 50000;
    // config.train_config.snapshot = 1000;
    config.train_config.max_iters = 1000;
    config.train_config.display = 50;

    config.compute_op_annotation = true;
    // config.comm_op_annotation = true;
    config.parallel_preference = COMM_SAVING;
    // config.parallel_preference = MEM_SAVING;

    /*when benchmark enabled, disable emit some code*/
    config.benchmark = true;
    /* not do lowering for node liek FC, FCGrad etc.*/
    config.enable_lowering = false;

    /* about parallel strategy*/
    // config.force_data_parallel = true;
    config.geneticalgo_opt_parallel = true;
    // config.handcraft_parallel = true;

    // optimzer
    config.decentralized_optimizer = true;
    config.use_ring_allreduce = true;

    resnet18->setConfig(config);

    Engine engine(resnet18);
    engine.compile();

    dotGen(resnet18, "resnet18_train.dot");

    cout << resnet18->getCommTrace() << "\n";
    cout << "resnet18-" << resnet18->getCommCost() << "\n";

    string code = engine.genCode();

    return 0;
}