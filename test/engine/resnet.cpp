/*************************************************************************
 *    > File Name: resnet.cpp
 *    > Author:  wayne
 *    > mail:
 *    > Created Time: Thu 24 Dec 2020 07:30:13 AM UTC
 ************************************************************************/

#include "SWC.h"
#include <iostream>

using namespace std;
using namespace swc;
using namespace swc::op;

TensorNode *createTensor(IRGraph *graph, OpNode *parent, std::string name) {
    TensorNode *tnode = new TensorNode(name, parent);
    graph->pushTensorNode(tnode);
    return tnode;
}

TensorNode *createTensor(IRGraph *graph, std::string name) {
    TensorNode *tnode = new TensorNode(name);
    graph->pushTensorNode(tnode);
    return tnode;
}

TensorNode *createTensor(IRGraph *graph, OpNode *parent,
                         const std::initializer_list<size_t> &shape,
                         std::string name, DataType dtype = DataType::Float_t,
                         mem_layout_t layout = layout_default) {

    TensorNode *tnode = new TensorNode(name, shape, parent, dtype, layout);
    graph->pushTensorNode(tnode);
    return tnode;
}

TensorNode *createTensor(IRGraph *graph,
                         const std::initializer_list<size_t> &shape,
                         std::string name, DataType dtype = DataType::Float_t,
                         mem_layout_t layout = layout_default) {

    TensorNode *tnode = new TensorNode(name, shape);
    graph->pushTensorNode(tnode);
    return tnode;
}

// suppose nhwc format
TensorNode *addConv2d(IRGraph *graph, std::string name, TensorNode *input,
                      size_t filters, size_t kernel, size_t stride,
                      size_t padding) {
    // auto idims = input->getDims();
    auto *w = createTensor(graph, {filters, kernel, kernel, 0}, name + "_w");
    auto *b = createTensor(graph, {filters}, name + "_b");
    w->setTraining(1);
    b->setTraining(1);
    vector<size_t> kernels{kernel, kernel};
    vector<size_t> strides{stride, stride};
    vector<size_t> paddings{padding, padding, padding, padding};
    auto *conv = new OpNode(name, new Conv2dOp(kernels, strides, paddings),
                            {input, w, b});
    graph->pushOpNode(conv);
    auto *output = createTensor(graph, conv, name + "_t");
    return output;
}

TensorNode *addFC(IRGraph *graph, std::string name, TensorNode *input,
                  size_t out_features) {
    // auto idims = input->getDims();
    auto *w = createTensor(graph, {0, out_features}, name + "_w");
    auto *b = createTensor(graph, {out_features}, name + "_b");
    w->setTraining(1);
    b->setTraining(1);
    auto *fc = new OpNode(name, new MatrixMatrixFCBiasOp(), {input, w, b});
    graph->pushOpNode(fc);
    auto *output = createTensor(graph, fc, name + "_t");
    return output;
}

TensorNode *addMaxPool(IRGraph *graph, std::string name, TensorNode *input,
                       size_t kernel, size_t stride, size_t padding) {
    vector<size_t> kernels{kernel, kernel};
    vector<size_t> strides{stride, stride};
    vector<size_t> paddings{padding, padding, padding, padding};
    auto *pool =
        new OpNode(name, new MaxPoolOp(kernels, strides, paddings), {input});
    graph->pushOpNode(pool);
    auto *output = createTensor(graph, pool, name + "_t");
    return output;
}

TensorNode *addAvgPool(IRGraph *graph, std::string name, TensorNode *input,
                       size_t kernel, size_t stride, size_t padding) {
    vector<size_t> kernels{kernel, kernel};
    vector<size_t> strides{stride, stride};
    vector<size_t> paddings{padding, padding, padding, padding};
    auto *pool =
        new OpNode(name, new AvgPoolOp(kernels, strides, paddings), {input});
    graph->pushOpNode(pool);
    auto *output = createTensor(graph, pool, name + "_t");
    return output;
}

TensorNode *addBatchNorm2d(IRGraph *graph, std::string name, TensorNode *input,
                           size_t num_features, float eps = 1e-5,
                           float momentum = 0.1) {
    auto *mean = createTensor(graph, {num_features}, name + "_mean");
    auto *var = createTensor(graph, {num_features}, name + "_var");
    auto *scale = createTensor(graph, {num_features}, name + "_scale");
    auto *shift = createTensor(graph, {num_features}, name + "_shift");

    auto *bn = new OpNode(name, new BatchNorm2dOp(eps, momentum),
                          {input, mean, var, scale, shift});
    graph->pushOpNode(bn);
    auto *output = createTensor(graph, bn, name + "_t");
    return output;
}

TensorNode *addRelu(IRGraph *graph, std::string name, TensorNode *input) {
    auto *relu = new OpNode(name, new ReluOp(), {input});
    graph->pushOpNode(relu);
    auto *output = createTensor(graph, relu, name + "_t");
    return output;
}

TensorNode *addElementAdd(IRGraph *graph, std::string name, TensorNode *lhs,
                          TensorNode *rhs) {
    auto *add = new OpNode(name, new ElementAddOp(), {lhs, rhs});
    graph->pushOpNode(add);
    auto *output = createTensor(graph, add, name + "_t");
    return output;
}

TensorNode *addSoftmax(IRGraph *graph, std::string name, TensorNode *input,
                       TensorNode *label) {
    auto *sfm = new OpNode(name, new MatrixSoftmaxWithLossOp(), {input, label});
    graph->pushOpNode(sfm);
    auto *prob = createTensor(graph, sfm, "prob");
    auto *loss = createTensor(graph, sfm, /*{1},*/ "loss");
    (void)prob;
    return loss;
}

TensorNode *basicblock(IRGraph *graph, std::string scope,
                       /*"resblock"*/ TensorNode *input, int filters,
                       bool downsample) {

    TensorNode *identity = input;
    TensorNode *out;

    if (downsample) {
        out = addConv2d(graph, scope + "_conv0", input, filters, 3,
                        /*stride*/ 2, 1);
        identity = addConv2d(graph, scope + "_conv_init", input, filters, 1,
                             /*stride*/ 2, 0);
        identity = addBatchNorm2d(graph, scope + "_bn", identity, filters);
    } else {
        out = addConv2d(graph, scope + "_conv0", input, filters, 3, 1, 1);
    }

    out = addBatchNorm2d(graph, scope + "_bn0", out, filters);
    out = addRelu(graph, scope + "_relu0", out);

    out = addConv2d(graph, scope + "_conv1", out, filters, 3, 1, 1);
    out = addBatchNorm2d(graph, scope + "_bn1", out, filters);
    // out = addRelu(graph, out, scope+"_relu1");

    out = addElementAdd(graph, scope + "_add", out, identity);
    out = addRelu(graph, scope + "_relu", out);

    return out;
}

// https://zhuanlan.zhihu.com/p/54289848
// ResNet V2 e full pre-activation
TensorNode *basicblockV2E(IRGraph *graph, std::string scope,
                          /*"resblock"*/ TensorNode *input, int filters,
                          bool downsample) {
    auto *identity = input;
    TensorNode *out;

    out = addBatchNorm2d(graph, scope + "_bn0", input, filters);
    out = addRelu(graph, scope + "_relu0", out);

    if (downsample) {
        out = addConv2d(graph, scope + "_conv0", out, filters, 3, 2, 1);

        identity =
            addConv2d(graph, scope + "_conv_init", input, filters, 1, 2, 0);
        identity = addBatchNorm2d(graph, scope + "_bn", identity, filters);
    } else {
        out = addConv2d(graph, scope + "_conv0", out, filters, 3, 1, 1);
    }

    out = addBatchNorm2d(graph, scope + "_bn1", out, filters);
    out = addRelu(graph, scope + "_relu1", out);
    out = addConv2d(graph, scope + "_conv1", out, filters, 3, 1, 1);

    auto *add = addElementAdd(graph, scope + "_add", out, identity);

    return add;
}

typedef TensorNode *(*BlockFuncPtr)(IRGraph *, std::string, TensorNode *, int,
                                    bool);

TensorNode *network(IRGraph *graph, TensorNode *input, std::vector<int> res_n) {

    BlockFuncPtr resblock;

    // resblock = basicblockV2E;
    resblock = basicblock;

    // assert(res_n.size() == 4);
    int filters = 64;
    int num_classes = 1000;

    auto *x = addConv2d(graph, "conv", input, filters, 7, 2, 3);
    x = addBatchNorm2d(graph, "bn", x, filters);
    x = addRelu(graph, "relu", x);

    x = addMaxPool(graph, "pool", x, 3, 2, 1);

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

    x = addBatchNorm2d(graph, "bn", x, filters * 8);
    x = addRelu(graph, "relu", x);

    x = addAvgPool(graph, "avgpool", x, 7, 1, 0);
    x = addFC(graph, "fc", x, num_classes);

    return x;
}

int main() {
    IRGraph *resnet18 = new IRGraph();
    auto *data = createTensor(resnet18, {8, 224, 224, 3}, "data");
    auto *label = createTensor(resnet18, {8}, "label", DataType::Int32_t);

    vector<int> res_n = {2, 2, 2, 2};
    auto *out = network(resnet18, data, res_n);

    // fc->out
    auto *loss = addSoftmax(resnet18, "softmax", out, label);

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
    config.force_data_parallel = true;
    // config.geneticalgo_opt_parallel = true;
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

    // TODO
    // 0. bottleneck, resnet50
    // 1. addBatchNorm2d 5 inputs
    // 2. BN autodiff
    // 3. bngrad kernels and codegen, cudacodegen
    // einSum of Element Add

    return 0;
}
