/*************************************************************************
 *    > File Name: alexnet_imagenet.cpp
 *    > Author:  wayne
 *    > mail:
 *    > Created Time: Thu 07 Jan 2021 02:03:16 AM UTC
 ************************************************************************/

#include "SWC.h"
#include <iostream>

using namespace swc;
using namespace swc::op;
using namespace swc::pass;
using namespace std;

#define MINIBATCH 32
#define PARA_DEGREE 4

int main() {
    auto *alexnet = new IRGraph();

    auto *data = alexnet->createTensor("data", {MINIBATCH, 227, 227, 3});
    auto *x = alexnet->createConv2d("conv1", data, 96, 11, 4, 0);
    x = alexnet->createRelu("relu1", x);
    x = alexnet->createLRN("lrn1", x);
    x = alexnet->createMaxPool("pool1", x, 3, 2, 0);

    x = alexnet->createConv2d("conv2", x, 256, 5, 1, 2);
    x = alexnet->createRelu("relu2", x);
    x = alexnet->createLRN("lrn2", x);
    x = alexnet->createMaxPool("pool2", x, 3, 2, 0);

    x = alexnet->createConv2d("conv3", x, 384, 3, 1, 1);
    x = alexnet->createRelu("relu3", x);

    x = alexnet->createConv2d("conv4", x, 384, 3, 1, 1);
    x = alexnet->createRelu("relu4", x);

    x = alexnet->createConv2d("conv5", x, 256, 3, 1, 1);
    x = alexnet->createRelu("relu5", x);
    x = alexnet->createMaxPool("pool5", x, 3, 2, 0);

    x = alexnet->createFC("fc6", x, 4096);
    x = alexnet->createRelu("relu6", x);
    x = alexnet->createDropout("dropout6", x);

    x = alexnet->createFC("fc7", x, 4096);
    x = alexnet->createRelu("relu7", x);
    x = alexnet->createDropout("dropout7", x);

    x = alexnet->createFC("fc8", x, 1000);

    auto *label =
        alexnet->createTensor("label", {MINIBATCH}, DataType::Int32_t);
    auto *loss = alexnet->createSoftmaxWithLoss("softmax", x, label);

    alexnet->findInOut();
    alexnet->updateTopology();

    alexnet->initTensorNodes();

    alexnet->setTrainDataNodes(label, data);
    alexnet->addDisplayTensorNodes(loss);

    Config config;

    config.train_mode = true;
    // config.mkldnn = true;
    config.mpi = true;
    config.mpi_size = PARA_DEGREE;

    config.train_config.optimizer = "sgd";
    config.train_config.train_data_file = "mnist_labels_images.bin";
    config.train_config.label_bytes = BytesProto::ONE_BYTE_AS_INT;
    config.train_config.data_bytes = BytesProto::FOUR_BYTES_AS_FLOAT;
    config.train_config.train_data_samples = 50000;
    // config.train_config.snapshot = 1000;
    config.train_config.max_iters = 100;
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

    alexnet->setConfig(config);
    std::cout << "alexnet_b" << MINIBATCH << "_p" << config.mpi_size << "\n";

    svgGen(alexnet, "alexnet_infer.dot");

    Engine engine(alexnet);
    engine.compile();

    dotGen(alexnet, "alexnet_train.dot");
    cout << alexnet->getCommTrace() << "\n";
    cout << alexnet->getCommCost() << "\n";

    string code = engine.genCode();
    // cout << code << "\n";

    return 0;
}
