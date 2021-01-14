/*************************************************************************
 *    > File Name: mnist_autoencoder.cpp
 *    > Author:  wayne
 *    > mail:
 *    > Created Time: Thu 07 Jan 2021 01:40:26 PM UTC
 ************************************************************************/

#include "SWC.h"
#include <iostream>

using namespace swc;
using namespace swc::op;
using namespace swc::pass;
using namespace std;

#define MINIBATCH 32
#define PARA_DEGREE 16 

int main() {
    auto *autoencoder = new IRGraph();

    auto *data = autoencoder->createTensor("data", {MINIBATCH, 784});

    auto *x = autoencoder->createFC("enc1", data, 1024);
    x = autoencoder->createSigmoid("enc1act", x);
    x = autoencoder->createFC("enc2", x, 512);
    x = autoencoder->createSigmoid("enc2act", x);
    x = autoencoder->createFC("enc3", x, 256);
    x = autoencoder->createSigmoid("enc3act", x);
    x = autoencoder->createFC("enc4", x, 32);
    x = autoencoder->createFC("dec4", x, 256);
    x = autoencoder->createSigmoid("dec4act", x);
    x = autoencoder->createFC("dec3", x, 512);
    x = autoencoder->createSigmoid("dec3act", x);
    x = autoencoder->createFC("dec2", x, 1024);
    x = autoencoder->createSigmoid("dec2act", x);
    x = autoencoder->createFC("dec1", x, 784);

    auto *loss1 = autoencoder->createSigmoidCrossEntropyLoss(
        "cross_entropy_loss", x, data);

    x = autoencoder->createSigmoid("dec1act", x);
    auto *loss2 = autoencoder->createEuclideanLoss("l2_loss", x, data);

    autoencoder->findInOut();
    autoencoder->updateTopology();

    autoencoder->initTensorNodes();

    auto *placeholder = autoencoder->createTensor("placeholder", {MINIBATCH});
    placeholder->getLabel()->setIsOut();

    autoencoder->setTrainDataNodes(placeholder, data);
    autoencoder->addDisplayTensorNodes(loss1, loss2);

    Config config;

    config.train_mode = true;
    // // config.mkldnn = true;
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
    config.force_data_parallel = true;
    // config.geneticalgo_opt_parallel = true;
    // config.handcraft_parallel = true;

    // optimzer
    config.decentralized_optimizer = true;
    // config.use_ring_allreduce = true;

    autoencoder->setConfig(config);
    std::cout << "autoencoder_b" << MINIBATCH << "_p" << config.mpi_size
              << "\n";

    // svgGen(autoencoder, "autoencoder.dot");

    Engine engine(autoencoder);
    engine.compile();

    // svgGen(autoencoder, "autoencoder_train.dot");

    cout << autoencoder->getCommTrace() << "\n";
    cout << autoencoder->getCommCost() << "\n";

    string code = engine.genCode();
    // cout << code << "\n";

    return 0;
}
