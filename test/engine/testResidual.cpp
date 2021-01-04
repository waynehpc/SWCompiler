/*************************************************************************
 *    > File Name: testResidual.cpp
 *    > Author:  wayne
 *    > mail:
 *    > Created Time: Wed 23 Dec 2020 02:59:02 PM UTC
 ************************************************************************/

#include <ctime>
#include <iostream>

#include "SWC.h"

using namespace swc;
using namespace swc::op;
using namespace swc::pass;
using namespace std;

#define MINIBATCH 64

int main() {
    //============================
	// Example of 2 FC layer:
	// T:data0   T:weight0
	// 	\       /
	// 	\     /
	// 	O:fc0 -- T:bias0
	// 		|
	// 	T:data1
	// 		|
	// 	O:tanh0
	// 		|
	// 	T:data2
	// 				T:weight1a
	// 	/	\       /
	// /	 \     /
	//fc1b      O:fc1a -- T:bias1a
	// 	\			|
	// 	T:data3b	T:data3a
	// 		\		/	
	// 		 O: add
	// 			|
	// 			data3
	// 			|
	// 		O: softmax
	// 			|
	// 		T:data4
    //=============================

    TENSOR(data0, MINIBATCH, 784);
    TENSOR(weight0, 784, 512);
    TENSOR(bias0, 512);
    weight0_Tensor->setTensorInit(TensorInitType::XAVIER, 784);
    bias0_Tensor->setTensorInit(TensorInitType::CONSTANT, 0);
    weight0->setTraining(1);
    bias0->setTraining(1);
    // setExternal can be depreciated when we do autodiff
    // directly on IRGraph
    weight0->setExternal(true);
    bias0->setExternal(true);

    OP(fc0, MatrixMatrixFCBiasOp);
    LINKUPPER(fc0, data0, weight0, bias0);

    TENSOR(data1, 0);
    LINKUPPER(data1, fc0);

    OP(tanh0, MatrixTanhOp);
    LINKUPPER(tanh0, data1);

    TENSOR(data2, 0);
    LINKUPPER(data2, tanh0);

    // define IR graph
    G(mlp);
    GpT(mlp, data0, data1, data2, weight0, bias0);
    GpO(mlp, fc0, tanh0);

	// ----------------------------------------------------------------
    TENSOR(weight1a, 0, 10);
    TENSOR(bias1a, 10);
    weight1a_Tensor->setTensorInit(TensorInitType::XAVIER, 512);
    bias1a_Tensor->setTensorInit(TensorInitType::CONSTANT, 0);
    weight1a->setTraining(1);
    bias1a->setTraining(1);

    OP(fc1a, MatrixMatrixFCBiasOp);
    LINKUPPER(fc1a, data2, weight1a, bias1a);

	TENSOR(data3a, 0);
    LINKUPPER(data3a, fc1a);
	// ----------------------------------------------------------------
	TENSOR(weight1b, 0, 10);
    TENSOR(bias1b, 10);
    weight1b_Tensor->setTensorInit(TensorInitType::XAVIER, 512);
    bias1b_Tensor->setTensorInit(TensorInitType::CONSTANT, 0);
    weight1b->setTraining(1);
    bias1b->setTraining(1);

    OP(fc1b, MatrixMatrixFCBiasOp);
    LINKUPPER(fc1b, data2, weight1b, bias1b);

    TENSOR(data3b, 0);
    LINKUPPER(data3b, fc1b);

	// ----------------------------------------------------------------
	OP(add, ElementAddOp);
    LINKUPPER(add, data3a, data3b);

	TENSOR(data3, 0);
    LINKUPPER(data3, add);
	// ----------------------------------------------------------------
	GpT(mlp, weight1a, bias1a, data3a, weight1b, bias1b, data3b, data3);
    GpO(mlp, fc1a, fc1b, add);
	// ----------------------------------------------------------------

    Tensor *labelt = new Tensor({MINIBATCH}, DataType::Int32_t);
    TensorNode *label = new TensorNode("selected", labelt);

    OP(softmax, MatrixSoftmaxWithLossOp);
    LINKUPPER(softmax, data3, label);
    TENSOR(data4, 0);
    LINKUPPER(data4, softmax);
    TENSOR(loss, 1);
    LINKUPPER(loss, softmax);

    GpT(mlp, data4, label, loss);
    GpO(mlp, softmax);

    mlp->findInOut();
    mlp->updateTopology();

    mlp->initTensorNodes();

    mlp->setTrainDataNodes(label, data0);
    mlp->addDisplayTensorNodes(loss);

    Config config;
    config.train_mode = true;
    // config.mpi = true;
    // config.mpi_size = 4;
    config.train_config.optimizer = "sgd";
    config.train_config.train_data_file = "mnist_labels_images.bin";
    config.train_config.label_bytes = BytesProto::ONE_BYTE_AS_INT;
    config.train_config.data_bytes = BytesProto::FOUR_BYTES_AS_FLOAT;
    config.train_config.train_data_samples = 50000;
    // config.train_config.snapshot = 1000;
    config.train_config.max_iters = 1000;
    config.train_config.display = 50;
    // config.compute_op_annotation = true;
    // config.comm_op_annotation = true;
    config.parallel_preference = COMM_SAVING;
    // config.parallel_preference = MEM_SAVING;

    mlp->setConfig(config);

    svgGen(mlp, "residual_def.dot");

    Engine engine(mlp);
    engine.compile();

    svgGen(mlp, "residual_train.dot");

    cout << mlp->getCommTrace() << "\n";
    cout << mlp->getCommCost() << "\n";

    string code = engine.genCode();
    // cout << code;

    return 0;
}

