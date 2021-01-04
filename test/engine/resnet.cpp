/*************************************************************************
 *    > File Name: resnet.cpp
 *    > Author:  wayne
 *    > mail:
 *    > Created Time: Thu 24 Dec 2020 07:30:13 AM UTC
 ************************************************************************/

#include <iostream>
#include "SWC.h"


using namespace std;
using namespace swc;
using namespace swc::op;

TensorNode* createTensor(IRGraph *graph, OpNode *parent,
	std::string name) {
	TensorNode * tnode = new TensorNode(name, parent);
	graph->pushTensorNode(tnode);
	return tnode;
}

TensorNode* createTensor(IRGraph *graph, std::string name) {
	TensorNode * tnode = new TensorNode(name);
	graph->pushTensorNode(tnode);
	return tnode;
}

TensorNode* createTensor(IRGraph *graph, OpNode *parent, const std::initializer_list<size_t> &shape, 
	std::string name, DataType dtype=DataType::Float_t, mem_layout_t layout = layout_default) {

	TensorNode * tnode = new TensorNode(name, shape, parent, dtype, layout);
	graph->pushTensorNode(tnode);
	return tnode;
}

TensorNode* createTensor(IRGraph *graph, const std::initializer_list<size_t> &shape, 
	std::string name, DataType dtype=DataType::Float_t, mem_layout_t layout = layout_default) {

	TensorNode * tnode = new TensorNode(name, shape);
	graph->pushTensorNode(tnode);
	return tnode;
}



// suppose nhwc format
TensorNode* addConv2d(IRGraph *graph, TensorNode *input, size_t filters, size_t kernel, size_t stride, size_t padding, std::string name="conv") {
	// auto idims = input->getDims();
	auto *w = createTensor(graph, {filters, kernel, kernel, 0}, name+"_w");
	auto *b = createTensor(graph, {filters}, name+"_b");
	// auto *w = createTensor(graph, name+"_w");
	// auto *b = createTensor(graph, name+"b");
	vector<size_t> kernels{kernel, kernel};
	vector<size_t> strides{stride, stride};
	vector<size_t> paddings{padding, padding, padding, padding};
	auto *conv = new OpNode(name, 
		new Conv2dOp(kernels, strides, paddings), 
		{input, w, b});
	graph->pushOpNode(conv);
	auto *output = createTensor(graph, conv, name+"_t");
	return output;
}

TensorNode* addFC(IRGraph *graph, TensorNode *input, size_t out_features, string name="fc") {
	// auto idims = input->getDims();
	auto *w = createTensor(graph, {0, out_features}, name+"_w");
	auto *b = createTensor(graph, {out_features}, name+"_b");

	auto *conv = new OpNode(name, 
		new MatrixMatrixFCBiasOp(), 
		{input, w, b});
	graph->pushOpNode(conv);
	auto *output = createTensor(graph, conv, name+"_t");
	return output;
}

TensorNode *addMaxPool(IRGraph *graph, TensorNode *input, size_t kernel, size_t stride, size_t padding, std::string name="pool") {
	vector<size_t> kernels{kernel, kernel};
	vector<size_t> strides{stride, stride};
	vector<size_t> paddings{padding, padding, padding, padding};
	auto *pool = new OpNode(name, 
		new MaxPoolOp(kernels, strides, paddings), 
		{input});
	graph->pushOpNode(pool);
	auto *output = createTensor(graph, pool, name+"_t");
	return output;
}

TensorNode *addAvgPool(IRGraph *graph, TensorNode *input, size_t kernel, size_t stride, size_t padding, std::string name="pool") {
	vector<size_t> kernels{kernel, kernel};
	vector<size_t> strides{stride, stride};
	vector<size_t> paddings{padding, padding, padding, padding};
	auto *pool = new OpNode(name, 
		new AvgPoolOp(kernels, strides, paddings), 
		{input});
	graph->pushOpNode(pool);
	auto *output = createTensor(graph, pool, name+"_t");
	return output;
}

TensorNode *addBN(IRGraph *graph, TensorNode *input, float eps=1e-3, std::string name="bn") {
	auto *bn = new OpNode(name,
		new BatchNormalizationOp(eps), {input});
	graph->pushOpNode(bn);
	auto *output = createTensor(graph, bn, name+"_t");
	return output;
}

TensorNode *addRelu(IRGraph *graph, TensorNode *input, std::string name="relu") {
	auto *relu = new OpNode(name,
		new ReluOp(), {input});
	graph->pushOpNode(relu);
	auto *output = createTensor(graph, relu, name+"_t");
	return output;
}

TensorNode *addElementAdd(IRGraph *graph, TensorNode* lhs, TensorNode *rhs, std::string name="add") {
	auto *add = new OpNode(name, 
		new ElementAddOp(), {lhs, rhs});
	graph->pushOpNode(add);
	auto *output = createTensor(graph, add, name+"_t");
	return output;
}

TensorNode *addSoftmax(IRGraph *graph, TensorNode *input, TensorNode *label, string name="softmax") {
	auto *sfm = new OpNode(name, new MatrixSoftmaxWithLossOp(), {input, label});
	graph->pushOpNode(sfm);
	auto *prob = createTensor(graph, sfm, "prob");
	auto *loss = createTensor(graph, sfm, /*{1},*/ "loss");
	(void)prob;
	return loss;
}

TensorNode *basicblock(IRGraph *graph, TensorNode* input, int filters, bool downsample, std::string scope="resblock") {
	
	TensorNode *identity = input;
	TensorNode *out;

	if(downsample) {
		out = addConv2d(graph, input, filters, 3, /*stride*/2, 1, scope+"_conv0");
		identity = addConv2d(graph, input, filters, 1, /*stride*/2, 0, scope+"_conv_init");
		identity = addBN(graph, identity, 1e-3, scope+"_bn");
	} else {
		out = addConv2d(graph, input, filters, 3, 1, 1, scope+"_conv0");
	}

	out = addBN(graph, out, 1e-3, scope+"_bn0");
	out = addRelu(graph, out, scope+"_relu0");

	out = addConv2d(graph, out, filters, 3, 1, 1, scope+"_conv1");
	out = addBN(graph, out, 1e-3, scope+"_bn1");
	// out = addRelu(graph, out, scope+"_relu1");

	out = addElementAdd(graph, out, identity, scope+"_add");
	out = addRelu(graph, out, scope+"_relu");
	
	return out;
}


// https://zhuanlan.zhihu.com/p/54289848
// ResNet V2 e full pre-activation
TensorNode *basicblockV2E(IRGraph *graph, TensorNode* input, int filters, bool downsample, std::string scope="resblock") {
	auto *identity = input;
	TensorNode *out;

	out = addBN(graph, input, 1e-3, scope+"_bn0");
	out = addRelu(graph, out, scope+"_relu0");

	if(downsample) {
		out = addConv2d(graph, out, filters, 3, 2, 1, scope+"_conv0");

		identity = addConv2d(graph, input, filters, 1, 2, 0, scope+"_conv_init");
		identity = addBN(graph, identity, 1e-3, scope+"_bn");
	} else {
		out = addConv2d(graph, out, filters, 3, 1, 1, scope+"_conv0");
	}

	out = addBN(graph, out, 1e-3, scope+"_bn1");
	out = addRelu(graph, out, scope+"_relu1");
	out = addConv2d(graph, out, filters, 3, 1, 1, scope+"_conv1");

	auto *add = addElementAdd(graph, out, identity, scope+"_add");
	
	return add;
}

typedef TensorNode* (*BlockFuncPtr)(IRGraph*, TensorNode* , int , bool , std::string);

TensorNode *network(IRGraph *graph, TensorNode *input, std::vector<int> res_n) {

	BlockFuncPtr resblock;

	// resblock = basicblockV2E;
	resblock = basicblock;

	// assert(res_n.size() == 4);
	int filters = 64;
	int num_classes = 1000;

	auto *x = addConv2d(graph, input, filters, 7, 2, 3, "conv");
	x = addBN(graph, x, 1e-3, "bn");
	x = addRelu(graph, x, "relu");

	x = addMaxPool(graph, x, 3, 2, 1, "pool");

	for(int i=0; i<res_n[0]; i++) {
		x = resblock(graph, x, filters, false, "blk0_"+to_string(i));
	}

	x = resblock(graph, x, filters*2, true, "blk1_0");
	for(int i=1; i<res_n[1]; i++) {
		x = resblock(graph, x, filters*2, false, "blk1_"+to_string(i));
	}

	x = resblock(graph, x, filters*4, true, "blk2_0");
	for(int i=1; i<res_n[2]; i++) {
		x = resblock(graph, x, filters*4, false, "blk2_"+to_string(i));
	}

	x = resblock(graph, x, filters*8, true, "blk3_0");
	for(int i=1; i<res_n[3]; i++) {
		x = resblock(graph, x, filters*8, false, "blk3_"+to_string(i));
	}

	x = addBN(graph, x, 1e-3, "bn");
	x = addRelu(graph, x, "relu");

	x = addAvgPool(graph, x, 7, 1, 0, "avgpool");
	x = addFC(graph, x, num_classes, "fc");

	return x;
}

int main()
{
	IRGraph *resnet18 = new IRGraph();
	auto *data = createTensor(resnet18, {8, 224, 224, 3}, "data");
	auto *label = createTensor(resnet18, {8}, "label", DataType::Int32_t);

	vector<int> res_n = {2, 2, 2, 2};
	auto *out = network(resnet18, data, res_n);

	// fc->out
	auto *loss = addSoftmax(resnet18, out, label, "softmax");

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

	resnet18->setConfig(config);

	Engine engine(resnet18);
	engine.compile();

	dotGen(resnet18, "resnet18_train.dot");
	

	cout << resnet18->getCommTrace() << "\n";
    cout << "resnet18-" << resnet18->getCommCost() << "\n";

	string code = engine.genCode();


	// TODO
	// 0. bottleneck, resnet50
	// 1. addBN 5 inputs
	// 2. BN autodiff
	// 3. bngrad kernels and codegen, cudacodegen
	// einSum of Element Add

	return 0;
}
