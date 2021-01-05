/**********************************************
  > File Name		: testRNN.cpp
  > Author			: wwz
  > Mail			: wumz17@163.com
  > Created Time	: 2021年01月05日 星期二 14时53分07秒
 ****************************************/

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


TensorNode *addSoftmax(IRGraph *graph, TensorNode *input, TensorNode *label, string name="softmax") {
	auto *sfm = new OpNode(name, new MatrixSoftmaxWithLossOp(), {input, label});
	graph->pushOpNode(sfm);
	auto *prob = createTensor(graph, sfm, "prob");
	auto *loss = createTensor(graph, sfm, /*{1},*/ "loss");
	(void)prob;
	return loss;
}

TensorNode *addElementAdd(IRGraph *graph, TensorNode* lhs, TensorNode *rhs, std::string name="add") {
	auto *add = new OpNode(name, 
		new ElementAddOp(), {lhs, rhs});
	graph->pushOpNode(add);
	auto *output = createTensor(graph, add, name+"_t");
	return output;
}

TensorNode *addTanh(IRGraph *graph, TensorNode *input, std::string name="tanh") {
	auto *tanh = new OpNode(name,
		new MatrixTanhOp(), {input});
	graph->pushOpNode(tanh);
	auto *output = createTensor(graph, tanh, name+"_tanh");
	return output;
}

TensorNode *addRNN_unit(IRGraph *graph, TensorNode *wx, TensorNode *wh, TensorNode *b, TensorNode *hidden, TensorNode *input, string name="rnn_unit") {
	auto *mmb_x = new OpNode(name+"_out", new MatrixMatrixFCBiasOp(), {input, wx, b});
	graph->pushOpNode(mmb_x);
	auto *hx = createTensor(graph, mmb_x, name+"_hx");
	auto *mm_h = new OpNode(name+"_hidden", new MatrixMatrixFCOp(), {hidden, wh});
	graph->pushOpNode(mm_h);
	auto *hh = createTensor(graph, mm_h, name+"_hh");
	auto *next_h = addElementAdd(graph, hx, hh, name+"add");

	next_h = addTanh(graph, next_h, name+"tanh");
	return next_h;
}


TensorNode *addRNN(IRGraph *graph, vector<TensorNode*> input, TensorNode *hidden_init, int time_stamp, size_t hidden_size, string name="rnn") {
	auto *wx = createTensor(graph, {0, hidden_size}, name+"_wx");
	auto *wh = createTensor(graph, {0, hidden_size}, name+"_wh");
	auto *b = createTensor(graph, {hidden_size}, name+"_b");
	wx->setTraining(1);
	wh->setTraining(1);
	b->setTraining(1);

	auto *hidden = hidden_init;
	for(int i=0; i<time_stamp; i++){
		hidden = addRNN_unit(graph, wx, wh, b, hidden, input[i], name+"rnn_unit_"+to_string(i));
	}
	return hidden;
}


TensorNode* addFC(IRGraph *graph, TensorNode *input, size_t out_features, string name="fc") {
	// auto idims = input->getDims();
	auto *w = createTensor(graph, {0, out_features}, name+"_w");
	auto *b = createTensor(graph, {out_features}, name+"_b");

	auto *mm = new OpNode(name, 
		new MatrixMatrixFCBiasOp(), 
		{input, w, b});
	graph->pushOpNode(mm);
	auto *output = createTensor(graph, mm, name+"_t");
	return output;
}

TensorNode *addRelu(IRGraph *graph, TensorNode *input, std::string name="relu") {
	auto *relu = new OpNode(name,
		new ReluOp(), {input});
	graph->pushOpNode(relu);
	auto *output = createTensor(graph, relu, name+"_t");
	return output;
}

TensorNode* network(IRGraph *graph,vector<TensorNode*> input, TensorNode* hidden_init, int time_stamp){
	int num_classes = 2;
	int hidden_size = 100;
	auto *x = addRNN(graph, input, hidden_init, time_stamp, hidden_size, "fc_1");
	//auto *x = addFC(graph, input, hidden_size, "fc_1");
	x = addRelu(graph, x, "relu_1");
	x = addFC(graph, x, num_classes, "fc_2");
	return x;
}



int main()
{
	IRGraph *lstmtc = new IRGraph();
	int time_stamp = 10;

	vector<TensorNode*> data;
	for(int i=0; i<time_stamp; i++){
		auto *tmp = createTensor(lstmtc, {8, 4096}, "data");
		data.push_back(tmp);
	}
	//auto *data = createTensor(lstmtc, {8, 4096}, "data");
	auto *hidden_init = createTensor(lstmtc, {8, 100}, "hidden_init");
	auto *label = createTensor(lstmtc, {8}, "label", DataType::Int32_t);

	auto *out = network(lstmtc, data, hidden_init, time_stamp); // 4096 -> 1024 -> 10
	
	auto *loss = addSoftmax(lstmtc, out, label, "softmax");

	lstmtc->findInOut();
	lstmtc->updateTopology();

	cout << "update topology ok" << endl;

	lstmtc->initTensorNodes();

	cout << "init tensornodes ok" << endl;

	// svgGen(lstmtc, "lstmtc_orig.dot");


	//label->require_grad = false;
	//data->require_grad = false;    
	//lstmtc->setTrainDataNodes(label, data);
	//lstmtc->addDisplayTensorNodes(loss);

	svgGen(lstmtc, "lstmtc.dot");

	Config config;
    config.train_mode = true;
	//config.mpi = true;
    //config.mpi_size = 4;
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

	lstmtc->setConfig(config);

	Engine engine(lstmtc);
	engine.compile();

	dotGen(lstmtc, "lstmtc_train.dot");
	//svgGen(lstmtc, "lstmtc_train.dot");
	

	//cout << lstmtc->getCommTrace() << "\n";
    //cout << "lstmtc-" << lstmtc->getCommCost() << "\n";

	//string code = engine.genCode();


	// TODO
	// 0. bottleneck, resnet50
	// 1. addBN 5 inputs
	// 2. BN autodiff
	// 3. bngrad kernels and codegen, cudacodegen
	// einSum of Element Add
	return 0;
}

