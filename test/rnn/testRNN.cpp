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


TensorNode *addElementMul(IRGraph *graph, TensorNode* lhs, TensorNode *rhs, std::string name="mul") {
	auto *mul = new OpNode(name, 
		new ElementMulOp(), {lhs, rhs});
	graph->pushOpNode(mul);
	auto *output = createTensor(graph, mul, name+"_t");
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

TensorNode* network(IRGraph *graph,vector<TensorNode*> &input, TensorNode* hidden_init, int time_stamp){
	int num_classes = 2;
	int hidden_size = 100;
	auto *x = addRNN(graph, input, hidden_init, time_stamp, hidden_size);
	//auto *x = addFC(graph, input, hidden_size, "fc_1");
	x = addRelu(graph, x, "relu_1");
	x = addFC(graph, x, num_classes, "fc_2");
	return x;
}


//TensorNode *addSigmoid(IRGraph *graph, TensorNode *input, std::string name="sigmoid") {
//	auto *sigmoid = new OpNode(name,
//		new SigmoidOp(), {input});
//	graph->pushOpNode(sigmoid);
//	auto *output = createTensor(graph, sigmoid, name+"_t");
//	return output;
//}


void addLSTM_unit(IRGraph *graph, TensorNode *input, TensorNode *prev_h, TensorNode *prev_c, vector<TensorNode*> &wx, vector<TensorNode*> &wh, vector<TensorNode*> b, TensorNode** next_h, TensorNode** next_c, string name="lstm_unit") {
	// wx 4*[D, H]
	// wh 4*[H, H]
	// b  4*[H]
	
	auto *mmb_zg = new OpNode(name+"_zg_mmb", new MatrixMatrixFCBiasOp(), {input, wx[0], b[0]});
	auto *mmb_zi = new OpNode(name+"_zi_mmb", new MatrixMatrixFCBiasOp(), {input, wx[1], b[1]});
	auto *mmb_zf = new OpNode(name+"_zf_mmb", new MatrixMatrixFCBiasOp(), {input, wx[2], b[2]});
	auto *mmb_zo = new OpNode(name+"_zo_mmb", new MatrixMatrixFCBiasOp(), {input, wx[3], b[3]});
	
	auto *mm_zg = new OpNode(name+"_zg_mm", new MatrixMatrixFCOp(), {prev_h, wh[0]});
	auto *mm_zi = new OpNode(name+"_zi_mm", new MatrixMatrixFCOp(), {prev_h, wh[1]});
	auto *mm_zf = new OpNode(name+"_zf_mm", new MatrixMatrixFCOp(), {prev_h, wh[2]});
	auto *mm_zo = new OpNode(name+"_zo_mm", new MatrixMatrixFCOp(), {prev_h, wh[3]});

	graph->pushOpNode(mmb_zg);
	graph->pushOpNode(mmb_zi);
	graph->pushOpNode(mmb_zf);
	graph->pushOpNode(mmb_zo);

	graph->pushOpNode(mm_zg);
	graph->pushOpNode(mm_zi);
	graph->pushOpNode(mm_zf);
	graph->pushOpNode(mm_zo);

	auto *zg_x = createTensor(graph, mmb_zg, name+"_zgx");
	auto *zi_x = createTensor(graph, mmb_zi, name+"_zix");
	auto *zf_x = createTensor(graph, mmb_zf, name+"_zfx");
	auto *zo_x = createTensor(graph, mmb_zo, name+"_zox");

	auto *zg_h = createTensor(graph, mm_zg, name+"_zgh");
	auto *zi_h = createTensor(graph, mm_zi, name+"_zih");
	auto *zf_h = createTensor(graph, mm_zf, name+"_zfh");
	auto *zo_h = createTensor(graph, mm_zo, name+"_zoh");

	auto *zg = addElementAdd(graph, zg_x, zg_h, name+"_zg_add");
	auto *zi = addElementAdd(graph, zi_x, zi_h, name+"_zi_add");
	auto *zf = addElementAdd(graph, zf_x, zf_h, name+"_zf_add");
	auto *zo = addElementAdd(graph, zo_x, zo_h, name+"_zo_add");

	zg = addTanh(graph, zg, name+"_zg_tanh");
	//zi = addSigmoid(graph, zi, name+"_zi_sigmoid");
	//zf = addSigmoid(graph, zf, name+"_zf_sigmoid");
	//zo = addSigmoid(graph, zo, name+"_zo_sigmoid");

	zi = addRelu(graph, zi, name+"_zi_sigmoid");
	zf = addRelu(graph, zf, name+"_zf_sigmoid");
	zo = addRelu(graph, zo, name+"_zo_sigmoid");

	auto *zfc = addElementMul(graph, zf, prev_c, name+"_zfc_mul");
	auto *zig = addElementMul(graph, zi, zg, name+"_zig_mul");
	*next_c = addElementAdd(graph, zfc, zig, name+"_nextc_add");
	//*next_c = addTanh(graph, *next_c, name+"_nextc_tanh");
	auto *next_ch = addTanh(graph, *next_c, name+"_nextch_tanh");
	
	*next_h = addElementMul(graph, zo, next_ch, name+"_nexth_mul"); 
}

void addLSTM(IRGraph *graph, vector<TensorNode*> &input, TensorNode *hidden_init, TensorNode* cell_init, int time_stamp, size_t hidden_size, vector<TensorNode*> &output, TensorNode **hidden, TensorNode **cell, string name="lstm") {
	vector<TensorNode*> wx;
	vector<TensorNode*> wh;
	vector<TensorNode*> b;
	vector<TensorNode*> h;
	for(int i=0; i<4; i++){
		auto *wx_tmp = createTensor(graph, {0, hidden_size}, name+"_wx"+to_string(i));
		auto *wh_tmp = createTensor(graph, {0, hidden_size}, name+"_wh"+to_string(i));
		auto *b_tmp = createTensor(graph, {hidden_size}, name+"_b"+to_string(i));
		wx.push_back(wx_tmp);
		wh.push_back(wh_tmp);
		b.push_back(b_tmp);
	}
	//wx->setTraining(1);
	//wh->setTraining(1);
	///b->setTraining(1);

	*hidden = hidden_init;
	*cell = cell_init;
	for(int i=0; i<time_stamp; i++){
		auto *next_c = new TensorNode();
		auto *next_h = new TensorNode();
		addLSTM_unit(graph, input[i], *hidden, *cell, wx, wh, b, &next_h, &next_c, name+"rnn_unit_"+to_string(i));
		*hidden = next_h;
		*cell = next_c;
		output.push_back(*hidden);
	}
}

vector<TensorNode*> NMT(IRGraph *graph, vector<TensorNode*> &enc_embd, vector<TensorNode*> &dec_embd, vector<TensorNode*> &cell_init, vector<TensorNode*> &hidden_init, int time_stamp, int hidden_size, int num_classes){
	// enc_embd ts
	// dec_embd ts
	// cell_init_enc num_layers
	// hidden_init_enc num_layers
	// encoder
	int num_layers = 2;
	vector<TensorNode*> hiddens;
	vector<TensorNode*> cells;
	vector<TensorNode*> input;
	for(int i=0; i<num_layers; i++){
		if(i==0){
			input = enc_embd;
		}
		vector<TensorNode*> output;
		TensorNode *hidden = new TensorNode();
		TensorNode *cell = new TensorNode();
		addLSTM(graph, input, hidden_init[i], cell_init[i], time_stamp, hidden_size, output, &hidden, &cell);
		hiddens.push_back(hidden);
		cells.push_back(cell);

		input = output;
	}
	// decoder	
	for(int i=0; i<num_layers; i++){
		if(i==0){
			input = dec_embd;
		}
		vector<TensorNode*> output;
		TensorNode *hidden = new TensorNode();
		TensorNode *cell = new TensorNode();
		addLSTM(graph, input,  hiddens[i], cells[i], time_stamp, hidden_size, output, &hidden, &cell);
		//hiddens.push_back(hidden);
		input = output;
	}

	vector<TensorNode*> ys;
	auto *w = createTensor(graph, {0, num_classes}, "fc_1_w");
	auto *b = createTensor(graph, {num_classes}, "fc_1_b");
	for(int i=0; i<time_stamp; i++){
		auto *mm = new OpNode("fc_out", new MatrixMatrixFCBiasOp(), {input[i], w, b});
		graph->pushOpNode(mm);
		auto *y = createTensor(graph, mm, "fc_1_y");
		ys.push_back(y);
	}

	return ys;
}

int main()
{
	IRGraph *lstmNMT = new IRGraph();
	int time_stamp = 3;
	int num_layers = 2;
	int hidden_size = 100;
	int num_classes = 12;
	vector<TensorNode*> enc_embd;
	vector<TensorNode*> dec_embd;
	vector<TensorNode*> cell_init;
	vector<TensorNode*> hidden_init;
	vector<TensorNode*> labels;
	for(int i=0; i<time_stamp; i++){
		auto *tmp1 = createTensor(lstmNMT, {8, 4096}, "enc_embd_ts"+to_string(i));
		auto *tmp2 = createTensor(lstmNMT, {8, 4096}, "dec_embd_ts"+to_string(i));
		auto *tmp3 = createTensor(lstmNMT, {8}, "label", DataType::Int32_t);
		enc_embd.push_back(tmp1);
		dec_embd.push_back(tmp2);
		labels.push_back(tmp3);
	}
	for(int i=0; i<num_layers; i++){
		auto *tmp1 = createTensor(lstmNMT, {8, 100}, "cell_init_layer"+to_string(i));
		auto *tmp2 = createTensor(lstmNMT, {8, 100}, "hidden_init_layer"+to_string(i));
		cell_init.push_back(tmp1);
		hidden_init.push_back(tmp2);
	}
	auto ys = NMT(lstmNMT, enc_embd, dec_embd, cell_init, hidden_init, time_stamp, hidden_size, num_classes);

	lstmNMT->findInOut();
	lstmNMT->updateTopology();

	lstmNMT->initTensorNodes();
	
	svgGen(lstmNMT, "lstmNMT.dot");
	
	//IRGraph *lstmtc = new IRGraph();
	//int time_stamp = 10;

	//vector<TensorNode*> data;
	//for(int i=0; i<time_stamp; i++){
	//	auto *tmp = createTensor(lstmtc, {8, 4096}, "data");
	//	data.push_back(tmp);
	//}
	////auto *data = createTensor(lstmtc, {8, 4096}, "data");
	//auto *hidden_init = createTensor(lstmtc, {8, 100}, "hidden_init");
	//auto *label = createTensor(lstmtc, {8}, "label", DataType::Int32_t);

	//auto *out = network(lstmtc, data, hidden_init, time_stamp); // 4096 -> 1024 -> 10
	//
	//auto *loss = addSoftmax(lstmtc, out, label, "softmax");

	//lstmtc->findInOut();
	//lstmtc->updateTopology();

	//cout << "update topology ok" << endl;

	//lstmtc->initTensorNodes();

	//cout << "init tensornodes ok" << endl;

	//// svgGen(lstmtc, "lstmtc_orig.dot");


	////label->require_grad = false;
	////data->require_grad = false;    
	////lstmtc->setTrainDataNodes(label, data);
	////lstmtc->addDisplayTensorNodes(loss);

	//svgGen(lstmtc, "lstmtc.dot");

	//Config config;
    //config.train_mode = true;
	////config.mpi = true;
    ////config.mpi_size = 4;
    //config.train_config.optimizer = "sgd";
    //config.train_config.train_data_file = "xxx";
    //config.train_config.label_bytes = BytesProto::ONE_BYTE_AS_INT;
    //config.train_config.data_bytes = BytesProto::FOUR_BYTES_AS_FLOAT;
    //config.train_config.train_data_samples = 50000;
    //// config.train_config.snapshot = 1000;
    //config.train_config.max_iters = 1000;
    //config.train_config.display = 50;
    //
	//config.compute_op_annotation = true;
    //// config.comm_op_annotation = true;
    //config.parallel_preference = COMM_SAVING;
    //// config.parallel_preference = MEM_SAVING;

	///*when benchmark enabled, disable emit some code*/
    //config.benchmark = true;
    ///* not do lowering for node liek FC, FCGrad etc.*/
    //config.enable_lowering = false;

	///* about parallel strategy*/
    //// config.force_data_parallel = true;
    //config.geneticalgo_opt_parallel = true;
    //// config.handcraft_parallel = true;

	//lstmtc->setConfig(config);

	//Engine engine(lstmtc);
	//engine.compile();

	//dotGen(lstmtc, "lstmtc_train.dot");
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

