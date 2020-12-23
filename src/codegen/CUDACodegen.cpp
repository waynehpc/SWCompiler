/*************************************************************************
        > File Name: CUDACodegen.cpp
        > Author: wayne
        > Mail:
        > Created Time: Fri 14 Feb 2020 08:48:18 AM UTC
 ************************************************************************/
#include "CUDACodegen.h"
#include "SWC.h"
#include <algorithm>
#include <map>
#include <vector>

namespace swc {
namespace codegen {

static const std::map<std::string, std::string> DTYPE_NCCL_DATATYPE_MAP = {
    {"int", "ncclInt32"}, {"float", "ncclFloat32"}, {"double", "ncclFloat64"}};

static const std::map<std::string, std::string> DTYPE_CUDNN_DATATYPE_MAP = {
    {"int", "CUDNN_DATA_INT32"},
    {"float", "CUDNN_DATA_FLOAT"},
    {"double", "CUDNN_DATA_DOUBLE"}};

template <typename T> static std::string getInitList(std::vector<T> &dims) {
    assert(dims.size() > 0 && "null vector");
    std::ostringstream os;
    os << "{";
    for (auto dim : dims) {
        os << dim << ", ";
    }
    auto str = os.str();
    return str.substr(0, str.length() - 2) + "}";
}

void CUDACodegen::initMakefileBuilder() {
    Codegen::initMakefileBuilder();
    // TODO
}

void CUDACodegen::emitCUDAInit() {
    // TODO create stream depending on number of device or config
    // one stream per device
    int ngpus = config_.ngpus_per_rank;
    UniqueName("ngpus");
    UniqueName("i");

    // ncclUniqueId id;
    // ncclComm_t comms[ngpus];
    UniqueName("id");
    UniqueName("comms");
    UniqueName("stream");

    writer_ << "int ngpus = " << ngpus << ";\n";

    std::vector<int> vec(ngpus);
    std::iota(vec.begin(), vec.end(), 0);

    writer_ << "int gpu_devs[" << ngpus << "] = " << getInitList(vec) << ";\n";

    writer_ << "ncclComm_t " << nccl_comms << "[" << ngpus << "];\n";

    writer_ << "checkNCCL(ncclCommInitAll(" << nccl_comms
            << ", ngpus, gpu_devs));\n";

    // cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
    writer_ << "cudaStream_t " << streams_ << "[" << ngpus << "];\n";
    writer_ << "for(int i=0; i<" << ngpus << "; i++) {\n";
    writer_.indentInc();
    writer_ << "checkCUDA(cudaSetDevice(i));\n";
    writer_ << "cudaStreamCreate(&" << streams_ << "[i]);\n\n";
    writer_.indentDec();
    writer_ << "}\n";
}

void CUDACodegen::allocateMemAddr() {
    SWLOG_DEBUG(4) << "begin allocateMemAddr...\n";
    for (int i = 0; i < graph_->topologyNum(); i++) {
        for (int j = 0; j < graph_->getNumInTopoLevel(i); j++) {
            auto node = graph_->getNodeInTopo(i, j);
            if (node->nodeType() == TENSOR_NODE) {
                auto *tnode = (TensorNode *)node;
                auto *tensor = tnode->getTensor();

                if (tensors_name_map_.count(tensor))
                    continue;

                std::string buf_name = UniqueName(tnode->name());
                size_t size = tensor->getSizeInBytes();
                Device dev = tnode->getLabel()->getDeviceLabel();

                SWLOG_DEBUG(1)
                    << "allocateMemAddr " << tnode->name() << " "
                    << "(" << tensor << ", " << size << ")"
                    << " as " << buf_name << " on dev(" << dev.rank << ", "
                    << static_cast<int>(dev.type) << ", " << dev.id << ")."
                    << "\n";

                auto *allocator = dev_allocator_map_.at(dev);
                if (!allocator) {
                    SWLOG_ERROR << "allocator" << static_cast<int>(dev.type)
                                << " " << dev.id << " not found\n";
                }
                uint64_t addr = allocator->allocate(tensor, size);
                std::string base = allocator->getBasePtrName();

                tensors_name_map_[tensor] = buf_name;
                tensors_offset_map_[tensor] = std::make_pair(base, addr);

                // TODO more concrete kinds
                if (dev.type == DeviceType::CPU) {
                    if (dev.rank == INT_MAX) {
                        host_parallel_tensors_.push_back(tensor);
                    } else {
                        host_tensors_.push_back(tensor);
                    }
                } else if (dev.type == DeviceType::GPU) {
                    if (dev.id == 0) {
                        master_gpu_tensors_.push_back(tensor);
                    } else if (dev.id == INT_MAX) {
                        parallel_gpu_tensors_.push_back(tensor);
                    }
                }
            }
        }
    }

    SWLOG_DEBUG(4) << "end allocateMemAddr...\n";
}

void CUDACodegen::emitVarDeclarations() {
    SWLOG_DEBUG(4) << "begin CUDACodegen::emitVarDeclarations...\n";
    // suppose a multiple multi gpus model
    /**
     * type=cpu rank=0   id=0:   single process / master process cpu mem tensor
     * type=cpu rank=INF id=*:   parallel cpu mem tensor
     * type=gpu rank=INF id=0:   master gpu of each node
     * type=gpu rank=0   id=INF: single process prallel gpu / master process
     * parallel gpu type=gpu rank=0   id=0:   single process master gpu
     *
     */
    for (auto m : mem_allocators_) {
        MemoryAllocator *allocator = m.get();
        std::string base = allocator->getBasePtrName();
        uint64_t size = allocator->getMemAllocated();
        if (size == 0)
            continue;
        writer_ << "char *" << base << ";\n";
    }

    if (p_mem_alllocator_->getMemAllocated()) {
        std::string base = p_mem_alllocator_->getBasePtrName();
        writer_ << "char *" << base << ";\n";
    }

    if (p_gpumem_allocator_->getMemAllocated()) {
        // auto dev = p_gpumem_allocator_->getDevice();
        std::string base = p_gpumem_allocator_->getBasePtrName();
        writer_ << "char *" << base << "[" << ngpus_ << "];\n";
    }

    for (auto tensor : host_tensors_) {
        std::string dtype = getTypeString(tensor);
        writer_ << dtype << " *" << tensors_name_map_.at(tensor) << ";\n";
    }

    for (auto tensor : host_parallel_tensors_) {
        std::string dtype = getTypeString(tensor);
        writer_ << dtype << " *" << tensors_name_map_.at(tensor) << ";\n";
    }

    for (auto tensor : master_gpu_tensors_) {
        std::string dtype = getTypeString(tensor);
        writer_ << dtype << " *" << tensors_name_map_.at(tensor) << ";\n";
    }

    for (auto tensor : parallel_gpu_tensors_) {
        std::string dtype = getTypeString(tensor);
        writer_ << dtype << " *" << tensors_name_map_.at(tensor) << "["
                << ngpus_ << "];\n";
    }

    writer_ << "\n";

    SWLOG_DEBUG(4) << "end CUDACodegen::emitVarDeclarations...\n";
}

void CUDACodegen::emitMemAllocations() {
    SWLOG_DEBUG(4) << "begin CUDACodegen::emitMemAllocations...\n";
    // std::string dtype = this->dtype();
    for (auto m : mem_allocators_) {
        MemoryAllocator *allocator = m.get();
        auto dev = allocator->getDevice();
        std::string base = allocator->getBasePtrName();
        uint64_t size = allocator->getMemAllocated();
        if (size == 0)
            continue;

        emitMemAllocation(base, size, dev);
    }

    if (p_mem_alllocator_->getMemAllocated()) {

        auto dev = p_mem_alllocator_->getDevice();
        std::string base = p_mem_alllocator_->getBasePtrName();
        uint64_t size = p_mem_alllocator_->getMemAllocated();

        emitMemAllocation(base, size, dev);
    }

    if (p_gpumem_allocator_->getMemAllocated()) {
        auto dev = p_gpumem_allocator_->getDevice();
        std::string base = p_gpumem_allocator_->getBasePtrName();
        uint64_t size = p_mem_alllocator_->getMemAllocated();
        emitMultiGPUMemAlloc(base, size, dev);
    }

    writer_ << "\n";
    SWLOG_DEBUG(4) << "end CUDACodegen::emitMemAllocations...\n";
}

void CUDACodegen::emitMultiGPUMemAlloc(std::string base, uint64_t size,
                                       Device &dev) {
    writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
    writer_.indentInc();
    writer_ << "checkCUDA(cudaSetDevice(i));\n";
    // size: size in bytes
    writer_ << "checkCUDA(cudaMalloc(" << base << "[i]" << size << "));\n";
    writer_.indentDec();
    writer_ << "}\n";
}

void CUDACodegen::emitMemFree() {
    SWLOG_DEBUG(4) << "begin CUDACodegen::emitMemFree...\n";

    for (auto m : mem_allocators_) {
        MemoryAllocator *allocator = m.get();
        auto dev = allocator->getDevice();
        std::string base = allocator->getBasePtrName();
        uint64_t size = allocator->getMemAllocated();
        if (size == 0)
            continue;
        emitMemFree(base, dev);
    }
    /*
    //free device buffers
    for (int i = 0; i < nDev; ++i) {
        checkCUDA(cudaSetDevice(i));
        checkCUDA(cudaFree(sendbuff[i]));
        checkCUDA(cudaFree(recvbuff[i]));
    }*/
    if (p_gpumem_allocator_->getMemAllocated()) {
        // auto dev = p_gpumem_allocator_->getDevice();
        std::string base = p_gpumem_allocator_->getBasePtrName();

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";
        // size: size in bytes
        writer_ << "checkCUDA(cudaFree(" << base << "[i]));\n";
        writer_.indentDec();
        writer_ << "}\n";
    }

    SWLOG_DEBUG(4) << "end CUDACodegen::emitMemFree...\n";
}

void CUDACodegen::emitMemFree(std::string name, Device dev) {
    switch (dev.type) {
    case DeviceType::CPU:
        writer_ << "free(" << name << ");\n";
        break;
    case DeviceType::GPU:
        writer_ << "\n";
        writer_ << "cudaSetDevice(" << dev.id << ");\n";
        writer_ << "cudaFree(" << name << ");\n";
        break;
    default:
        SWLOG_ERROR << "Unknown DeviceType\n";
        break;
    }
}

void CUDACodegen::emitTensorAddresses() {
    SWLOG_DEBUG(4) << "begin CUDACodegen::emitTensorAddresses...\n";

    std::string base;
    uint64_t offset;
    for (auto &tensor : host_tensors_) {
        auto name = tensors_name_map_.at(tensor);
        std::string dtype = getTypeString(tensor);
        std::tie(base, offset) = tensors_offset_map_.at(tensor);
        writer_ << name << " = reinterpret_cast<" << dtype << "*>(" << base
                << " + " << offset << ");\n";
    }

    // writer_ << "checkCUDA(cudaSetDevice(0));\n";
    for (auto &tensor : master_gpu_tensors_) {
        auto name = tensors_name_map_.at(tensor);
        std::string dtype = getTypeString(tensor);
        std::tie(base, offset) = tensors_offset_map_.at(tensor);

        // size: size in bytes
        writer_ << name << " = reinterpret_cast<" << dtype << "*>(" << base
                << " + " << offset << ");\n";
    }

    writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
    writer_.indentInc();
    writer_ << "checkCUDA(cudaSetDevice(i));\n";

    for (auto &tensor : parallel_gpu_tensors_) {
        auto name = tensors_name_map_.at(tensor);
        std::string dtype = getTypeString(tensor);
        std::tie(base, offset) = tensors_offset_map_.at(tensor);

        // size: size in bytes
        writer_ << name << "[i] = reinterpret_cast<" << dtype << "*>(" << base
                << "[i] + " << offset << ");\n";
    }

    writer_.indentDec();
    writer_ << "}\n";

    SWLOG_DEBUG(4) << "end CUDACodegen::emitTensorAddresses...\n";
}

void CUDACodegen::emitExecute() {
    SWLOG_DEBUG(4) << "begin CUDACodegen::emitExecute ...\n";
    if (config_.train_mode) {
        TensorNode *label = graph_->getTrainLabelNode();
        TensorNode *data = graph_->getTrainDataNode();
        std::string label_var;
        std::string data_var;

        // in benchmark mode, we suppose parallel io is supported, then,
        // original label and data node may be eliminated in
        // IRGraph::elimRedundantScatter()
        if (!config_.benchmark) {
            if (!tensors_name_map_.count(label->getTensor())) {
                SWLOG_DEBUG(10) << "label tensor " << label->name() << " "
                                << label->getTensor() << " not in map ...\n";
                exit(0);
            }
            if (!tensors_name_map_.count(data->getTensor())) {
                SWLOG_DEBUG(10) << "data tensor " << data->name() << " "
                                << data->getTensor() << " not in map ...\n";
                exit(0);
            }

            label_var = tensors_name_map_.at(label->getTensor());
            data_var = tensors_name_map_.at(data->getTensor());
        }

        TrainingConfig tconfig = config_.train_config;
        tconfig.batch = data->getDims()[0];

        if (!config_.benchmark) {
            writer_ << "std::string train_data_file = \""
                    << tconfig.train_data_file << "\";\n";
            writer_ << "DataLoader loader("
                    << "train_data_file, "
                    << getBytesProtoString(tconfig.label_bytes) << ", "
                    << getBytesProtoString(tconfig.data_bytes) << ", "
                    << tconfig.max_epoch << ", " << tconfig.train_data_samples
                    << ", " << getInitializerString(label->getDims()) << ", "
                    << getInitializerString(data->getDims()) << ");\n";
        }

        emitcudnnDescs();

        size_t max_iter = tconfig.max_iters == 0
                              ? (tconfig.train_data_samples *
                                 tconfig.max_epoch / tconfig.batch)
                              : tconfig.max_iters;
        writer_ << "size_t max_iter = " << max_iter << ";\n";
        writer_ << "size_t iter = 0; "
                << "\n\n";

        // writer_ << "gettimeofday(&ts, NULL);\n";

        writer_ << "while(iter < max_iter) {\n";
        writer_.indentInc();

        writer_ << "gettimeofday(&ts, NULL);\n";

        if (!config_.benchmark) {
            writer_ << "loader.next(" << label_var << ", " << data_var
                    << ");\n";
        } else {
            writer_ << "// batch load disabled in benchmark mode\n";
        }
    }

    emitFuncCalls();

    if (config_.train_mode) {

        writer_ << "\n";
        writer_ << "iter++;\n";

        // if do benchmark, comment save snapshot and print
        if (!config_.benchmark) {

            if (config_.train_config.snapshot) {
                emitSaveSnapshot();
            }

            if (config_.train_config.display) {
                emitPrintGraphOutputs();
            }
        } else {
            writer_ << "// emitSaveSnapshot() and emitPrintGraphOutputs() "
                       "disable in benchmark mode\n";
        }

        writer_ << "gettimeofday(&te, NULL);\n";
        writer_ << "double time = TIME_MS(ts, te);\n";

        writer_ << R"(cout << "time cost (ms)\n" << time << endl;)"
                << "\n";

        writer_.indentDec();
        writer_ << "} //while\n";

        // writer_ << "gettimeofday(&te, NULL);\n";
        // writer_ << "double time = TIME_MS(ts, te);\n";

        // writer_ << R"(cout << "time cost (ms)\n" << time << endl;)" << "\n";
    }

    SWLOG_DEBUG(4) << "end CUDACodegen::emitExecute ...\n";
}

void CUDACodegen::emitcudnnDescs() {
    writer_ << "cublasHandle_t " << cublas_handles_ << "[" << ngpus_ << "];\n";
    writer_ << "cudnnHandle_t " << cudnn_handles_ << "[" << ngpus_ << "];\n";

    writer_ << "cudnnTensorDescriptor_t " << srcDesc_ << ", " << dstDesc_
            << ", " << biasDesc_ << ", " << srcGradDesc_ << ", " << dstGradDesc_
            << ", " << biasGradDesc_ << ";\n"
            << "cudnnFilterDescriptor_t " << filterDesc_ << ", "
            << filterGradDesc_ << ";\n"
            << "cudnnConvolutionDescriptor_t " << convDesc_ << ";\n"
            << "cudnnPoolingDescriptor_t " << poolingDesc_ << ";\n"
            << "cudnnActivationDescriptor_t " << activDesc_ << ";\n"
            << "cudnnLRNDescriptor_t " << lrnDesc_ << ";\n";

    writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
    writer_.indentInc();
    writer_ << "checkCUDNN( cudnnCreate(&" << cudnn_handles_ << "[i]) );\n";
    writer_ << "cudnnSetStream(" << cudnn_handles_ << "[i], " << streams_
            << "[i]);\n";

    writer_ << "checkCUBLAS( cublasCreate(&" << cublas_handles_ << "[i]) );\n";
    writer_ << "cublasSetStream(" << cublas_handles_ << "[i], " << streams_
            << "[i]);\n";
    writer_.indentDec();
    writer_ << "}\n";

    writer_ << "checkCUDNN( cudnnCreateTensorDescriptor(&" << srcDesc_
            << ") );\n"
            << "checkCUDNN( cudnnCreateTensorDescriptor(&" << dstDesc_
            << ") );\n"
            << "checkCUDNN( cudnnCreateTensorDescriptor(&" << biasDesc_
            << ") );\n"
            << "checkCUDNN( cudnnCreateFilterDescriptor(&" << filterDesc_
            << ") );\n"
            << "checkCUDNN( cudnnCreateTensorDescriptor(&" << srcGradDesc_
            << ") );\n"
            << "checkCUDNN( cudnnCreateTensorDescriptor(&" << dstGradDesc_
            << ") );\n"
            << "checkCUDNN( cudnnCreateTensorDescriptor(&" << biasGradDesc_
            << ") );\n"
            << "checkCUDNN( cudnnCreateFilterDescriptor(&" << filterGradDesc_
            << ") );\n"
            << "checkCUDNN( cudnnCreateConvolutionDescriptor(&" << convDesc_
            << ") );\n"
            << "int " << convAlgo_
            << " = 0;\n" /*CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM=0*/
            << "checkCUDNN( cudnnCreatePoolingDescriptor(&" << poolingDesc_
            << ") );\n"
            << "checkCUDNN( cudnnCreateActivationDescriptor(&" << activDesc_
            << ") );\n"
            << "checkCUDNN( cudnnCreateLRNDescriptor(&" << lrnDesc_ << ") );\n";

    std::string dataType = dtype();

    writer_ << "// cudnnDataType_t cudnnDataType = "
            << DTYPE_CUDNN_DATATYPE_MAP.at(dataType) << ";\n";
    writer_ << "// cudnnTensorFormat_t cudnnTensorFormat = "
            << "CUDNN_TENSOR_NHWC"
            << ";\n";

    for (int i = 0; i < graph_->topologyNum(); i++) {
        for (int j = 0; j < graph_->getNumInTopoLevel(i); j++) {
            auto irnode = graph_->getNodeInTopo(i, j);
            if (irnode->nodeType() != OP_NODE) {
                continue;
            }

            // opnode
            // auto *node = (OpNode *)irnode;
            // writer_ << "\n";
            // writer_ << "// topology(" << i << ", " << j << "): " <<
            // node->name()
            //         << " : " << node->getOpName() << "\n";

            // // emit or return
            // emitcudnnDescriptor(node);
        }
    }
}

void CUDACodegen::emitFuncCalls() {
    for (int i = 0; i < graph_->topologyNum(); i++) {
        for (int j = 0; j < graph_->getNumInTopoLevel(i); j++) {
            auto irnode = graph_->getNodeInTopo(i, j);
            if (irnode->nodeType() != OP_NODE) {
                continue;
            }

            // opnode
            auto *node = (OpNode *)irnode;
            writer_ << "\n";
            writer_ << "// topology(" << i << ", " << j << "): " << node->name()
                    << " : " << node->getOpName() << "\n";

            dispatchOpNode(node);
        }
    }
}

void CUDACodegen::dispatchOpNode(OpNode *op) {
    // if (!op->runable())
    //     return;

    auto *label = op->getLabel();       // Label*
    auto dev = label->getDeviceLabel(); // Device

    SWLOG_DEBUG(4) << "dispatchOpNode " << op->name() << " " << dev.toString()
                   << "\n";
    // writer_ << "// " << op->name() << " : " << op->getOpName() << "\n";

    if (dynamic_cast<ScatterOp *>(op->getOp())) {
        dispatchScatterOp(op);
    } else if (dynamic_cast<GatherOp *>(op->getOp())) {
        dispatchGatherOp(op);
    } else if (dynamic_cast<TransformOp *>(op->getOp())) {
        dispatchTransformOp(op);
    } else {
        dispatchDevOpNode(op);
        // if(dev.type == DeviceType::CPU) {
        //     dispatchHostOpNode(op);
        // }else if(dev.type == DeviceType::GPU) {
        //     dispatchDevOpNode(op);
        // }
    }
}

void CUDACodegen::dispatchScatterOp(OpNode *op) {
    SWLOG_DEBUG(2) << "genKernelCall for " << op->name() << "\n";

    auto *scatter = dynamic_cast<ScatterOp *>(op->getOp());
    auto *in = (TensorNode *)op->getParentNode(0);
    auto *in_tensor = in->getTensor();
    auto *out = (TensorNode *)op->getChildNode(0);
    auto *out_tensor = out->getTensor();

    std::string fname = tensors_name_map_.at(in_tensor);
    std::string tname = tensors_name_map_.at(out_tensor);
    Device fdev = in->getLabel()->getDeviceLabel();
    Device tdev = out->getLabel()->getDeviceLabel();

    int axis = scatter->getAxis();
    int degree = scatter->getDegree();
    assert(degree == ngpus_ && "scatter wrong degree");

    writer_ << "// scatter(" << axis << ") (" << fdev.toString() << ", "
            << fname << ")->(" << tdev.toString() << ", " << tname << ")\n";

    if (config_.comm_op_annotation) {
        writer_ << "// !!! communication call emit disabled because "
                   "comm_op_annotation=true\n";
        return;
    }

    // const std::vector<size_t>
    auto idims = in_tensor->getDims();
    auto odims = out_tensor->getDims();
    int n = in_tensor->getNDim();
    std::string dtype = getTypeString(in_tensor);

    if (axis == -1) {
        size_t count = out_tensor->size();
        // ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff,
        // size_t count, ncclDataType_t datatype,
        // int root, ncclComm_t comm, cudaStream_t stream)
        writer_ << "checkNCCL(ncclGroupStart());\n";
        writer_ << "for(int i=0; i<" << ngpus_ << "; i++)\n";
        writer_.indentInc();
        writer_ << "checkNCCL(ncclBroadcast("
                // << "(const void*)" << fname << "[0], "
                << "(const void*)" << fname << ", "
                << "(void*)" << tname << "[i], " << count << ", "
                << DTYPE_NCCL_DATATYPE_MAP.at(dtype) << ", "
                << "0, " << nccl_comms << "[i], " << streams_ << "[i]"
                << "));\n";
        writer_.indentDec();
        writer_ << "checkNCCL(ncclGroupEnd());\n";
        writer_ << "cuSyncStreams(" << streams_ << ", " << ngpus_ << ");\n";

        return;
    }

    size_t count = 1, len = 1, stride = 1;
    for (int i = 0; i < axis; i++)
        count *= idims[i];
    for (int i = axis; i < n; i++)
        stride *= idims[i];
    len = stride / degree;

    int slicesz = count * len;
    writer_ << "cuScatter(" << fname << ", " << tname << ", "
            << "0, " << slicesz << ", " << count << ", " << len << ", "
            << stride << ", " << degree << ", " << streams_ << ");\n";
}

void CUDACodegen::dispatchGatherOp(OpNode *op) {
    SWLOG_DEBUG(2) << "genKernelCall for " << op->name() << "\n";

    auto *gather = dynamic_cast<GatherOp *>(op->getOp());
    auto *in = (TensorNode *)op->getParentNode(0);
    auto *in_tensor = in->getTensor();
    auto *out = (TensorNode *)op->getChildNode(0);
    auto *out_tensor = out->getTensor();

    std::string fname = tensors_name_map_.at(in_tensor);
    std::string tname = tensors_name_map_.at(out_tensor);
    Device fdev = in->getLabel()->getDeviceLabel();
    Device tdev = out->getLabel()->getDeviceLabel();

    int axis = gather->getAxis();
    int degree = gather->getDegree();
    assert(degree == ngpus_ && "gather wrong degree");

    writer_ << "// gather(" << axis << ") (" << fdev.toString() << ", " << fname
            << ")->(" << tdev.toString() << ", " << tname << ")\n";

    if (config_.comm_op_annotation) {
        writer_ << "// !!! communication call emit disabled because "
                   "comm_op_annotation=true\n";
        return;
    }

    // const std::vector<size_t>
    auto idims = in_tensor->getDims();
    auto odims = out_tensor->getDims();
    int n = in_tensor->getNDim();
    std::string dtype = getTypeString(in_tensor);

    if (axis == -2) {
        size_t count = out_tensor->size();
        // ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff,
        // size_t count, ncclDataType_t datatype,
        // int root, ncclComm_t comm, cudaStream_t stream)
        writer_ << "checkNCCL(ncclGroupStart());\n";
        writer_ << "for(int i=0; i<" << ngpus_ << "; i++)\n";
        writer_.indentInc();
        writer_ << "checkNCCL(ncclReduce("
                << "(void*)" << fname << "[i], "
                << "(const void*)" << tname << ", " << count << ", "
                << DTYPE_NCCL_DATATYPE_MAP.at(dtype) << ", "
                << "ncclSum, " << nccl_comms << "[i], "
                << "0, " << streams_ << "[i]"
                << "));\n";
        writer_.indentDec();
        writer_ << "checkNCCL(ncclGroupEnd());\n";
        writer_ << "cuSyncStreams(" << streams_ << ", " << ngpus_ << ");\n";

        return;
    }

    size_t count = 1, len = 1, stride = 1;
    for (int i = 0; i < axis; i++)
        count *= idims[i];
    for (int i = axis; i < n; i++)
        stride *= idims[i];
    len = stride / degree;

    int slicesz = count * len;

    writer_ << "cuGather(" << fname << ", " << tname << ", "
            << "0, " << slicesz << ", " << count << ", " << len << ", "
            << stride << ", " << degree << ", " << streams_ << ");\n";
}

void CUDACodegen::dispatchTransformOp(OpNode *node) {
    SWLOG_DEBUG(2) << "genKernelCall for " << node->name() << "\n";

    auto *transform = dynamic_cast<TransformOp *>(node->getOp());
    int ki = transform->getPreAxis();
    int ko = transform->getPostAxis();
    int degree = transform->getDegree();

    auto *in = ((TensorNode *)node->getParentNode(0));
    auto *in_tensor = in->getTensor();
    auto *out = ((TensorNode *)node->getChildNode(0));
    auto *out_tensor = out->getTensor();

    std::string fname = tensors_name_map_[in_tensor];
    std::string tname = tensors_name_map_[out_tensor];
    Device fdev = in->getLabel()->getDeviceLabel();
    Device tdev = out->getLabel()->getDeviceLabel();

    std::string dtype = getTypeString(in_tensor);
    // size_t in_count = in_tensor->size();
    // size_t out_count = out_tensor->size();

    // std::vector<size_t>
    auto idims = in_tensor->getDims();
    auto odims = out_tensor->getDims();
    int n = in_tensor->getNDim();

    writer_ << "// transform(" << ki << ", " << ko << ") (" << fdev.toString()
            << ", " << fname << ", " << getInitList(idims) << ", "
            << ")->(" << tdev.toString() << ", " << tname << ", "
            << getInitList(odims) << ")\n";

    if (config_.comm_op_annotation) {
        writer_ << "// !!! communication call emit disabled because "
                   "comm_op_annotation=true\n";
        return;
    }

    if (ki >= 0 && ko >= 0) {
        assert((ki >= 0 && ki < n && ko >= 0 && ko < n) && "illegal strategy");
        assert((ki != ko) && "same in/out strategy");

        /* e.g.
        void cuGatherScatter( T** sbuf, T** rbuf,  int slicesz,
            int scount, int slen, int sstride,
            int rcount, int rlen, int rstride,
            int ndev, cudaStream_t* streams)
        cuGatherScatter(buf_t0, recvbuf_t1, mini_slicesz,
        scount, slen, sstride,
        rcount, rlen, rstride,
        p, s);
        */
        int scount = 1, sstride = 1, slen = 1;
        for (int i = 0; i < ko; i++)
            scount *= idims[i];
        for (int i = ko; i < n; i++)
            sstride *= idims[i];
        slen = sstride / degree;

        int rcount = 1, rstride = 1, rlen = 1;
        for (int i = 0; i < ki; i++)
            rcount *= odims[i];
        for (int i = ki; i < n; i++)
            rstride *= odims[i];
        rlen = rstride / degree;

        assert(slen * scount == rcount * rlen && "transform wrong dimension");
        int mini_slicesz = scount * slen;

        writer_ << "cuGatherScatter(" << fname << ", " << tname << ", "
                << mini_slicesz << ", " << scount << ", " << slen << ", "
                << sstride << ", " << rcount << ", " << rlen << ", " << rstride
                << ", " << degree << ", " << streams_ << ");\n";

        return;
    }

    // reduce-scatter
    if (ki == -2 && ko >= 0) {

        int count = 1, stride = 1, len = 1;
        for (int i = 0; i < ko; i++)
            count *= idims[i];
        for (int i = ko; i < n; i++)
            stride *= idims[i];
        len = stride / degree;

        int slicesz = count * len;

        if (ko == 0) {
            writer_ << "checkNCCL(ncclGroupStart());\n";
            writer_ << "for(int i=0; i<" << ngpus_ << "; i++)\n";
            writer_.indentInc();
            writer_ << "checkNCCL(ncclReduceScatter("
                    << "(const void*)" << fname << "[i], "
                    << "(void*)" << tname << "[i], " << slicesz << ", "
                    << DTYPE_NCCL_DATATYPE_MAP.at(dtype) << ", "
                    << "ncclSum, " << nccl_comms << "[i], " << streams_ << "[i]"
                    << "));\n";
            writer_.indentDec();
            writer_ << "checkNCCL(ncclGroupEnd());\n";
            writer_ << "cuSyncStreams(" << streams_ << ", " << ngpus_ << ");\n";

            return;
        }

        writer_ << "cuReduceScatter(" << fname << ", " << tname << ", "
                << slicesz << ", " << count << ", " << len << ", " << stride
                << ", " << degree << ", " << nccl_comms << ", " << streams_
                << ");\n";

        return;
    }

    if (ki == -2 && ko == -1) {
        // AllReduce
        size_t count = out_tensor->size();

        writer_ << "checkNCCL(ncclGroupStart());\n";
        writer_ << "for(int i=0; i<" << ngpus_ << "; i++)\n";
        writer_.indentInc();
        writer_ << "checkNCCL(ncclAllReduce("
                << "(const void*)" << fname << "[i], "
                << "(void*)" << tname << "[i], " << count << ", "
                << DTYPE_NCCL_DATATYPE_MAP.at(dtype) << ", "
                << "ncclSum, " << nccl_comms << "[i], " << streams_ << "[i]"
                << "));\n";
        writer_.indentDec();
        writer_ << "checkNCCL(ncclGroupEnd());\n";
        writer_ << "cuSyncStreams(" << streams_ << ", " << ngpus_ << ");\n";

        return;
    }

    if (ki == -1 && ko >= 0) {
        // memPack
        int count = 1, stride = 1, len = 1;
        for (int i = 0; i < ko; i++)
            count *= idims[i];
        for (int i = ko; i < n; i++)
            stride *= idims[i];
        len = stride / degree;

        int slicesz = count * len;

        writer_ << "cuMempack(" << fname << ", " << tname << ", " << slicesz
                << ", " << count << ", " << len << ", " << stride << ", "
                << degree << ", " << streams_ << ");\n";

        return;
    }

    if (ki >= 0 && ko == -1) {
        // AllGather
        int count = 1, stride = 1, len = 1;
        for (int i = 0; i < ki; i++)
            count *= odims[i];
        for (int i = ki; i < n; i++)
            stride *= odims[i];
        len = stride / degree;

        int slicesz = count * len;

        if (ki == 0) {
            writer_ << "checkNCCL(ncclGroupStart());\n";
            writer_ << "for(int i=0; i<" << ngpus_ << "; i++)\n";
            writer_.indentInc();
            writer_ << "checkNCCL(ncclAllGather("
                    << "(const void*)" << fname << "[i], "
                    << "(void*)" << tname << "[i], " << slicesz << ", "
                    << DTYPE_NCCL_DATATYPE_MAP.at(dtype) << ", " << nccl_comms
                    << "[i], " << streams_ << "[i]"
                    << "));\n";
            writer_.indentDec();
            writer_ << "checkNCCL(ncclGroupEnd());\n";
            writer_ << "cuSyncStreams(" << streams_ << ", " << ngpus_ << ");\n";
            return;
        }

        writer_ << "cuAllGather(" << fname << ", " << tname << ", " << slicesz
                << ", " << count << ", " << len << ", " << stride << ", "
                << degree << ", " << nccl_comms << ", " << streams_ << ");\n";

        return;
    }

    SWLOG_ERROR << "error, unimplemented transform code\n";
    exit(0);
}

void CUDACodegen::dispatchHostOpNode(OpNode *op) {}

void CUDACodegen::dispatchDevOpNode(OpNode *op) { emitFuncCallCUDA(op); }

/*
void CUDACodegen::emitMemcpyH2D() {

}

void CUDACodegen::emitMemcpyD2H() {

}


void CUDACodegen::emitcudnnDescriptor(OpNode *op) {

}
*/

void CUDACodegen::emitFuncCallCUDA(OpNode *op) {
    if (config_.compute_op_annotation) {
        writer_ << "/*\n";
    }

    std::string dtype_flag = dtype();

    Label *oplabel = op->getLabel();
    SWLOG_DEBUG(2) << "genKernelCall for " << op->name() << " : "
                   << oplabel->getTypeNameLabel() << "\n";

    // TODO assert legal dimensions
    if ((oplabel->getTypeNameLabel()) == "Conv2d") {
        auto *src = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *filter = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *bias = ((TensorNode *)op->getParentNode(2))->getTensor();
        auto *dst = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto *conv_op = (Conv2dOp *)op->getOp();
        auto kernels = conv_op->getKernels();
        auto strides = conv_op->getStrides();
        auto pads = conv_op->getPads();
        // auto group = conv_op->getGroup(); // unsupported yet

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        auto idims = src->getDims();
        auto odims = dst->getDims();

        writer_ << "cudnnConv2d(" << tensors_name_map_[src] << "[i], "
                << tensors_name_map_[dst] << "[i], "
                << tensors_name_map_[filter] << "[i], "
                << tensors_name_map_[bias] << "[i], " << cudnn_handles_
                << "[i], " << srcDesc_ << ", " << dstDesc_ << ", "
                << filterDesc_ << ", " << biasDesc_ << ", " << convDesc_ << ", "
                << convAlgo_ << ", " << getInitList(idims) << ", "
                << getInitList(odims) << ", " << getInitList(kernels) << ", "
                << getInitList(strides) << ", " << getInitList(pads) << ");\n";

        writer_.indentDec();
        writer_ << "}\n";
    }
    if ((oplabel->getTypeNameLabel()) == "Conv2dGrad") {
        auto *src = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *filter = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *dstG = ((TensorNode *)op->getParentNode(2))->getTensor();

        auto *srcG = ((TensorNode *)op->getChildNode(0))->getTensor();
        auto *filterG = ((TensorNode *)op->getChildNode(1))->getTensor();
        auto *biasG = ((TensorNode *)op->getChildNode(2))->getTensor();

        auto *conv_op = (Conv2dGradOp *)op->getOp();
        auto kernels = conv_op->getKernels();
        auto strides = conv_op->getStrides();
        auto pads = conv_op->getPads();
        // auto group = conv_op->getGroup();

        auto idims = src->getDims();
        auto odims = dstG->getDims();

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        writer_ << "cudnnConv2dGrad(" << tensors_name_map_[src] << "[i], "
                << tensors_name_map_[dstG] << "[i], "
                << tensors_name_map_[filter] << "[i], "
                << tensors_name_map_[srcG] << "[i], "
                << tensors_name_map_[filterG] << "[i], "
                << tensors_name_map_[biasG] << "[i], " << cudnn_handles_
                << "[i], " << srcDesc_ << ", " << dstGradDesc_ << ", "
                << filterDesc_ << ", " << srcGradDesc_ << ", "
                << filterGradDesc_ << ", " << biasGradDesc_ << ", " << convDesc_
                << ", " << getInitList(idims) << ", " << getInitList(odims)
                << ", " << getInitList(kernels) << ", " << getInitList(strides)
                << ", " << getInitList(pads) << ");\n";

        writer_.indentDec();
        writer_ << "}\n";
    }
    if ((oplabel->getTypeNameLabel()) == "MaxPool") {
        auto *src = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *dst = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto *pool_op = (MaxPoolOp *)op->getOp();
        auto kernels = pool_op->getKernels();
        auto strides = pool_op->getStrides();
        auto pads = pool_op->getPads();

        auto idims = src->getDims();
        auto odims = dst->getDims();

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        std::string poolingMode = "CUDNN_POOLING_MAX";

        writer_ << "cudnnPooling(" << tensors_name_map_[src] << "[i], "
                << tensors_name_map_[dst] << "[i], " << cudnn_handles_
                << "[i], " << srcDesc_ << ", " << dstDesc_ << ", "
                << poolingDesc_ << ", " << poolingMode << ", "
                << getInitList(idims) << ", " << getInitList(odims) << ", "
                << getInitList(kernels) << ", " << getInitList(strides) << ", "
                << getInitList(pads) << ");\n";

        writer_.indentDec();
        writer_ << "}\n";
    }
    if ((oplabel->getTypeNameLabel()) == "MaxPoolGrad") {
        auto *src = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *dst = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *dstG = ((TensorNode *)op->getParentNode(2))->getTensor();

        auto *srcG = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto *pool_op = (MaxPoolOp *)op->getOp();
        auto kernels = pool_op->getKernels();
        auto strides = pool_op->getStrides();
        auto pads = pool_op->getPads();

        auto idims = src->getDims();
        auto odims = dst->getDims();

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        std::string poolingMode = "CUDNN_POOLING_MAX";

        writer_ << "cudnnPoolingGrad(" << tensors_name_map_[src] << "[i], "
                << tensors_name_map_[dst] << "[i], " << tensors_name_map_[dstG]
                << "[i], " << tensors_name_map_[srcG] << "[i], "
                << cudnn_handles_ << "[i], " << srcDesc_ << ", " << dstDesc_
                << ", " << dstGradDesc_ << ", " << srcGradDesc_ << ", "
                << poolingDesc_ << ", " << poolingMode << ", "
                << getInitList(idims) << ", " << getInitList(odims) << ", "
                << getInitList(kernels) << ", " << getInitList(strides) << ", "
                << getInitList(pads) << ");\n";

        writer_.indentDec();
        writer_ << "}\n";
    }
    if ((oplabel->getTypeNameLabel()) == "AveragePool") {
        auto *src = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *dst = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto *pool_op = (AvgPoolOp *)op->getOp();
        auto kernels = pool_op->getKernels();
        auto strides = pool_op->getStrides();
        auto pads = pool_op->getPads();

        auto idims = src->getDims();
        auto odims = dst->getDims();

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        std::string poolingMode = "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING";

        writer_ << "cudnnPooling(" << tensors_name_map_[src] << "[i], "
                << tensors_name_map_[dst] << "[i], " << cudnn_handles_
                << "[i], " << srcDesc_ << ", " << dstDesc_ << ", "
                << poolingDesc_ << ", " << poolingMode << ", "
                << getInitList(idims) << ", " << getInitList(odims) << ", "
                << getInitList(kernels) << ", " << getInitList(strides) << ", "
                << getInitList(pads) << ");\n";

        writer_.indentDec();
        writer_ << "}\n";
    }
    if ((oplabel->getTypeNameLabel()) == "AveragePoolGrad") {
        auto *src = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *dst = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *dstG = ((TensorNode *)op->getParentNode(2))->getTensor();

        auto *srcG = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto *pool_op = (AvgPoolGradOp *)op->getOp();
        auto kernels = pool_op->getKernels();
        auto strides = pool_op->getStrides();
        auto pads = pool_op->getPads();

        auto idims = src->getDims();
        auto odims = dst->getDims();

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        std::string poolingMode = "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING";

        writer_ << "cudnnPoolingGrad(" << tensors_name_map_[src] << "[i], "
                << tensors_name_map_[dst] << "[i], " << tensors_name_map_[dstG]
                << "[i], " << tensors_name_map_[srcG] << "[i], "
                << cudnn_handles_ << "[i], " << srcDesc_ << ", " << dstDesc_
                << ", " << dstGradDesc_ << ", " << srcGradDesc_ << ", "
                << poolingDesc_ << ", " << poolingMode << ", "
                << getInitList(idims) << ", " << getInitList(odims) << ", "
                << getInitList(kernels) << ", " << getInitList(strides) << ", "
                << getInitList(pads) << ");\n";

        writer_.indentDec();
        writer_ << "}\n";
    }

    if ((oplabel->getTypeNameLabel()) == "MatrixTanh") {
        // TODO assert
        auto *src = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *dst = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto idims = src->getDims();
        if (idims.size() == 2) {
            idims = {idims[0], 1, 1, idims[1]};
        }

        auto activMode = "CUDNN_ACTIVATION_TANH";

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        writer_ << "cudnnActivation(" << tensors_name_map_[src] << "[i], "
                << tensors_name_map_[dst] << "[i], " << cudnn_handles_
                << "[i], " << srcDesc_ << ", " << dstDesc_ << ", " << activDesc_
                << ", " << activMode << ", "
                << "0, " << getInitList(idims) << ");\n";

        writer_.indentDec();
        writer_ << "}\n";

        // writer_ << "matrixTanh_" << dtype_flag << "<<<1, " << m
        //         << ", 0, stream[" << oplabel->getDeviceLabel().id << "]>>>("
        //         << tensors_name_map_[A] << ", " << tensors_name_map_[B] << ",
        //         "
        //         << n << ");\n";
    }
    if ((oplabel->getTypeNameLabel()) == "MatrixTanhGrad") {
        auto *src = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *dst = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *dstG = ((TensorNode *)op->getParentNode(2))->getTensor();

        auto *srcG = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto idims = src->getDims();
        if (idims.size() == 2) {
            idims = {idims[0], 1, 1, idims[1]};
        }

        auto activMode = "CUDNN_ACTIVATION_TANH";

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        std::string poolingMode = "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING";

        writer_ << "cudnnActivGrad(" << tensors_name_map_[src] << "[i], "
                << tensors_name_map_[dst] << "[i], " << tensors_name_map_[dstG]
                << "[i], " << tensors_name_map_[srcG] << "[i], "
                << cudnn_handles_ << "[i]," << srcDesc_ << ", " << dstDesc_
                << ", " << dstGradDesc_ << ", " << srcGradDesc_ << ", "
                << activDesc_ << ", " << activMode << ", " << getInitList(idims)
                << ", "
                << "0, " << getInitList(idims) << ");\n";

        writer_.indentDec();
        writer_ << "}\n";
    }
    if ((oplabel->getTypeNameLabel()) == "Relu") {
        // TODO assert
        auto *src = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *dst = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto idims = src->getDims();
        if (idims.size() == 2) {
            idims = {idims[0], 1, 1, idims[1]};
        }

        auto activMode = "CUDNN_ACTIVATION_RELU";

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        writer_ << "cudnnActivation(" << tensors_name_map_[src] << "[i], "
                << tensors_name_map_[dst] << "[i], " << cudnn_handles_
                << "[i], " << srcDesc_ << ", " << dstDesc_ << ", " << activDesc_
                << ", " << activMode << ", "
                << "0, " << getInitList(idims) << ");\n";

        writer_.indentDec();
        writer_ << "}\n";

        // writer_ << "matrixTanh_" << dtype_flag << "<<<1, " << m
        //         << ", 0, stream[" << oplabel->getDeviceLabel().id << "]>>>("
        //         << tensors_name_map_[A] << ", " << tensors_name_map_[B] << ",
        //         "
        //         << n << ");\n";
    }
    if ((oplabel->getTypeNameLabel()) == "ReluGrad") {
        auto *src = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *dst = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *dstG = ((TensorNode *)op->getParentNode(2))->getTensor();

        auto *srcG = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto idims = src->getDims();
        if (idims.size() == 2) {
            idims = {idims[0], 1, 1, idims[1]};
        }

        auto activMode = "CUDNN_ACTIVATION_RELU";

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        writer_ << "cudnnActivGrad(" << tensors_name_map_[src] << "[i], "
                << tensors_name_map_[dst] << "[i], " << tensors_name_map_[dstG]
                << "[i], " << tensors_name_map_[srcG] << "[i], "
                << cudnn_handles_ << "[i]," << srcDesc_ << ", " << dstDesc_
                << ", " << dstGradDesc_ << ", " << srcGradDesc_ << ", "
                << activDesc_ << ", " << activMode << ", "
                << "0, " << getInitList(idims) << ");\n";

        writer_.indentDec();
        writer_ << "}\n";
    }

    if ((oplabel->getTypeNameLabel()) == "LRN") {
        auto *src = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *dst = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto *lrn_op = (LRNOp *)op->getOp();
        float alpha = lrn_op->getAlpha();
        float beta = lrn_op->getBeta();
        float k = lrn_op->getK();
        size_t n = lrn_op->getN();

        auto idims = src->getDims();

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        writer_ << "cudnnLRN(" << tensors_name_map_[src] << "[i], "
                << tensors_name_map_[dst] << "[i], " << cudnn_handles_
                << "[i], " << srcDesc_ << ", " << dstDesc_ << ", " << lrnDesc_
                << ", "
                << "CUDNN_LRN_CROSS_CHANNEL_DIM1"
                << ", " << n << ", " << alpha << ", " << beta << ", " << k
                << ", " << getInitList(idims) << ");\n";

        writer_.indentDec();
        writer_ << "}\n";
    }
    if ((oplabel->getTypeNameLabel()) == "LRNGrad") {
        auto *src = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *dst = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *dstG = ((TensorNode *)op->getParentNode(2))->getTensor();

        auto *srcG = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto *lrn_op = (LRNGradOp *)op->getOp();
        float alpha = lrn_op->getAlpha();
        float beta = lrn_op->getBeta();
        float k = lrn_op->getK();
        size_t n = lrn_op->getN();

        auto idims = src->getDims();

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        writer_ << "cudnnLRNGrad(" << tensors_name_map_[src] << "[i], "
                << tensors_name_map_[dst] << "[i], " << tensors_name_map_[dstG]
                << "[i], " << tensors_name_map_[srcG] << "[i], "
                << cudnn_handles_ << "[i]," << srcDesc_ << ", " << dstDesc_
                << ", " << dstGradDesc_ << ", " << srcGradDesc_ << ", "
                << lrnDesc_ << ", "
                << "CUDNN_LRN_CROSS_CHANNEL_DIM1"
                << ", " << n << ", " << alpha << ", " << beta << ", " << k
                << ", " << getInitList(idims) << ");\n";

        writer_.indentDec();
        writer_ << "}\n";
    }

    if ((oplabel->getTypeNameLabel()) == "MatrixSoftmaxWithLoss") {
        auto *src = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *label = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *prob = ((TensorNode *)op->getChildNode(0))->getTensor();
        auto *loss = ((TensorNode *)op->getChildNode(1))->getTensor();

        auto idims = src->getDims();

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        writer_ << "matrixSoftmaxLoss(" << tensors_name_map_[src] << "[i], "
                << tensors_name_map_[prob] << "[i], "
                << tensors_name_map_[label] << "[i], "
                << tensors_name_map_[loss] << "[i], " << cublas_handles_
                << "[i], " << streams_ << "[i], " << getInitList(idims)
                << ");\n";

        writer_.indentDec();
        writer_ << "}\n";
    }

    if ((oplabel->getTypeNameLabel()) == "MatrixSoftmaxWithLossGrad") {
        auto *label = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *prob = ((TensorNode *)op->getParentNode(1))->getTensor();

        auto *srcG = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto idims = srcG->getDims();

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        writer_ << "matrixSoftmaxLossGrad(" << tensors_name_map_[prob]
                << "[i], " << tensors_name_map_[label] << "[i], "
                << tensors_name_map_[srcG] << "[i], " << streams_ << "[i], "
                << getInitList(idims) << ");\n";

        writer_.indentDec();
        writer_ << "}\n";
    }

    if ((oplabel->getTypeNameLabel()) == "MatrixMatrixFCBias") {
        auto *src = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *weight = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *bias = ((TensorNode *)op->getParentNode(2))->getTensor();
        auto *dst = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto idims = src->getDims(); // n, ci
        auto odims = dst->getDims(); // n, co

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        writer_ << "cublasFCBias(" << tensors_name_map_[src] << "[i], "
                << tensors_name_map_[weight] << "[i], "
                << tensors_name_map_[bias] << "[i], " << tensors_name_map_[dst]
                << "[i], " << cublas_handles_ << "[i], " << getInitList(idims)
                << ", " << getInitList(odims) << ");\n";

        writer_.indentDec();
        writer_ << "}\n";
    }

    if ((oplabel->getTypeNameLabel()) == "MatrixMatrixFCBiasGrad") {
        auto *src = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *w = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *b = ((TensorNode *)op->getParentNode(2))->getTensor();
        auto *dstG = ((TensorNode *)op->getParentNode(3))->getTensor();

        auto *srcG = ((TensorNode *)op->getChildNode(0))->getTensor();
        auto *wG = ((TensorNode *)op->getChildNode(1))->getTensor();
        auto *bG = ((TensorNode *)op->getChildNode(2))->getTensor();

        auto idims = src->getDims();  // n, ci
        auto odims = dstG->getDims(); // n, co

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        writer_ << "cublasFCBiasGrad(" << tensors_name_map_[src] << "[i], "
                << tensors_name_map_[w] << "[i], " << tensors_name_map_[b]
                << "[i], " << tensors_name_map_[dstG] << "[i], "
                << tensors_name_map_[srcG] << "[i], " << tensors_name_map_[wG]
                << "[i], " << tensors_name_map_[bG] << "[i], "
                << cublas_handles_ << "[i], " << getInitList(idims) << ", "
                << getInitList(odims) << ");\n";

        writer_.indentDec();
        writer_ << "}\n";
    }

    if ((oplabel->getTypeNameLabel()).compare("SGD") == 0) {
        auto *w = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *dw = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *momen = ((TensorNode *)op->getParentNode(2))->getTensor();

        auto *sgdOp = (SGDOp *)op->getOp();
        float lr = sgdOp->getLR();
        float decay = sgdOp->getDecay();
        float momentum = sgdOp->getMomentum();
        size_t batch = sgdOp->getBatch();

        // assert(input == input_mirror && "SGD input and output ptr should
        // refer to the same Tensor\n");
        size_t size = w->size();

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        writer_ << "cuSGD(" << tensors_name_map_[w] << "[i], "
                << tensors_name_map_[dw] << "[i], " << tensors_name_map_[momen]
                << "[i], " << tensors_name_map_[w] << "[i], " << size << ", "
                << batch << ", " << lr << ", " << decay << ", " << momentum
                << ", " << streams_ << "[i]);\n";

        writer_.indentDec();
        writer_ << "}\n";
    }

    if ((oplabel->getTypeNameLabel()) == "ElementAdd") {
        auto *lhs = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *rhs = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *dst = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto num = lhs->size();

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        writer_ << "cuElementAdd(" << tensors_name_map_[lhs] << "[i], "
                << tensors_name_map_[rhs] << "[i], " << tensors_name_map_[dst]
                << "[i], " << num << ", " << streams_ << "[i]);\n";

        writer_.indentDec();
        writer_ << "}\n";
    }

    if ((oplabel->getTypeNameLabel()) == "Dropout" ||
        (oplabel->getTypeNameLabel()) == "ElementMul") {
        auto *lhs = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *rhs = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *dst = ((TensorNode *)op->getChildNode(0))->getTensor();

        auto num = lhs->size();

        writer_ << "for(int i=0; i<" << ngpus_ << "; i++) {\n";
        writer_.indentInc();
        writer_ << "checkCUDA(cudaSetDevice(i));\n";

        writer_ << "cuElementMul(" << tensors_name_map_[lhs] << "[i], "
                << tensors_name_map_[rhs] << "[i], " << tensors_name_map_[dst]
                << "[i], " << num << ", " << streams_ << "[i]);\n";

        writer_.indentDec();
        writer_ << "}\n";
    }

    if ((oplabel->getTypeNameLabel()).compare("MatrixMatrixMul") == 0) {

        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *C = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = C->getDim(0);
        int k = A->getDim(1);
        int n = C->getDim(1);
        if (config_.cublas) {
            std::string alpha = UniqueName("alpha");
            std::string beta = UniqueName("beta");
            writer_ << "const float " << alpha << "=1, " << beta << "=0;\n";
            writer_ << "cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,\n";
            writer_.indentInc();
            writer_ << n << ", " << m << ", " << k << ",\n";
            writer_ << "&" << alpha << ",\n";
            writer_ << tensors_name_map_[B] << ", " << n << ",\n";
            writer_ << tensors_name_map_[A] << ", " << k << ",\n";
            writer_ << "&" << beta << ",\n";
            writer_ << tensors_name_map_[C] << ", " << n << ");\n";
            writer_.indentDec();
        }
    }

    if ((oplabel->getTypeNameLabel()) == "BatchedAdd" ||
        (oplabel->getTypeNameLabel()) == "MatrixVectorAdd") {
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getParentNode(1))->getTensor();
        auto *C = ((TensorNode *)op->getChildNode(0))->getTensor();

        size_t sliceNum, sliceSize;
        std::tie(sliceNum, sliceSize) = convertToDim2(A->getDims());
        auto bdim = B->size();
        (void)bdim;
        assert((sliceSize == bdim) &&
               "batch flattened dim.second != bias dim!");

        writer_ << "batchedadd_" << dtype_flag << "<<<1, " << sliceNum
                << ", 0, stream[" << oplabel->getDeviceLabel().id << "]>>>("
                << tensors_name_map_[C] << ", " << tensors_name_map_[A] << ", "
                << tensors_name_map_[B] << ", " << sliceSize << ");\n";
    }

    if ((oplabel->getTypeNameLabel()).compare("MatrixSoftmax") == 0) {
        // TODO assert
        auto *A = ((TensorNode *)op->getParentNode(0))->getTensor();
        auto *B = ((TensorNode *)op->getChildNode(0))->getTensor();
        int m = A->getDim(0);
        int n = A->getDim(1);

        writer_ << "matrixSoftmax_" << dtype_flag << "<<<1, " << m
                << ", 0, stream[" << oplabel->getDeviceLabel().id << "]>>>("
                << tensors_name_map_[A] << ", " << tensors_name_map_[B] << ", "
                << n << ");\n";
    }

    if (config_.compute_op_annotation) {
        writer_ << "*/\n";
    }
}

} // namespace codegen
} // namespace swc