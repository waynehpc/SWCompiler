/*************************************************************************
    > File Name: src/codegen/ParallelCodegen.cpp
    > Author: wayne
    > Mail:
    > Created Time: 二  7/30 15:12:17 2019
 ************************************************************************/

#include "Codegen.h"
#include "SWC.h"
#include <fstream>

namespace swc {
namespace codegen {

static const std::map<std::string, std::string> DTYPE_MPI_DATATYPE_MAP = {
    {"int", "MPI_INT"},
    {"float", "MPI_FLOAT"},
    {"double", "MPI_DOUBLE"}
};

bool isParallel(TensorNode *node) {
    Label *label = node->getLabel();
    Device dev = label->getDeviceLabel();
    return dev.rank == INT_MAX;
}
bool isParallel(OpNode *node) {
    // OpNode must have parent TensorNode
    auto *p0 = (TensorNode *)node->getParentNode(0);
    Device dev_p0 = p0->getLabel()->getDeviceLabel();

    // but may not have child tensor node (debug)
    if (node->childNum() > 0) {
        auto *c0 = (TensorNode *)node->getChildNode(0);
        Device dev_c0 = c0->getLabel()->getDeviceLabel();
        SWLOG_DEBUG(1) << node->name() << " isParallel = "
                       << static_cast<int>((dev_p0.rank == INT_MAX) &&
                                           (dev_c0.rank == INT_MAX))
                       << "\n";
        return (dev_p0.rank == INT_MAX) && (dev_c0.rank == INT_MAX);
    }
    SWLOG_DEBUG(1) << node->name() << " isParallel = "
                   << static_cast<int>((dev_p0.rank == INT_MAX)) << "\n";
    return (dev_p0.rank == INT_MAX);
}

int ParallelCodegen::getMPISendRecvTag(Tensor *tensor) {
    int idx = 0;
    for (auto &t : mpi_sendRecv_tags_) {
        if (t == tensor)
            return idx;
        idx++;
    }
    mpi_sendRecv_tags_.push_back(tensor);
    return idx;
}

bool ParallelCodegen::delMPISendRecvTag(Tensor *tensor) {
    for (auto it = mpi_sendRecv_tags_.begin(); it != mpi_sendRecv_tags_.end();
         it++) {
        if (*it == tensor) {
            mpi_sendRecv_tags_.erase(it);
            return true;
        }
    }
    return false;
}

void ParallelCodegen::initMakefileBuilder() {
    Codegen::initMakefileBuilder();
    makefile_builder_.setCXXCompiler("mpic++");
}

void ParallelCodegen::emitHeader() {
    Codegen::emitHeader();
    headerWriter_ << "#include <mpi.h>\n";
}

void ParallelCodegen::emitEnvInit() { emitMPIInit(); }
void ParallelCodegen::emitEnvFinalize() { emitMPIFinalize(); }

void ParallelCodegen::emitMPIInit() {
    writer_ << "// ========================================================\n";
    writer_ << "// MPI INIT\n";
    writer_ << "int rank, nprocs;\n";
    writer_ << "char proc_name[MPI_MAX_PROCESSOR_NAME];\n";
    writer_ << "int proc_name_len;\n";
    writer_ << "MPI_Status status;\n";

    writer_ << "MPI_Init(&argc, &argv);\n";
    writer_ << "MPI_Comm_size(MPI_COMM_WORLD, &nprocs);\n";
    writer_ << "MPI_Comm_rank(MPI_COMM_WORLD, &rank);\n";
    writer_ << "MPI_Get_processor_name(proc_name,&proc_name_len);\n";
    writer_ << "std::cout << \"process \" << rank << \" of \" << nprocs << \" "
               "run on \" << proc_name << std::endl;\n";
}
void ParallelCodegen::emitMPIFinalize() { writer_ << "MPI_Finalize();\n"; }

void ParallelCodegen::allocateMemAddr() {
    SWLOG_DEBUG(4) << "begin allocateMemAddr...\n";
    for (int i = 0; i < graph_->topologyNum(); i++) {
        for (int j = 0; j < graph_->getNumInTopoLevel(i); j++) {
            auto node = graph_->getNodeInTopo(i, j);
            if (node->nodeType() == TENSOR_NODE) {
                auto *tnode = (TensorNode *)node;
                Tensor *tensor = tnode->getTensor();
                if (tensors_name_map_.count(tensor))
                    continue;

                std::string buf_name = UniqueName(tnode->name());
                size_t size = tensor->getSizeInBytes();
                Device dev = tnode->getLabel()->getDeviceLabel();

                SWLOG_DEBUG(1)
                    << "allocateMemAddr " << tnode->name() << " " << size
                    << " on dev(" << dev.rank << ", "
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

                if (dev.rank == 0) {
                    _master_tensors.push_back(tnode);
                } else {
                    _parallel_tensors.push_back(tnode);
                }
            }
        }
    }

    SWLOG_DEBUG(4) << "end allocateMemAddr...\n";
}

void ParallelCodegen::emitVarDeclarations() {
    SWLOG_DEBUG(4) << "begin emitVarDeclarations...\n";

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

    for (auto it : tensors_name_map_) {
        auto *tensor = it.first;
        std::string dtype = getTypeString(tensor);
        writer_ << dtype << " *" << it.second << ";\n";
    }

    writer_ << "\n";
    SWLOG_DEBUG(4) << "end emitVarDeclarations...\n";
}

void ParallelCodegen::emitMemAllocations() {
    SWLOG_DEBUG(4) << "begin emitMemAllocations...\n";

    writer_ << "if(rank == 0) {\n";
    writer_.indentInc();
    for (auto m : mem_allocators_) {
        MemoryAllocator *allocator = m.get();
        auto dev = allocator->getDevice();
        std::string base = allocator->getBasePtrName();
        uint64_t size = allocator->getMemAllocated();
        if (dev.rank != 0 || size == 0)
            continue;

        emitMemAllocation(base, size, dev);
    }

    writer_.indentDec();
    writer_ << "} // if rank\n";

    /*
        writer_ << "if(rank != 0) {\n";
        writer_.indentInc();
    */

    if (p_mem_alllocator_->getMemAllocated()) {

        auto dev = p_mem_alllocator_->getDevice();
        std::string base = p_mem_alllocator_->getBasePtrName();
        uint64_t size = p_mem_alllocator_->getMemAllocated();

        emitMemAllocation(base, size, dev);
    }
    // for (auto m : mem_allocators_) {
    //     MemoryAllocator *allocator = m.get();
    //     auto dev = allocator->getDevice();
    //     std::string base = allocator->getBasePtrName();
    //     uint64_t size = allocator->getMemAllocated();
    //     if (dev.rank == 0 || size == 0)
    //         continue;
    //
    //     emitMemAllocation(base, size, dev);
    // }
    /*
        writer_.indentDec();
        writer_ << "} // if rank != 0\n";
    */

    writer_ << "\n";
    SWLOG_DEBUG(4) << "end emitMemAllocations...\n";
}

void ParallelCodegen::emitTensorAddresses() {
    SWLOG_DEBUG(4) << "begin emitTensorAddresses...\n";
    writer_ << "if(rank == 0) {\n";
    writer_.indentInc();

    for (auto *tnode : _master_tensors) {
        auto *tensor = tnode->getTensor();
        std::string dtype = getTypeString(tensor);

        std::string name = tensors_name_map_[tensor];
        std::string base;
        uint64_t offset;
        std::tie(base, offset) = tensors_offset_map_[tensor];
        writer_ << name << " = reinterpret_cast<" << dtype << "*>(" << base
                << " + " << offset << ");\n";
    }

    writer_.indentDec();
    writer_ << "} // if rank\n";

    /*
        writer_ << "if(rank != 0) {\n";
        writer_.indentInc();
    */

    for (auto *tnode : _parallel_tensors) {
        auto *tensor = tnode->getTensor();
        std::string dtype = getTypeString(tensor);

        std::string name = tensors_name_map_[tensor];
        std::string base;
        uint64_t offset;
        std::tie(base, offset) = tensors_offset_map_[tensor];
        writer_ << name << " = reinterpret_cast<" << dtype << "*>(" << base
                << " + " << offset << ");\n";
    }

    /*
        writer_.indentDec();
        writer_ << "} // if rank != 0\n";
    */

    writer_ << "\n";
    SWLOG_DEBUG(4) << "end emitTensorAddresses...\n";
}

void ParallelCodegen::emitTensorInitializations() {
    SWLOG_DEBUG(4) << "begin emitTensorInitializations...\n";
    writer_ << "if(rank == 0) {\n";
    writer_.indentInc();

    for (auto *tnode : _master_tensors) {
        emitTensorInitialization(tnode);
    }

    for (auto *tnode : _parallel_tensors) {
        auto *tensor = tnode->getTensor();
        if (tensor->getTensorInitType() == TensorInitType::PARENTOP) {
            auto *parent = (OpNode *)tnode->getParentNode(0);
            if (auto *scatter = dynamic_cast<ScatterOp *>(parent->getOp())) {
                writer_ << "// master to worker send statements\n";
                masterWorkerDispatcher(parent, 0);
            }
        }
    }

    writer_.indentDec();
    writer_ << "} // if rank == 0\n";

    writer_ << "\n";

    /*
        writer_ << "if(rank != 0) {\n";
        writer_.indentInc();
    */

    for (auto *tnode : _parallel_tensors) {
        auto *tensor = tnode->getTensor();
        if (tensor->getTensorInitType() == TensorInitType::PARENTOP) {
            auto *parent = (OpNode *)tnode->getParentNode(0);
            if (auto *scatter = dynamic_cast<ScatterOp *>(parent->getOp())) {
                writer_ << "// worker recv from master statements\n";
                if (parent->runable())
                    masterWorkerDispatcher(parent, 1);
                continue;
            }
        }
        emitTensorInitialization(tnode);
    }

    /*
        writer_.indentDec();
        writer_ << "} // if rank != 0\n";
    */

    SWLOG_DEBUG(4) << "end emitTensorInitializations...\n";
}

void ParallelCodegen::masterWorkerDispatcher(OpNode *op,
                                             int side /*master:0, worker:1*/) {
    if (auto *scatter = dynamic_cast<ScatterOp *>(op->getOp())) {
        auto *in = ((TensorNode *)op->getParentNode(0));
        auto *in_tensor = in->getTensor();
        // Device in_dev = in->getLabel()->getDeviceLabel();
        auto *out = ((TensorNode *)op->getChildNode(0));
        auto *out_tensor = out->getTensor();

        std::string fname = tensors_name_map_[in_tensor];
        std::string tname = tensors_name_map_[out_tensor];

        int axis = scatter->getAxis();
        int degree = scatter->getDegree();

        // size_t offset = (axis == -1) ? 0 : out_tensor->size();

        auto idims = in_tensor->getDims();
        auto odims = out_tensor->getDims();
        int n = in_tensor->getNDim();
        std::string dtype = getTypeString(in_tensor);

        size_t out_count = out_tensor->size();


        // TODO: -1 BroadcastOp
        if(axis == -1) {
            if(side == 0) {
                masterWriter_ << "// rank 0 memcpy from " << fname << " to " << tname
                        << "\n";
                masterWriter_ << tname << " = " << fname << ";\n";
                masterWriter_ << "MPI_Bcast(" << tname << ", " << out_count << ", "
                        << DTYPE_MPI_DATATYPE_MAP.at(dtype) << ", "
                        << "0, " << "MPI_COMM_WORLD);\n";
            }
            else {
                workerWriter_ << "MPI_Bcast(" << tname << ", " << out_count << ", "
                        << DTYPE_MPI_DATATYPE_MAP.at(dtype) << ", "
                        << "0, " << "MPI_COMM_WORLD);\n";
            }

            return;
        }

        int tag = getMPISendRecvTag(out_tensor);

        size_t sendType_count = 1, sendType_num=1, sendType_stride = 1;
        for(int i=0; i<axis; i++)
            sendType_count *= idims[i];
        for(int i=axis; i<n; i++)
            sendType_stride*= idims[i];
        sendType_num = sendType_stride / degree;


        if (side == 0) {
            masterWriter_ << "MPI_Datatype " << fname+"_sendType;\n"
                << "MPI_Type_vector("
                << sendType_count << ", "
                << sendType_num<< ", "
                << sendType_stride<< ", "
                << DTYPE_MPI_DATATYPE_MAP.at(dtype) << ", "
                << "&"<<fname+"_sendType);\n"
                <<"MPI_Type_commit(&"<<fname+"_sendType);\n";

            masterWriter_ << "size_t " << fname << "_sendOffset = 0;\n";
            masterWriter_ << "for(int r=1; r<" << degree << "; r++) {\n";
            masterWriter_.indentInc();
            masterWriter_ << fname << "_sendOffset = " << sendType_num << " * r;\n";
            masterWriter_ << "MPI_Send(" << fname << "+" << fname+"_sendOffset"<< ", "
            << "1, " << fname+"_sendType, "
            << "r, " << tag << ",  MPI_COMM_WORLD);\n";
            masterWriter_.indentDec();
            masterWriter_ << "} //for \n";

            masterWriter_ << "// rank 0 memcpy from " << fname << " to " << tname
                    << "\n";
            if(axis == 0) {
                masterWriter_ << tname << " = " << fname << ";\n";
            }else {
                masterWriter_ << "MPI_Sendrecv(" << fname << ", "
                    << "1, " << fname+"_sendType" << ", " << "0, " << tag << ", "
                    << tname << ", " << out_count << ", "
                    << DTYPE_MPI_DATATYPE_MAP.at(dtype) << ", "
                    << "0, " << tag
                    << ",  MPI_COMM_WORLD, &status);\n";
            }
            /*
            // masterWriter_ << "for(int r=1; r<=" << degree <<"; r++) {\n";
            masterWriter_ << "for(int r=1; r<" << degree << "; r++) {\n";
            masterWriter_.indentInc();
            masterWriter_ << "MPI_Send(" << fname << "+r*" << offset << ", " << size
                    << ", "
                    << "MPI_CHAR, r, " << tag << ",  MPI_COMM_WORLD);\n";
            masterWriter_.indentDec();
            masterWriter_ << "} //for \n";

            masterWriter_ << "// rank 0 memcpy from " << fname << " to " << tname
                    << "\n";
            masterWriter_ << "memcpy(" << tname << ", " << fname << ", " << size
                    << ");\n";
            */
        } else {
            workerWriter_ << "MPI_Recv(" << tname << ", " << out_count << ", "
                    << DTYPE_MPI_DATATYPE_MAP.at(dtype) << ", "
                    << "0, " << tag
                    << ",  MPI_COMM_WORLD, &status);\n";
            /*
            workerWriter_ << "MPI_Recv(" << tname << ", " << size << ", "
                    << "MPI_CHAR, 0, " << tag
                    << ",  MPI_COMM_WORLD, &status);\n";
            */
        }

    } else if (auto gather = dynamic_cast<GatherOp *>(op->getOp())) {
        auto *in = ((TensorNode *)op->getParentNode(0));
        auto *in_tensor = in->getTensor();
        auto *out = ((TensorNode *)op->getChildNode(0));
        auto *out_tensor = out->getTensor();
        // Device to_dev = to->getLabel()->getDeviceLabel();

        std::string fname = tensors_name_map_[in_tensor];
        std::string tname = tensors_name_map_[out_tensor];
        SWLOG_DEBUG(2) << "gather\n";

        int axis = gather->getAxis();
        int degree = gather->getDegree();

        // -1: replicate -2: reduce
        // size_t offset = (axis < 0) ? 0 : in_tensor->size();

        SWLOG_DEBUG(2) << "GatherOp " << op->name() << " axis=" << axis << " degree=" << degree
                       << "\n";

        int tag = getMPISendRecvTag(in_tensor);

        // reduce
        if(axis == -2) {
            size_t count = in_tensor->size();
            std::string dtype = getTypeString(in_tensor);

            writer_ << "MPI_Reduce(" << fname << ", " << tname << ", "
                << count << ", " << DTYPE_MPI_DATATYPE_MAP.at(dtype) << ", "
                << "MPI_SUM, " << 0 << ", MPI_COMM_WORLD);\n";
            return;
        }

        auto idims = in_tensor->getDims();
        auto odims = out_tensor->getDims();
        int n = in_tensor->getNDim();
        std::string dtype = getTypeString(in_tensor);
        size_t in_count = in_tensor->size();

        size_t recvType_count = 1, recvType_num=1, recvType_stride = 1;
        for(int i=0; i<axis; i++)
            recvType_count *= odims[i];
        for(int i=axis; i<n; i++)
            recvType_stride*= odims[i];
        recvType_num = recvType_stride / degree;

        if (side == 0) {
            masterWriter_ << "MPI_Datatype " << tname+"_recvType;\n"
            << "MPI_Type_vector("
            << recvType_count << ", "
            << recvType_num<< ", "
            << recvType_stride<< ", "
            << DTYPE_MPI_DATATYPE_MAP.at(dtype) << ", "
            << "&"<<tname+"_recvType);\n"
            <<"MPI_Type_commit(&"<<tname+"_recvType);\n";

            masterWriter_ << "size_t " << tname << "_recvOffset = 0;\n";
            masterWriter_ << "for(int r=1; r<" << degree << "; r++) {\n";
            masterWriter_.indentInc();
            masterWriter_ << tname << "_recvOffset = " << recvType_num << " * r;\n";
            masterWriter_ << "MPI_Recv(" << tname << "+" << tname+"_recvOffset"<< ", "
            << "1, " << tname+"_recvType, "
            << "r, " << tag << ",  MPI_COMM_WORLD, &status);\n";
            masterWriter_.indentDec();
            masterWriter_ << "} //for \n";

            masterWriter_ << "// rank 0 memcpy in " << fname << " to " << tname
                    << "\n";
            masterWriter_ << "MPI_Sendrecv(" << fname << ", "
                    << in_count << ", "
                    << DTYPE_MPI_DATATYPE_MAP.at(dtype) << ", "
                    << "0, " << tag << ", "
                    << tname << ", "
                    << "1, " << tname+"_recvType" << ", " << "0, " << tag << ", "
                    << "MPI_COMM_WORLD, &status);\n";
            /*
            masterWriter_ << "for(int r=1; r<" << degree << "; r++) {\n";
            masterWriter_.indentInc();
            masterWriter_ << "MPI_Recv(" << tname << "+r*" << offset << ", " << size
                    << ", "
                    << "MPI_CHAR, r, " << tag
                    << ",  MPI_COMM_WORLD, &status);\n";
            masterWriter_.indentDec();
            masterWriter_ << "} //for \n";

            masterWriter_ << "// rank 0 memcpy in " << fname << " to " << tname
                    << "\n";
            masterWriter_ << "memcpy(" << tname << ", " << fname << ", " << size
                    << ");\n";
            */
        } else {
            workerWriter_ << "MPI_Send(" << fname << ", " << in_count << ", "
                    << DTYPE_MPI_DATATYPE_MAP.at(dtype) << ", "
                    << "0, " << tag
                    << ",  MPI_COMM_WORLD);\n";
            /*
            workerWriter_ << "MPI_Send(" << fname << ", " << size << ", "
                    << "MPI_CHAR, 0, " << tag << ",  MPI_COMM_WORLD);\n";
            */
        }
    }
}

void ParallelCodegen::transformOpDispatcher(OpNode *node) {

    auto *transform = dynamic_cast<TransformOp*>(node->getOp());
    int ki = transform->getPreAxis();
    int ko = transform->getPostAxis();
    int p = transform->getDegree();

    writer_ << "// " << node->name() << "transform parallel strategy "
            << ki << "->" << ko << "\n";

    auto *in = ((TensorNode *)node->getParentNode(0));
    auto *in_tensor = in->getTensor();
    auto *out = ((TensorNode *)node->getChildNode(0));
    auto *out_tensor = out->getTensor();

    std::string fname = tensors_name_map_[in_tensor];
    std::string tname = tensors_name_map_[out_tensor];

    std::string dtype = getTypeString(in_tensor);
    size_t in_count = in_tensor->size();
    size_t out_count = in_tensor->size();

    // std::vector<size_t>
    auto idims = in_tensor->getDims();
    auto odims = out_tensor->getDims();
    int n = in_tensor->getNDim();


    if(ki>=0 && ko>=0) {
        assert(in_count == out_count && "in_tensor size() inequal to out_tensor");

        assert((ki>=0 && ki<n && ko>=0 && ko<n) && "illegal strategy");
        assert((ki != ko) && "same in/out strategy");

        size_t sendType_count = 1, sendType_num=1, sendType_stride = 1;
        for(int i=0; i<ko; i++)
            sendType_count *= idims[i];
        for(int i=ko; i<n; i++)
            sendType_stride*= idims[i];
        sendType_num = sendType_stride / p;

        writer_ << "MPI_Datatype " << fname+"_sendType;\n"
            << "MPI_Type_vector("
            << sendType_count << ", "
            << sendType_num<< ", "
            << sendType_stride<< ", "
            << DTYPE_MPI_DATATYPE_MAP.at(dtype) << ", "
            << "&"<<fname+"_sendType);\n"
            <<"MPI_Type_commit(&"<<fname+"_sendType);\n";

        size_t recvType_count = 1, recvType_num=1, recvType_stride = 1;
        for(int i=0; i<ki; i++)
            recvType_count *= odims[i];
        for(int i=ki; i<n; i++)
            recvType_stride*= odims[i];
        recvType_num = recvType_stride / p;

        writer_<< "MPI_Datatype " << tname+"_recvType;\n"
            << "MPI_Type_vector("
            << recvType_count << ", "
            << recvType_num<< ", "
            << recvType_stride<< ", "
            << DTYPE_MPI_DATATYPE_MAP.at(dtype) << ", "
            << "&"<<tname+"_recvType);\n"
            <<"MPI_Type_commit(&"<<tname+"_recvType);\n";

        writer_ << "size_t " << fname << "_sendOffset = 0;\n"
                << "size_t " << tname << "_recvOffset = 0;\n"
                << "int dest, src;\n";

        writer_ << "for(int i=0; i<nprocs; i++) {\n";
        writer_.indentInc();

        writer_ << "src = " << "(rank + nprocs - i) % nprocs;\n"
                << "dest = " << "(rank + nprocs + i) % nprocs;\n"
                << "\n"
                << fname << "_sendOffset = " << sendType_num << " * dest;\n"
                << tname << "_recvOffset = " << recvType_num << " * src;\n";

        int tag = getMPISendRecvTag(out_tensor);

        writer_ << "\n"
                << "MPI_Sendrecv(" << fname << "+" << fname+"_sendOffset" << ", "
                << "1, " << fname+"_sendType" << ", " << "dest, " << tag << ", "
                << tname << "+" << tname+"_recvOffset" << ", "
                << "1, " << tname+"_recvType" << ", " << "src, " << tag << ", "
                << "MPI_COMM_WORLD, &status);\n";

        writer_.indentDec();
        writer_ << "} // for\n";

        return;
    }
    if(ki==-2 && ko>=0) {
        writer_ << "MPI_Reduce(" << fname << ", " << fname << ", "
            << in_count << ", " << DTYPE_MPI_DATATYPE_MAP.at(dtype) << ", "
            << "MPI_SUM, " << 0 << ", MPI_COMM_WORLD);\n";

        size_t sendType_count = 1, sendType_num=1, sendType_stride = 1;
        for(int i=0; i<ko; i++)
            sendType_count *= idims[i];
        for(int i=ko; i<n; i++)
            sendType_stride*= idims[i];
        sendType_num = sendType_stride / p;

        heteroBegin();

        masterWriter_ << "MPI_Datatype " << fname+"_sendType;\n"
            << "MPI_Type_vector("
            << sendType_count << ", "
            << sendType_num<< ", "
            << sendType_stride<< ", "
            << DTYPE_MPI_DATATYPE_MAP.at(dtype) << ", "
            << "&"<<fname+"_sendType);\n"
            <<"MPI_Type_commit(&"<<fname+"_sendType);\n";
        
        masterWriter_ << "size_t " << fname << "_sendOffset = 0;\n";
        masterWriter_ << "for(int r=1; r<" << p << "; r++) {\n";
        masterWriter_.indentInc();
        masterWriter_ << fname << "_sendOffset = " << sendType_num << " * r;\n";

        int tag = getMPISendRecvTag(out_tensor);

        masterWriter_ << "MPI_Send(" << fname << "+" << fname+"_sendOffset"<< ", "
        << "1, " << fname+"_sendType, "
        << "r, " << tag << ",  MPI_COMM_WORLD);\n";
        masterWriter_.indentDec();
        masterWriter_ << "} //for \n";

        masterWriter_ << "// rank 0 memcpy from " << fname << " to " << tname
                << "\n";
        if(ko == 0) {
            masterWriter_ << tname << " = " << fname << ";\n";
        } else {
            int tag_r0 = getMPISendRecvTag(in_tensor);
            masterWriter_ << "MPI_Sendrecv(" << fname << ", "
                << "1, " << fname+"_sendType" << ", " << "0, " << tag_r0 << ", "
                << tname << ", " << out_count << ", "
                << DTYPE_MPI_DATATYPE_MAP.at(dtype) << ", "
                << "0, " << tag_r0
                << ",  MPI_COMM_WORLD, &status);\n";
        }

        workerWriter_ << "MPI_Recv(" << tname << ", " << out_count << ", "
                    << DTYPE_MPI_DATATYPE_MAP.at(dtype) << ", "
                    << "0, " << tag
                    << ",  MPI_COMM_WORLD, &status);\n";
        heteroEnd();

        return;
    }
}

void ParallelCodegen::reduceOpDispatcher(OpNode *op) {
    auto *from = ((TensorNode *)op->getParentNode(0));
    auto *from_tensor = from->getTensor();
    auto *to = ((TensorNode *)op->getChildNode(0));
    auto *to_tensor = to->getTensor();
    // Device to_dev = to->getLabel()->getDeviceLabel();

    std::string fname = tensors_name_map_[from_tensor];
    std::string tname = tensors_name_map_[to_tensor];

    // reduce
    size_t count = from_tensor->size();
    std::string dtype = getTypeString(from_tensor);

    writer_ << "MPI_Reduce(" << fname << ", " << tname << ", "
        << count << ", " << DTYPE_MPI_DATATYPE_MAP.at(dtype) << ", "
        << "MPI_SUM, " << 0 << ", MPI_COMM_WORLD);\n";
}

void ParallelCodegen::emitTensorInitialization(TensorNode *tnode) {
    auto *tensor = tnode->getTensor();

    std::string dtype = getTypeString(tensor);
    std::string name = tensors_name_map_[tensor];
    uint64_t size = tensor->size();
    std::string base;
    uint64_t offset;
    std::tie(base, offset) = tensors_offset_map_[tensor];

    TensorInitInfo info = tensor->getTensorInitInfo();
    switch (tensor->getTensorInitType()) {
    case TensorInitType::NONE:
        break;
    case TensorInitType::XAVIER: {
        // TODO
        writer_ << "initTensorXavier(" << name << ", " << size << ", "
                << info.getFilterSize() << ");\n";
        break;
    }
    case TensorInitType::CONSTANT: {
        writer_ << "initTensorConstant(" << name << ", " << size << ", "
                << info.getConstant() << ");\n";
        break;
    }
    case TensorInitType::ZERO: {
        writer_ << "initTensorZero(" << name << ", " << size << ");\n";
        break;
    }
    case TensorInitType::FILE: {
        writer_ << "load(" << name << ", " << size << ", " << info.getOffset()
                << ", "
                << "\"" << info.getFilePath() << "\");\n";
        break;
    }
    case TensorInitType::PARENTOP: {
        // auto *op = (OpNode *)tnode->getParentNode(0);
        // dispatchOpNode(op);
        break;
    }
    default:
        SWLOG_DEBUG(1) << name << " TensorInitType= NONE\n";
        break;

    } // switch
}

std::vector<OpNode *> ParallelCodegen::schedule() {
    std::vector<OpNode *> scheduled_nodes;
    for (int i = 0; i < graph_->topologyNum(); i++) {
        for (int j = 0; j < graph_->getNumInTopoLevel(i); j++) {
            auto *node = graph_->getNodeInTopo(i, j);
            // if(node->nodeType() == OP_NODE) {
            if (auto *opnode = dynamic_cast<OpNode *>(node)) {
                scheduled_nodes.push_back(opnode);
            }
        }
    }
    return scheduled_nodes;
}

void ParallelCodegen::emitFuncCalls() {
    SWLOG_DEBUG(4) << "begin emitFuncCalls...\n";

    auto scheduled = schedule();

    SWLOG_DEBUG(4) << ">>>> dispatching\n";
    for (auto *node : scheduled) {
        if (dynamic_cast<ScatterOp *>(node->getOp()) || dynamic_cast<GatherOp *>(node->getOp())) {
            if(!hetero_pending_) {
                heteroBegin();
            }
            dispatchOpNode(node, 0);
            dispatchOpNode(node, 1);

            continue;
        }

        if(hetero_pending_)
            heteroEnd();

        /*
        if (dynamic_cast<TransformOp *>(node->getOp())) {
            dispatchOpNode(node);
            continue;
        }
        */
        if (dynamic_cast<ReduceOp *>(node->getOp())) {
            dispatchOpNode(node);
            continue;
        }
        if (isParallel(node)) {
            dispatchOpNode(node);
        }
        else {
            writer_ << "if(rank == 0) {\n";
            writer_.indentInc();

            dispatchOpNode(node, 0);

            writer_.indentDec();
            writer_ << "} // if rank == 0\n";
        }
    }

    writer_ << "\n";

    SWLOG_DEBUG(4) << "end emitFuncCalls...\n";
}

void ParallelCodegen::heteroBegin() {
    masterWriter_ << "if(rank == 0) {\n";
    masterWriter_.indentInc();
    workerWriter_ << "if(rank != 0) {\n";
    workerWriter_.indentInc();

    hetero_pending_ = true;
}
void ParallelCodegen::heteroEnd() {
    masterWriter_.indentDec();
    masterWriter_<< "} // if rank == 0\n";
    workerWriter_.indentDec();
    workerWriter_ << "} // if rank != 0\n";

    writer_ << masterWriter_.get_code()
            << workerWriter_.get_code();

    masterWriter_.clear();
    workerWriter_.clear();

    writer_ << masterWriter_.get_code()
            << workerWriter_.get_code();

    hetero_pending_ = false;
}

void ParallelCodegen::dispatchOpNode(OpNode *op,
                                     int side /*0:master, ~0: worker*/) {
    if (!op->runable())
        return;

    SWLOG_DEBUG(4) << "dispatchOpNode " << op->name() << " for rank " << side
                   << "\n";
    if(side == 0)
        masterWriter_ << "// " << op->name() << " : " << op->getOpName() << "\n";
    else if(side == 1)
        workerWriter_ << "// " << op->name() << " : " << op->getOpName() << "\n";
    else
        writer_ << "// " << op->name() << " : " << op->getOpName() << "\n";


    Label *label = op->getLabel();
    Device dev = label->getDeviceLabel();
    if (dynamic_cast<ScatterOp *>(op->getOp()) ||
        dynamic_cast<GatherOp *>(op->getOp())) {
        masterWorkerDispatcher(op, side);
    }
    else if (dynamic_cast<ReduceOp*>(op->getOp())) {
        reduceOpDispatcher(op);
    }
    else if (dynamic_cast<TransformOp *>(op->getOp())) {
        transformOpDispatcher(op);
    }
    else {
        switch (dev.type) {
        case DeviceType::CPU:
            emitFuncCall(op);
            break;
        case DeviceType::GPU:
            emitFuncCallCUDA(op);
            break;
        default:
            SWLOG_ERROR << "unknown device type in dispatchOpNode\n";
        }
    }
}

void ParallelCodegen::emitMemFree(std::string name, Device dev) {
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

void ParallelCodegen::emitMemFree() {
    SWLOG_DEBUG(4) << "begin emitMemFree...\n";

    writer_ << "if(rank == 0) {\n";
    writer_.indentInc();
    for (auto m : mem_allocators_) {
        MemoryAllocator *allocator = m.get();
        auto dev = allocator->getDevice();
        std::string base = allocator->getBasePtrName();
        uint64_t size = allocator->getMemAllocated();
        if (dev.rank != 0 || size == 0)
            continue;
        emitMemFree(base, dev);
    }

    writer_.indentDec();
    writer_ << "} // if rank\n";

    /*
        writer_ << "if(rank != 0) {\n";
        writer_.indentInc();
    */

    if (p_mem_alllocator_->getMemAllocated()) {

        auto dev = p_mem_alllocator_->getDevice();
        std::string base = p_mem_alllocator_->getBasePtrName();
        // uint64_t size = p_mem_alllocator_->getMemAllocated();

        emitMemFree(base, dev);
    }
    // for (auto m : mem_allocators_) {
    //     MemoryAllocator *allocator = m.get();
    //     auto dev = allocator->getDevice();
    //     std::string base = allocator->getBasePtrName();
    //     uint64_t size = allocator->getMemAllocated();
    //     if (dev.rank == 0 || size == 0)
    //         continue;
    //
    //     emitMemAllocation(base, size, dev);
    // }

    /*
        writer_.indentDec();
        writer_ << "} // if rank!= 0\n";
    */

    writer_ << "\n";
    SWLOG_DEBUG(4) << "end emitMemFree...\n";
}

} // namespace codegen
} // namespace swc
