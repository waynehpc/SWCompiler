/*************************************************************************
        > File Name: CUDACodegen.h
        > Author: wayne
        > Mail:
        > Created Time: Fri 14 Feb 2020 08:46:48 AM UTC
 ************************************************************************/
#ifndef _CUDACODEGEN_H_
#define _CUDACODEGEN_H_

#include "Codegen.h"

namespace swc {
namespace codegen {

class CUDACodegen : public Codegen {
  public:
    CUDACodegen(IRGraph *graph, Config &config) : Codegen(graph, config) {
        ngpus_ = config.ngpus_per_rank;
    }

    /// emit CUDA related code. e.g. cuBlas handle, cudaStream creating
    /// NCCL init
    /// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-1-single-process-single-thread-multiple-devices
    /// cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
    /// NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
    void emitCUDAInit() override;

    void initMakefileBuilder() override;

    // ================================================================
    // allocate  mem in parallel gpu
    void allocateMemAddr() override;

    // emitVarDeclarations implemented in Codegen::emitVarDeclarations
    void emitVarDeclarations() override;

    /**
     * cudaMalloc through a for loop
     * e.g.
     * for(int i=0; i<nDev; i++) {
     *   CUDACHECK(cudaSetDevice(i));
     *   CUDACHECK(cudaMalloc(buf+i, size * sizeof(char)));
     * }
     */
    void emitMemAllocations() override;
    void emitMultiGPUMemAlloc(std::string base, uint64_t size, Device &dev);

    // host tensors and device tensors
    void emitTensorAddresses() override;

    void emitDataLoaderInit() override {}
    void emitInferDataLoaderInit() override {}

    // ================================================================
    void emitExecute() override;

    void emitcudnnDescs();

    void emitFuncCalls() override;
    void emitFuncCallCUDA(OpNode *op) override;
    void dispatchOpNode(OpNode *op) override;

    // cudaFree in a for loop
    void emitMemFree() override;
    void emitMemFree(std::string name, Device dev) override;

    void emitEnvFinalize() override {}

  private:
    const std::string nccl_comms = "nccl_comms";
    const std::string streams_ = "cuda_streams";
    const std::string cublas_handles_ = "cublas_handles";
    const std::string cudnn_handles_ = "cudnn_handles";
    const std::string srcDesc_ = "srcDesc";
    const std::string srcGradDesc_ = "srcGradDesc";
    const std::string dstDesc_ = "dstDesc";
    const std::string dstGradDesc_ = "dstGradDesc";
    const std::string filterDesc_ = "filterDesc";
    const std::string filterGradDesc_ = "filterGradDesc";
    const std::string biasDesc_ = "biasDesc";
    const std::string biasGradDesc_ = "biasGradDesc";
    const std::string convDesc_ = "convDesc";
    const std::string poolingDesc_ = "poolingDesc";
    const std::string activDesc_ = "activDesc";
    const std::string normDesc = "normDesc";
    const std::string lrnDesc_ = "normDesc";
    const std::string convAlgo_ = "convAlgo";

    int ngpus_;
    std::vector<Tensor *> host_tensors_;
    std::vector<Tensor *> host_parallel_tensors_;
    std::vector<Tensor *> master_gpu_tensors_;
    std::vector<Tensor *> parallel_gpu_tensors_;

    void dispatchScatterOp(OpNode *op);
    void dispatchGatherOp(OpNode *op);
    void dispatchTransformOp(OpNode *op);
    void dispatchHostOpNode(OpNode *op);
    void dispatchDevOpNode(OpNode *op);
};
} // namespace codegen
} // namespace swc

#endif
