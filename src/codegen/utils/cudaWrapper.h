#ifndef _CUDAWRAPPER_H_
#define _CUDAWRAPPER_H_

#include "cuda_kernels.h"
#include <cublas_v2.h>
#include <cuda.h>
#include <cudnn.h>
#include <iostream>
#include <nccl.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define FatalError(s)                                                          \
    {                                                                          \
        std::stringstream _where, _message;                                    \
        _where << __FILE__ << ':' << __LINE__;                                 \
        _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;      \
        std::cerr << _message.str() << "\nAborting...\n";                      \
        cudaDeviceReset();                                                     \
        exit(EXIT_FAILURE);                                                    \
    }

#define checkCUDNN(status)                                                     \
    {                                                                          \
        std::stringstream _error;                                              \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            _error << "CUDNN failure\nError: " << cudnnGetErrorString(status); \
            FatalError(_error.str());                                          \
        }                                                                      \
    }

#define checkCUDA(status)                                                      \
    {                                                                          \
        std::stringstream _error;                                              \
        if (status != 0) {                                                     \
            _error << "Cuda failure\nError: " << cudaGetErrorString(status);   \
            FatalError(_error.str());                                          \
        }                                                                      \
    }

#define checkCUBLAS(status)                                                    \
    {                                                                          \
        std::stringstream _error;                                              \
        if (status != 0) {                                                     \
            _error << "Cublas failure\nError code " << status;                 \
            FatalError(_error.str());                                          \
        }                                                                      \
    }

#define checkNCCL(cmd)                                                         \
    do {                                                                       \
        ncclResult_t r = cmd;                                                  \
        if (r != ncclSuccess) {                                                \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,      \
                   ncclGetErrorString(r));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

template <typename Dtype> ncclDataType_t ncclType();
template <> ncclDataType_t ncclType<float>() { return ncclFloat; }
template <> ncclDataType_t ncclType<int>() { return ncclInt; }
template <> ncclDataType_t ncclType<double>() { return ncclDouble; }

#define VEC_INT(v) std::vector<int> v

#define CUDA_NUM_THREADS 512

void setTensor4dDesc(cudnnTensorDescriptor_t &tensorDesc,
                     cudnnTensorFormat_t &tensorFormat,
                     cudnnDataType_t &dataType, int n, int c, int h, int w) {
    checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc, tensorFormat, dataType, n,
                                          c, h, w));
}

void cudnnConv2d(float *src, float *dst, float *filter, float *bias,
                 cudnnHandle_t &handle, cudnnTensorDescriptor_t &srcTensorDesc,
                 cudnnTensorDescriptor_t &dstTensorDesc,
                 cudnnFilterDescriptor_t &filterDesc,
                 cudnnTensorDescriptor_t &biasTensorDesc,
                 cudnnConvolutionDescriptor_t &convDesc,
                 /*cudnnConvolutionFwdAlgo_t*/ int &algo, VEC_INT(idims),
                 VEC_INT(odims), VEC_INT(kernels), VEC_INT(strides),
                 VEC_INT(pads)) {

    cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

    setTensor4dDesc(srcTensorDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorFormat,
                                          odims[3], idims[3], kernels[0],
                                          kernels[1]));
    setTensor4dDesc(dstTensorDesc, tensorFormat, dataType, odims[0], odims[3],
                    odims[1], odims[2]);

    checkCUDNN(cudnnSetConvolution2dDescriptor(
        convDesc, pads[0], pads[1], // pads top and left
        strides[0], strides[1],     // strides height and width
        1, 1,                       // dilation h and w
        CUDNN_CROSS_CORRELATION,    // mode
        dataType));

    cudnnConvolutionFwdAlgo_t fwdAlgo;
    if (algo < 0) {
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
            handle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwdAlgo));
        algo = fwdAlgo;
    } else {
        fwdAlgo = (cudnnConvolutionFwdAlgo_t)algo;
    }

    size_t workspace_bytes = 0;
    void *workspace = NULL;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle, srcTensorDesc, filterDesc, convDesc, dstTensorDesc, fwdAlgo,
        &workspace_bytes));

    checkCUDA(cudaMalloc(&workspace, workspace_bytes));

    const float alpha = 1.0, beta = 0;
    checkCUDNN(cudnnConvolutionForward(
        handle, &alpha, srcTensorDesc, src, filterDesc, filter, convDesc,
        fwdAlgo, workspace, workspace_bytes, &beta, dstTensorDesc, dst));

    setTensor4dDesc(biasTensorDesc, tensorFormat, dataType, 1, odims[3], 1, 1);

    const float beta_add_bias = 1.0;
    checkCUDNN(cudnnAddTensor(handle, &alpha, biasTensorDesc, bias,
                              &beta_add_bias, dstTensorDesc, dst));

    checkCUDA(cudaFree(workspace));
}

void cudnnConv2dGrad(float *src, float *dstGrad, float *filter, float *srcGrad,
                     float *filterGrad, float *biasGrad, cudnnHandle_t &handle,
                     cudnnTensorDescriptor_t &srcDesc,
                     cudnnTensorDescriptor_t &dstGradDesc,
                     cudnnFilterDescriptor_t &filterDesc,
                     cudnnTensorDescriptor_t &srcGradDesc,
                     cudnnFilterDescriptor_t &filterGradDesc,
                     cudnnTensorDescriptor_t &biasGradDesc,
                     cudnnConvolutionDescriptor_t &convDesc,
                     // int  &algo, // disable algo selection
                     VEC_INT(idims), VEC_INT(odims), VEC_INT(kernels),
                     VEC_INT(strides), VEC_INT(pads)) {

    cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

    setTensor4dDesc(srcDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);
    setTensor4dDesc(dstGradDesc, tensorFormat, dataType, odims[0], odims[3],
                    odims[1], odims[2]);
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, dataType, tensorFormat,
                                          odims[3], idims[3], kernels[0],
                                          kernels[1]));

    setTensor4dDesc(srcGradDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);
    checkCUDNN(cudnnSetFilter4dDescriptor(filterGradDesc, dataType,
                                          tensorFormat, odims[3], idims[3],
                                          kernels[0], kernels[1]));
    setTensor4dDesc(biasGradDesc, tensorFormat, dataType, 1, odims[3], 1, 1);

    checkCUDNN(cudnnSetConvolution2dDescriptor(
        convDesc, pads[0], pads[1], // pads top and left
        strides[0], strides[1],     // strides height and width
        1, 1,                       // dilation h and w
        CUDNN_CROSS_CORRELATION,    // mode
        dataType));

    size_t bwdFilterWorkspaceBytes = 0;
    size_t bwdDataWorkspaceBytes = 0;
    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle, filterDesc, dstGradDesc, convDesc, srcGradDesc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, &bwdDataWorkspaceBytes));
    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        handle, srcDesc, dstGradDesc, convDesc, filterGradDesc,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, &bwdFilterWorkspaceBytes));

    void *bwdDataWorkspace = NULL;
    void *bwdFilterWorkSpace = NULL;
    checkCUDA(cudaMalloc(&bwdDataWorkspace, bwdDataWorkspaceBytes));
    checkCUDA(cudaMalloc(&bwdFilterWorkSpace, bwdFilterWorkspaceBytes));

    const float alpha = 1.0, beta = 0;

    checkCUDNN(cudnnConvolutionBackwardBias(
        handle, &alpha, dstGradDesc, dstGrad, &beta, biasGradDesc, biasGrad));

    // dwDesc may only have format CUDNN_TENSOR_NHWC when all of the following
    // are true:
    //   algo is CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 or
    //   CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 xDesc and dyDesc is NHWC HWC-packed
    //   Data type configuration is PSEUDO_HALF_CONFIG or FLOAT_CONFIG
    //   The convolution is 2-dimensional
    checkCUDNN(cudnnConvolutionBackwardFilter(
        handle, &alpha, srcDesc, src, dstGradDesc, dstGrad, convDesc,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, bwdFilterWorkSpace,
        bwdFilterWorkspaceBytes, &beta, filterGradDesc, filterGrad));
    checkCUDA(cudaFree(bwdFilterWorkSpace));

    checkCUDNN(cudnnConvolutionBackwardData(
        handle, &alpha, filterDesc, filter, dstGradDesc, dstGrad, convDesc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, bwdDataWorkspace,
        bwdDataWorkspaceBytes, &beta, srcGradDesc, srcGrad));
    checkCUDA(cudaFree(bwdDataWorkspace));
}

void cudnnPooling(float *src, float *dst, cudnnHandle_t &handle,
                  cudnnTensorDescriptor_t &srcDesc,
                  cudnnTensorDescriptor_t &dstDesc,
                  cudnnPoolingDescriptor_t &poolingDesc,
                  const cudnnPoolingMode_t mode, VEC_INT(idims), VEC_INT(odims),
                  VEC_INT(kernels), VEC_INT(strides), VEC_INT(pads)) {

    cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

    setTensor4dDesc(srcDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);
    setTensor4dDesc(dstDesc, tensorFormat, dataType, odims[0], odims[3],
                    odims[1], odims[2]);

    const int nbDims = 2; // 2dPooling
    int windowDimA[nbDims] = {kernels[0], kernels[1]};
    int strideA[nbDims] = {strides[0], strides[1]};
    int paddingA[nbDims] = {pads[0], pads[1]};
    checkCUDNN(cudnnSetPoolingNdDescriptor(
        poolingDesc,
        /*const cudnnPoolingMode_t*/ mode,
        /*const cudnnNanPropagation_t*/ CUDNN_PROPAGATE_NAN, nbDims, windowDimA,
        paddingA, strideA));

    const float alpha = 1.0, beta = 0;
    checkCUDNN(cudnnPoolingForward(handle, poolingDesc, &alpha, srcDesc, src,
                                   &beta, dstDesc, dst));
}

void cudnnPoolingGrad(float *src, float *dst, float *dstGrad, float *srcGrad,
                      cudnnHandle_t &handle, cudnnTensorDescriptor_t &srcDesc,
                      cudnnTensorDescriptor_t &dstDesc,
                      cudnnTensorDescriptor_t &dstGradDesc,
                      cudnnTensorDescriptor_t &srcGradDesc,
                      cudnnPoolingDescriptor_t &poolingDesc,
                      const cudnnPoolingMode_t mode, VEC_INT(idims),
                      VEC_INT(odims), VEC_INT(kernels), VEC_INT(strides),
                      VEC_INT(pads)) {

    cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

    setTensor4dDesc(srcDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);
    setTensor4dDesc(dstDesc, tensorFormat, dataType, odims[0], odims[3],
                    odims[1], odims[2]);
    setTensor4dDesc(dstGradDesc, tensorFormat, dataType, odims[0], odims[3],
                    odims[1], odims[2]);

    setTensor4dDesc(srcGradDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);

    const int nbDims = 2; // 2dPooling
    int windowDimA[nbDims] = {kernels[0], kernels[1]};
    int strideA[nbDims] = {strides[0], strides[1]};
    int paddingA[nbDims] = {pads[0], pads[1]};
    checkCUDNN(cudnnSetPoolingNdDescriptor(
        poolingDesc,
        /*const cudnnPoolingMode_t*/ mode,
        /*const cudnnNanPropagation_t*/ CUDNN_PROPAGATE_NAN, nbDims, windowDimA,
        paddingA, strideA));

    const float alpha = 1.0, beta = 0;
    checkCUDNN(cudnnPoolingBackward(handle, poolingDesc, &alpha, dstDesc, dst,
                                    dstGradDesc, dstGrad, srcDesc, src, &beta,
                                    srcGradDesc, srcGrad));
}

void cudnnActivation(
    float *src, float *dst, cudnnHandle_t &handle,
    cudnnTensorDescriptor_t &srcDesc, cudnnTensorDescriptor_t &dstDesc,
    cudnnActivationDescriptor_t &activDesc,
    cudnnActivationMode_t mode, /*sigmoid:0 relu:1, tanh:2 ...*/
    double coef, VEC_INT(idims)) {

    cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

    setTensor4dDesc(srcDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);
    setTensor4dDesc(dstDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);

    checkCUDNN(cudnnSetActivationDescriptor(
        activDesc, mode, CUDNN_PROPAGATE_NAN,
        /* used when CUDNN_ACTIVATION_CLIPPED_RELU*/ coef));

    const float alpha = 1.0, beta = 0;
    checkCUDNN(cudnnActivationForward(handle, activDesc, &alpha, srcDesc, src,
                                      &beta, dstDesc, dst));
}

void cudnnActivGrad(float *src, float *dst, float *dstGrad, float *srcGrad,
                    cudnnHandle_t &handle, cudnnTensorDescriptor_t &srcDesc,
                    cudnnTensorDescriptor_t &dstDesc,
                    cudnnTensorDescriptor_t &dstGradDesc,
                    cudnnTensorDescriptor_t &srcGradDesc,
                    cudnnActivationDescriptor_t &activDesc,
                    cudnnActivationMode_t mode, /*sigmoid:0 relu:1, tanh:2 ...*/
                    double coef, VEC_INT(idims)) {

    cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

    setTensor4dDesc(srcDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);
    setTensor4dDesc(dstDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);
    setTensor4dDesc(dstGradDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);
    setTensor4dDesc(srcGradDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);

    checkCUDNN(cudnnSetActivationDescriptor(
        activDesc, mode, CUDNN_PROPAGATE_NAN,
        /* used when CUDNN_ACTIVATION_CLIPPED_RELU*/ coef));

    const float alpha = 1.0, beta = 0;
    checkCUDNN(cudnnActivationBackward(handle, activDesc, &alpha, dstDesc, dst,
                                       dstGradDesc, dstGrad, srcDesc, src,
                                       &beta, srcGradDesc, srcGrad));
}

void cudnnLRN(float *src, float *dst, cudnnHandle_t &handle,
              cudnnTensorDescriptor_t &srcDesc,
              cudnnTensorDescriptor_t &dstDesc, cudnnLRNDescriptor_t &normDesc,
              cudnnLRNMode_t mode, // CUDNN_LRN_CROSS_CHANNEL_DIM1
              unsigned lrnN, double lrnAlpha, double lrnBeta, double lrnK,
              VEC_INT(idims)) {

    cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

    setTensor4dDesc(srcDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);
    setTensor4dDesc(dstDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);

    checkCUDNN(cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK));
    const float alpha = 1.0, beta = 0;
    checkCUDNN(cudnnLRNCrossChannelForward(handle, normDesc, mode, &alpha,
                                           srcDesc, src, &beta, dstDesc, dst));
}

void cudnnLRNGrad(float *src, float *dst, float *dstGrad, float *srcGrad,
                  cudnnHandle_t &handle, cudnnTensorDescriptor_t &srcDesc,
                  cudnnTensorDescriptor_t &dstDesc,
                  cudnnTensorDescriptor_t &dstGradDesc,
                  cudnnTensorDescriptor_t &srcGradDesc,
                  cudnnLRNDescriptor_t &normDesc,
                  cudnnLRNMode_t mode, // CUDNN_LRN_CROSS_CHANNEL_DIM1
                  unsigned lrnN, double lrnAlpha, double lrnBeta, double lrnK,
                  VEC_INT(idims)) {

    cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NHWC;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

    setTensor4dDesc(srcDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);
    setTensor4dDesc(dstDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);
    setTensor4dDesc(dstGradDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);
    setTensor4dDesc(srcGradDesc, tensorFormat, dataType, idims[0], idims[3],
                    idims[1], idims[2]);

    checkCUDNN(cudnnSetLRNDescriptor(normDesc, lrnN, lrnAlpha, lrnBeta, lrnK));

    const float alpha = 1.0, beta = 0;
    checkCUDNN(cudnnLRNCrossChannelBackward(
        handle, normDesc, mode, &alpha, dstDesc, dst, dstGradDesc, dstGrad,
        srcDesc, src, &beta, srcGradDesc, srcGrad));
}

void matrixSoftmaxLoss(float *src, float *prob, int *label, float *loss,
                       cublasHandle_t &handle, cudaStream_t &stream,
                       VEC_INT(idims)) {

    int num = idims[0];
    int channels = idims[1];

    float *loss_data;
    checkCUDA(cudaMalloc(&loss_data, sizeof(float) * num));
    const int nb = (num + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    matrixSoftmaxLossGPU<<<nb, 512, CUDA_NUM_THREADS, stream>>>(
        src, prob, label, loss_data, num, channels);

    const float alpha = 1.0 / num;
    checkCUBLAS(cublasSscal(handle, num, &alpha, loss_data, 1));

    // TODO: let loss be num * 1 rather than scalar
    // checkCUBLAS(cublasSasum(handle, num, loss_data, 1, loss));

    checkCUDA(cudaFree(loss_data));
}

void matrixSoftmaxLossGrad(const float *prob, const int *label, float *srcGrad,
                           cudaStream_t &stream, VEC_INT(idims)) {
    int num = idims[0];
    int channels = idims[1];

    const int nb = (num + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    matrixSoftmaxLossGradGPU<<<nb, CUDA_NUM_THREADS, 0, stream>>>(
        prob, label, srcGrad, num, channels);
}

void cublasFCBias(const float *src, const float *weight, const float *bias,
                  float *dst, cublasHandle_t &handle, VEC_INT(idims),
                  VEC_INT(odims)) {

    int m = idims[0], k = idims[1], n = odims[1];
    const float alpha = 1.0, beta = 0;
    checkCUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
                            weight, n, src, k, &beta, dst, n));

    std::vector<float> h_bias_multipler(m, 1.0); // m 1
    float *bias_multiplier;
    cudaStream_t stream;
    checkCUBLAS(cublasGetStream(handle, &stream));
    checkCUDA(cudaMalloc(&bias_multiplier, sizeof(float) * m));
    checkCUDA(cudaMemcpyAsync(bias_multiplier, &h_bias_multipler[0],
                              sizeof(float) * m, cudaMemcpyHostToDevice,
                              stream));

    // mx1 * 1*n
    // dst = 1.0*dst + 1.0*bias_multiplier*bias
    const float beta_dst = 1.0;
    checkCUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, 1, &alpha,
                            bias, n, bias_multiplier, 1, &beta_dst, dst, n));

    h_bias_multipler.clear();
    checkCUDA(cudaFree(bias_multiplier));
}

void cublasFCBiasGrad(const float *src, const float *weight, const float *bias,
                      const float *dstG, float *srcG, float *weightG,
                      float *biasG, cublasHandle_t &handle, VEC_INT(idims),
                      VEC_INT(odims)) {

    int m = idims[0], k = idims[1], n = odims[1];

    const float alpha = 1.0, beta = 0;
    // dw = xT * dy row major
    // dwT[n,k] = dyT[n,m] * xT[k,m]^T column major
    checkCUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m, &alpha,
                            dstG, n, src, k, &beta, weightG, n));

    // RM dx = dy * wT
    // dxT[k,m] = wT[n,k]^T * dyT[n,m]
    checkCUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n, &alpha,
                            weightG, n, dstG, n, &beta, srcG, k));

    cudaStream_t stream;
    checkCUBLAS(cublasGetStream(handle, &stream));

    std::vector<float> h_db_multipler(m, 1.0); // m 1
    float *db_multipler;
    checkCUDA(cudaMalloc(&db_multipler, sizeof(float) * m));
    checkCUDA(cudaMemcpyAsync(db_multipler, &h_db_multipler[0],
                              sizeof(float) * m, cudaMemcpyHostToDevice,
                              stream));

    // RM db[n,1] = dy[m,n]T * x1[m,1];
    // CM db[n,1] = dyT[n,m] * x1[m,1];

    checkCUBLAS(cublasSgemv(handle, CUBLAS_OP_N, n, m, &alpha, dstG, n,
                            db_multipler, 1, &beta, biasG, 1));

    h_db_multipler.clear();
    checkCUDA(cudaFree(db_multipler));
}

void cuSGD(float *w, float *dw, float *dw_mom, float *out, int num, int batchsz,
           float lr, float decay, float momentum, cudaStream_t &stream) {

    const int nb = (num + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    sgdGPU<<<nb, CUDA_NUM_THREADS, 0, stream>>>(w, dw, dw_mom, out, num,
                                                batchsz, lr, decay, momentum);
    checkCUDA(cudaStreamSynchronize(stream));
}

void cuElementAdd(const float *lhs, const float *rhs, float *dst, int num,
                  cudaStream_t &stream) {
    const int nb = (num + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    elementAddGPU<<<nb, CUDA_NUM_THREADS, 0, stream>>>(num, lhs, rhs, dst);
    checkCUDA(cudaStreamSynchronize(stream));
}
void cuElementSub(const float *lhs, const float *rhs, float *dst, int num,
                  cudaStream_t &stream) {
    const int nb = (num + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    elementSubGPU<<<nb, CUDA_NUM_THREADS, 0, stream>>>(num, lhs, rhs, dst);
    checkCUDA(cudaStreamSynchronize(stream));
}
void cuElementMul(const float *lhs, const float *rhs, float *dst, int num,
                  cudaStream_t &stream) {
    const int nb = (num + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    elementMulGPU<<<nb, CUDA_NUM_THREADS, 0, stream>>>(num, lhs, rhs, dst);
    checkCUDA(cudaStreamSynchronize(stream));
}
void cuElementDiv(const float *lhs, const float *rhs, float *dst, int num,
                  cudaStream_t &stream) {
    const int nb = (num + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    elementDivGPU<<<nb, CUDA_NUM_THREADS, 0, stream>>>(num, lhs, rhs, dst);
    checkCUDA(cudaStreamSynchronize(stream));
}

// src on root device
// dst[d] on dev d for d in 0,1...,nDev-1
// slicesz = count*len sz of dst
template <typename T>
void cuScatter(const T *src, T **dst, int root, int /*sendcount*/ slicesz,
               int count, int len, int stride, int ndev,
               cudaStream_t *streams) {
    T *packbuf;
    int size = slicesz * ndev;
    checkCUDA(cudaSetDevice(root));
    checkCUDA(cudaMalloc(&packbuf, size * sizeof(T)));

    int nblk = (count * len + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

    for (int d = 0; d < ndev; d++) {
        int offset = len * d;
        memPack<<<nblk, CUDA_NUM_THREADS, 0, streams[root]>>>(
            src + offset, packbuf + slicesz * d, count, len, stride);
        checkCUDA(cudaMemcpyAsync(dst[d], packbuf + slicesz * d,
                                  slicesz * sizeof(T), cudaMemcpyDeviceToDevice,
                                  streams[root]));
    }

    for (int d = 0; d < ndev; d++) {
        checkCUDA(cudaStreamSynchronize(streams[d]));
    }
}

// dst on root device
// src[d] on dev d for d in 0,1...,nDev-1
// slicesz = count*len sz of src
template <typename T>
void cuGather(T **src, T *dst, int root, int /*sendcount*/ slicesz, int count,
              int len, int stride, int ndev, cudaStream_t *streams) {
    T *packbuf;
    int size = slicesz * ndev;
    checkCUDA(cudaSetDevice(root));
    checkCUDA(cudaMalloc(&packbuf, size * sizeof(T)));

    int nblk = (count * len + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

    for (int d = 0; d < ndev; d++) {
        int offset = len * d;
        checkCUDA(cudaMemcpyAsync(packbuf + slicesz * d, src[d],
                                  slicesz * sizeof(T), cudaMemcpyDeviceToDevice,
                                  streams[root]));
        memUnPack<<<nblk, CUDA_NUM_THREADS, 0, streams[root]>>>(
            dst + offset, packbuf + slicesz * d, count, len, stride);
    }

    for (int d = 0; d < ndev; d++) {
        checkCUDA(cudaStreamSynchronize(streams[d]));
    }
}

// I->J
template <typename T>
void cuGatherScatter(T **sbuf, T **rbuf, int slicesz, int scount, int slen,
                     int sstride, int rcount, int rlen, int rstride, int ndev,
                     cudaStream_t *streams) {

    T **spackbuf = (T **)malloc(ndev * sizeof(T *));
    T **rpackbuf = (T **)malloc(ndev * sizeof(T *));
    int size = slicesz * ndev;
    for (int i = 0; i < ndev; ++i) {
        checkCUDA(cudaSetDevice(i));
        checkCUDA(cudaMalloc(spackbuf + i, size * sizeof(T)));
        checkCUDA(cudaMalloc(rpackbuf + i, size * sizeof(T)));
    }

    int nblk = (rcount * rlen + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

    int tdev, soffset, spackbuf_off, tdev_rpackbuf_off;
    int fdev, roffset, rpackbuf_off;

    for (int i = 0; i < ndev; i++) {
        // in iter i, device d
        for (int d = 0; d < ndev; d++) {
            tdev = (d + i) % ndev;
            soffset = slen * tdev;
            spackbuf_off = tdev * slicesz;

            tdev_rpackbuf_off = d * slicesz;

            checkCUDA(cudaSetDevice(d));
            // send piece to tdev.rpackbuf
            memPack<<<nblk, CUDA_NUM_THREADS, 0, streams[d]>>>(
                sbuf[d] + soffset, spackbuf[d] + spackbuf_off, scount, slen,
                sstride);
            checkCUDA(cudaMemcpyAsync(
                rpackbuf[tdev] + tdev_rpackbuf_off, spackbuf[d] + spackbuf_off,
                slicesz * sizeof(T), cudaMemcpyDeviceToDevice, streams[d]));
        }
        for (int d = 0; d < ndev; d++) {
            checkCUDA(cudaStreamSynchronize(streams[d]));
        }

        // already recv from fdev
        for (int d = 0; d < ndev; d++) {
            fdev = (d - i + ndev) % ndev;
            roffset = rlen * fdev;
            rpackbuf_off = fdev * slicesz;

            // unpack rpackbuf from fdev
            checkCUDA(cudaSetDevice(d));
            memUnPack<<<nblk, CUDA_NUM_THREADS, 0, streams[d]>>>(
                rbuf[d] + roffset, rpackbuf[d] + rpackbuf_off, rcount, rlen,
                rstride);
        }
    }

    for (int d = 0; d < ndev; d++) {
        checkCUDA(cudaStreamSynchronize(streams[d]));
    }

    for (int i = 0; i < ndev; ++i) {
        checkCUDA(cudaSetDevice(i));
        checkCUDA(cudaFree(spackbuf[i]));
        checkCUDA(cudaFree(rpackbuf[i]));
    }
}

template <typename T>
void cuReduceScatter(T **sbuf, T **rbuf, int slicesz, int count, int len,
                     int stride, int ndev, ncclComm_t *comms,
                     cudaStream_t *streams) {

    T **spackbuf = (T **)malloc(ndev * sizeof(T *));
    int size = slicesz * ndev;
    for (int i = 0; i < ndev; ++i) {
        checkCUDA(cudaSetDevice(i));
        checkCUDA(cudaMalloc(spackbuf + i, size * sizeof(T)));
    }

    int nblk = (slicesz + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

    for (int i = 0; i < ndev; i++) {
        // for iteration i, pack spackbuf[:]+i*slicesz on each dev
        for (int d = 0; d < ndev; d++) {
            // for device d, different streams enable concurrent kernels
            checkCUDA(cudaSetDevice(d));
            int offset = len * i;
            memPack<<<nblk, CUDA_NUM_THREADS, 0, streams[d]>>>(
                sbuf[d] + offset, spackbuf[d] + slicesz * i, count, len,
                stride);
        }
    }

    for (int d = 0; d < ndev; d++) {
        // reduce part of device d
        // TODO: with streams[DEV][STREAMS_PER_DEV] iterate on streams[i][d]
        // to enable concurrent reduce on multiple devs
        checkNCCL(ncclGroupStart());
        for (int i = 0; i < ndev; i++) {
            checkNCCL(ncclReduce(spackbuf[i] + d * slicesz, rbuf[d], slicesz,
                                 ncclType<T>(), ncclSum, d, comms[i],
                                 streams[i]));
        }
        checkNCCL(ncclGroupEnd());
    }

    for (int d = 0; d < ndev; d++) {
        checkCUDA(cudaStreamSynchronize(streams[d]));
    }

    for (int i = 0; i < ndev; ++i) {
        checkCUDA(cudaSetDevice(i));
        checkCUDA(cudaFree(spackbuf[i]));
    }
}

// pack part of sbuf to rbuf for each dev
// sbuf: size = slizesz * ndev
// sbuf[0]=sbuf[1]=....sbuf[ndev-1]
// streams[d] on device d
// -1 -> i
template <typename T>
void cuMempack(T **sbuf, T **rbuf, int slicesz, int count, int len, int stride,
               int ndev, cudaStream_t *streams) {

    // int size = slicesz * ndev;

    int nblk = (slicesz + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

    for (int d = 0; d < ndev; d++) {
        checkCUDA(cudaSetDevice(d));
        int offset = len * d;

        memPack<<<nblk, CUDA_NUM_THREADS, 0, streams[d]>>>(
            sbuf[d] + offset, rbuf[d], count, len, stride);
    }

    for (int d = 0; d < ndev; d++) {
        checkCUDA(cudaStreamSynchronize(streams[d]));
    }
}

// sbuf: slicesz; rbuf: size = slicesz * ndev
// allgather i->-1
template <typename T>
void cuAllGather(T **sbuf, T **rbuf, int slicesz, int count, int len,
                 int stride, int ndev, ncclComm_t *comms,
                 cudaStream_t *streams) {

    T **rpackbuf = (T **)malloc(ndev * sizeof(T *));
    int size = slicesz * ndev;
    for (int i = 0; i < ndev; ++i) {
        checkCUDA(cudaSetDevice(i));
        checkCUDA(cudaMalloc(rpackbuf + i, size * sizeof(T)));
    }

    checkNCCL(ncclGroupStart());
    for (int i = 0; i < ndev; i++) {
        checkNCCL(ncclAllGather(sbuf[i], rpackbuf[i], slicesz, ncclType<T>(),
                                comms[i], streams[i]));
    }
    checkNCCL(ncclGroupEnd());
    for (int d = 0; d < ndev; d++) {
        checkCUDA(cudaStreamSynchronize(streams[d]));
    }

    int nblk = (slicesz + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;

    for (int i = 0; i < ndev; i++) {
        // for iteration i, unpack rpackbuf[:]+i*slicesz on each dev
        for (int d = 0; d < ndev; d++) {
            // for device d, different streams enable concurrent kernels
            checkCUDA(cudaSetDevice(d));
            int offset = len * i;
            memUnPack<<<nblk, CUDA_NUM_THREADS, 0, streams[d]>>>(
                rbuf[d] + offset, rpackbuf[d] + slicesz * i, count, len,
                stride);
        }
    }
    for (int d = 0; d < ndev; d++) {
        checkCUDA(cudaStreamSynchronize(streams[d]));
    }
}

void cuSyncStreams(cudaStream_t *streams, int ndev) {
    for (int d = 0; d < ndev; d++) {
        checkCUDA(cudaStreamSynchronize(streams[d]));
    }
}

#endif
