#ifndef _CUDAKERNELS_H_
#define _CUDKERNELS_H_

template <typename T>
__global__ void matrixSoftmaxLossGPU(T *src, T *prob, int *label, T *loss,
                                     int num, int channels) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
         i += gridDim.x * blockDim.x) {
        T max_ = src[i * channels];
        for (int j = 0; j < channels; j++) {
            max_ = max(max_, src[i * channels + j]);
        }

        T sum = 0;
        for (int j = 0; j < channels; j++) {
            T e = exp(src[i * channels + j] - max_);
            sum += e;
            prob[i * channels + j] = e;
        }
        for (int j = 0; j < channels; j++) {
            prob[i * channels + j] /= sum;
        }

        const int k = label[i];
        loss[i] = log(sum * exp(max_)) - src[i * channels + k];
    } // loop over i
}

template <typename T>
__global__ void matrixSoftmaxLossGradGPU(const T *prob, const int *label,
                                         T *srcGrad, int num, int channels) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
         i += gridDim.x * blockDim.x) {
        for (int j = 0; j < channels; j++) {
            T delta = (label[i] == j);
            srcGrad[i * channels + j] = prob[i * channels + j] - delta;
        }
    } // loop over i
}

template <typename T>
__global__ void sgdGPU(T *w, T *dw, T *dw_mom, T *out, size_t num, size_t batch,
                       T lr, T decay, T momentum) {

    float neglr = -lr / batch;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
         i += gridDim.x * blockDim.x) {
        float negDelta = (w[i] * decay + dw[i]) * neglr + dw_mom[i] * momentum;
        dw_mom[i] = negDelta;
        out[i] += negDelta;
    } // loop over i
}

template <typename T>
__global__ void matrixTanh(T *src, T *dest, int num, int channels) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
         i += gridDim.x * blockDim.x) {
        for (int j = 0; j < channels; j++) {
            dest[i * channels + j] =
                1 - 2 / (exp(src[i * channels + j] * 2) + 1);
        }
    } // loop over i
}

template <typename T>
__global__ void batchedAddGPU(T *dest, const T *batch, const T *slice, int num,
                              int channels) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
         i += gridDim.x * blockDim.x) {
        for (int j = 0; j < channels; j++) {
            dest[i * channels + j] = batch[i * channels + j] + slice[j];
        }
    } // loop over i

    // another way: map channels to tid
}

template <typename T>
__global__ void elementMulGPU(const int num, const T *lhs, const T *rhs,
                              T *dst) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
         i += gridDim.x * blockDim.x) {
        dst[i] = lhs[i] * rhs[i];
    }
}
template <typename T>
__global__ void elementAddGPU(const int num, const T *lhs, const T *rhs,
                              T *dst) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
         i += gridDim.x * blockDim.x) {
        dst[i] = lhs[i] + rhs[i];
    }
}
template <typename T>
__global__ void elementSubGPU(const int num, const T *lhs, const T *rhs,
                              T *dst) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
         i += gridDim.x * blockDim.x) {
        dst[i] = lhs[i] - rhs[i];
    }
}
template <typename T>
__global__ void elementDivGPU(const int num, const T *lhs, const T *rhs,
                              T *dst) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
         i += gridDim.x * blockDim.x) {
        dst[i] = lhs[i] / rhs[i];
    }
}

#ifndef CUDA_NUM_THREADS
#define CUDA_NUM_THREADS 512
#endif

template <typename T>
__global__ void memPack(const T *src, T *buf, int count, int len, int stride) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count * len;
         idx += gridDim.x * blockDim.x) {
        // idx mapped to 1d index in dest buf
        int i = idx / len;
        int j = idx % len;
        buf[idx] = src[i * stride + j];
    }
}

template <typename T>
__global__ void memUnPack(T *src, const T *buf, int count, int len,
                          int stride) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count * len;
         idx += gridDim.x * blockDim.x) {
        // idx mapped to 1d index in dest buf
        int i = idx / len;
        int j = idx % len;
        src[i * stride + j] = buf[idx];
    }
}

#endif
