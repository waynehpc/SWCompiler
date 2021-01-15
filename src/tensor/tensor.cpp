/*************************************************************************
        > File Name: tensor.cpp
        > Author: cryinlaugh
        > Mail: cryinlaugh@gmail.com
        > Created Time: 二 12/ 4 15:56:42 2018
 ************************************************************************/
#include "tensor.h"
#include "SWLOG.h"

namespace swc {

void TensorType::initDimsAndLayout(const std::initializer_list<size_t> &dims) {
    assert(dims.size() <= max_dimensions && "illegal dims.");
    numDims_ = dims.size();
    int idx = 0;
    for (auto d : dims) {
        shape_[idx++] = d;
    }

    if (layout_ == layout_default && numDims_ == 4) {
        layout_ = layout_nhwc;
    }
}

TensorType::TensorType(const std::initializer_list<size_t> &dims) {
    initDimsAndLayout(dims);
}

size_t TensorType::size() const {
    size_t s = 1;
    for (int i = 0; i < numDims_; i++) {
        s *= shape_[i];
    }
    return s;
}

TensorType TensorType::getTiledTensorType(int idx, int n) {
    int ndim = numDims_;
    assert((int)idx < ndim && "illegal tiled dim.");

    std::vector<size_t> dims;
    for (int i = 0; i < ndim; i++) {
        if (i == idx) {
            dims.push_back(shape_[i] / n);
        } else {
            dims.push_back(shape_[i]);
        }
    }

    TensorType t(dims, dtype_, layout_);
    return t;
}

size_t TensorType::getSizeInBytes() const {
    switch (dtype_) {
    case DataType::Float_t:
        return size() * sizeof(float);
    case DataType::Double_t:
        return size() * sizeof(double);
    case DataType::Int8_t:
        return size() * sizeof(int8_t);
    case DataType::Int32_t:
        return size() * sizeof(int32_t);
    default:
        SWLOG_ERROR << "UNKNOWN DataType\n";
        return size() * sizeof(float);
    }
}

Tensor *Tensor::clone() const {
    Tensor *t = new Tensor(type_);
    t->setTraining(train_);
    t->setTensorInit(initType_, initInfo_);
    return t;
}

const std::vector<size_t> Tensor::getDims() const {
    std::vector<size_t> dims(getNDim(), 0);
    for (int i = 0; i < getNDim(); i++)
        dims[i] = getDim(i);
    return dims;
}

void Tensor::setTensorInit(TensorInitType type, float value) {
    initType_ = type;
    switch (type) {
    case TensorInitType::CONSTANT: {
        initInfo_.setConstant(value);
        break;
    }
    case TensorInitType::ZERO: {
        initInfo_.setConstant(0);
        break;
    }
    case TensorInitType::XAVIER: {
        initInfo_.setFilterSize(value);
        break;
    }
    default:
        break;
    }
}

void Tensor::setTensorInit(TensorInitType type, std::string file,
                           size_t offset) {
    assert((type == TensorInitType::FILE) && "init type does not match value");
    initType_ = type;
    initInfo_.setFilePath(file);
    initInfo_.setOffset(offset);
}

void Tensor::setTensorInit(TensorInitType type, TensorInitInfo info) {
    initType_ = type;
    initInfo_ = info;
}

std::vector<size_t>
Tensor::getShuffledDims(const std::vector<size_t> &shuffle) const {
    std::vector<size_t> dims;
    for (auto idx : shuffle) {
        if ((int)idx < getNDim())
            dims.push_back(getDim(idx));
    }
    return dims;
}

TensorType
Tensor::getShuffledTensorType(const std::vector<size_t> &shuffle) const {
    assert((int)shuffle.size() == getNDim() && "illegal shuffle.");

    std::vector<size_t> dims;
    for (auto idx : shuffle) {
        if ((int)idx < getNDim())
            dims.push_back(getDim(idx));
    }

    TensorType t(dims, getDataType(), getMemLayout());
    return t;
}

std::pair<size_t, size_t> Tensor::viewAs2D(int n) {
    assert(n >= 1 && n <= getNDim() && "illegal n");
    size_t dim0 = getDim(0);
    int i;
    for (i = 1; i < n; i++)
        dim0 *= getDim(i);
    size_t dim1 = 1;
    for (; i < getNDim(); i++)
        dim1 *= getDim(i);
    return {dim0, dim1};
}

std::string TensorType::getMemLayoutTag() const {

    int ndim = numDims_;

    switch (layout_) {
    case layout_default:
        if (ndim == 1)
            return "x";
        if (ndim == 2)
            return "nc";
        break;
    case layout_nhwc:
        return "nhwc";
        break;
    case layout_nchw:
        return "nchw";
        break;
    case layout_nc:
        return "nc";
        break;
    case layout_cn:
        return "cn";
        break;
    default:
        return "any";
    }
    return "any";
}

/*

template<>
size_t Tensor<double>::getSizeInBytes() const {
    return shape_->size() * sizeof(double);
}

template<>
size_t Tensor<int>::getSizeInBytes() const {
    return shape_->size() * sizeof(int);
}
*/
} // namespace swc
