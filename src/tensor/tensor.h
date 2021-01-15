/*************************************************************************
    > File Name: tensor.h
    > Author: cryinlaugh
    > Mail: cryinlaugh@gmail.com
    > Created Time: Tues. 12/ 4 15:53:19 2018
 ************************************************************************/

#ifndef _TENSOR_H
#define _TENSOR_H

#include "SWLOG.h"
#include "common.h"
#include <cassert>
#include <iostream>
#include <string>

namespace swc {

constexpr size_t max_dimensions = 6;

class TensorType {
  public:
    TensorType() = default;
    TensorType(const std::initializer_list<size_t> &dims);

    TensorType(const std::initializer_list<size_t> &dims, DataType dtype,
               mem_layout_t layout)
        : dtype_(dtype), layout_(layout) {
        initDimsAndLayout(dims);
    }

    // TensorType(std::vector<size_t> *shape) {
    //     numDims_ = shape->size();
    //     for (size_t i = 0; i < shape->size(); i++) {
    //         shape_[i] = shape->at(i);
    //     }
    // }

    TensorType(const std::vector<size_t> &shape,
               DataType dtype = DataType::Float_t,
               mem_layout_t layout = layout_default)
        : dtype_(dtype), layout_(layout) {
        numDims_ = shape.size();
        for (size_t i = 0; i < shape.size(); i++) {
            shape_[i] = shape.at(i);
        }
    }

    ~TensorType() = default;

    int numDims() const { return numDims_; }
    size_t getDim(int idx) const { return shape_[idx]; }
    size_t size() const;

    size_t getSizeInBytes() const;

    DataType getDataType() const { return dtype_; }
    void setMemLayout(mem_layout_t layout) { layout_ = layout; }
    mem_layout_t getMemLayout() const { return layout_; }
    std::string getMemLayoutTag() const;

    TensorType getTiledTensorType(int index, int n);

  private:
    size_t shape_[max_dimensions] = {0};
    int numDims_{0};

    DataType dtype_{DataType::Float_t};
    mem_layout_t layout_{layout_default};

    void initDimsAndLayout(const std::initializer_list<size_t> &dims);
};

class TensorInitInfo {
    std::string file_{nullptr}; // FILE initialization
    float constant_{0};         // constant initialization
    float filterSize_{1};       // xavier initialization
    size_t offset_{0};

  public:
    TensorInitInfo() : file_(""), constant_(0), filterSize_(3) {}

    std::string getFilePath() const { return file_; }
    float getConstant() const { return constant_; }
    float getFilterSize() const { return filterSize_; }
    size_t getOffset() const { return offset_; }

    void setFilePath(std::string f) { file_ = f; }
    void setConstant(float c) { constant_ = c; }
    void setFilterSize(float fsize) { filterSize_ = fsize; }
    void setOffset(size_t offset) { offset_ = offset; }
};

class Tensor {
  private:
    TensorType type_;
    TensorInitType initType_{TensorInitType::NONE};
    TensorInitInfo initInfo_;
    int train_{0};

  public:
    Tensor() = default;

    Tensor(const TensorType &type) : type_(type) {}

    Tensor(const std::initializer_list<size_t> &shape,
           DataType dtype = DataType::Float_t,
           mem_layout_t layout = layout_default)
        : type_(shape, dtype, layout) {}

    Tensor(const std::vector<size_t> &dims, DataType dtype = DataType::Float_t,
           mem_layout_t layout = layout_default)
        : type_(dims, dtype, layout) {}

    Tensor(const Tensor &other) = delete;
    Tensor &operator=(const Tensor &other) = delete;

    ~Tensor() = default;

    void reset(const TensorType &type) { type_ = type; }

    void reset(const std::initializer_list<size_t> &shape,
               DataType dtype = DataType::Float_t,
               mem_layout_t layout = layout_default) {
        type_ = TensorType(shape, dtype, layout);
    }

    Tensor *clone() const;

    std::vector<size_t>
    getShuffledDims(const std::vector<size_t> &shuffle) const;

    TensorType getShuffledTensorType(const std::vector<size_t> &shuffle) const;

    TensorType getTiledTensorType(int idx, int n) {
        return type_.getTiledTensorType(idx, n);
    }

    DataType getDataType() const { return type_.getDataType(); }

    const std::vector<size_t> getDims() const;

    int getNDim() const { return type_.numDims(); }
    size_t getDim(int idx) const { return type_.getDim(idx); }

    std::pair<size_t, size_t> viewAs2D(int n);

    void setTensorInit(TensorInitType type, float value = 0);
    void setTensorInit(TensorInitType type, std::string file,
                       size_t offset = 0);

    void setTensorInit(TensorInitType type, TensorInitInfo info);

    TensorInitType getTensorInitType() { return initType_; }
    TensorInitInfo getTensorInitInfo() const { return initInfo_; }

    void setTraining(int train) { train_ = train; }
    int getTraining() const { return train_; }

    TensorType getType() const { return type_; }
    size_t size() const { return type_.size(); }
    size_t getSizeInBytes() const { return type_.getSizeInBytes(); }

    void setMemLayout(mem_layout_t layout) { type_.setMemLayout(layout); }
    mem_layout_t getMemLayout() const { return type_.getMemLayout(); }
    std::string getMemLayoutTag() const { return type_.getMemLayoutTag(); }
};

} // namespace swc

#endif
