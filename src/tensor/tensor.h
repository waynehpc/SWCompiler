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

class TensorXXShape {
  private:
    int _ndim;
    std::vector<size_t> *shape_;

  public:
    TensorXXShape(std::vector<size_t> *shape);
    TensorXXShape(const std::initializer_list<size_t> &shape) {
        shape_ = new std::vector<size_t>();
        for (auto i : shape) {
            shape_->push_back(i);
        }
        _ndim = shape_->size();
    }

    ~TensorXXShape() {}
    void destroy() {}

    int getNDim() const;
    size_t getDim(int idx) const;
    size_t size() const;

    void setShape(const std::vector<size_t> &shape) {
        _ndim = shape.size();
        shape_->resize(_ndim);
        for (int i = 0; i < _ndim; i++)
            (*shape_)[i] = shape[i];
    };

    std::vector<size_t> *getShape() { return shape_; }

    TensorXXShape *
    getShuffledTensorXXShape(const std::vector<size_t> &shuffle) const;

    TensorXXShape *getTiledShape(int index, int n);
};

class TensorType {
  public:
    TensorType() = default;
    TensorType(const std::initializer_list<size_t> &dims) {
        assert(dims.size() <= max_dimensions && "illegal dims.");
        numDims_ = dims.size();
        int idx = 0;
        for (auto d : dims) {
            shape_[idx++] = d;
        }

        if (layout_ == mem_layout_t::layout_default && numDims_ == 4) {
            layout_ = mem_layout_t::layout_nhwc;
        }
    }

    TensorType(const std::initializer_list<size_t> &dims, DataType dtype,
               mem_layout_t layout)
        : dtype_(dtype), layout_(layout) {
        assert(dims.size() <= max_dimensions && "illegal dims.");
        numDims_ = dims.size();
        int idx = 0;
        for (auto d : dims) {
            shape_[idx++] = d;
        }

        if (layout_ == mem_layout_t::layout_default && numDims_ == 4) {
            layout_ = mem_layout_t::layout_nhwc;
        }
    }

    TensorType(std::vector<size_t> *shape) {
        numDims_ = shape->size();
        for (size_t i = 0; i < shape->size(); i++) {
            shape_[i] = shape->at(i);
        }
    }

    TensorType(const std::vector<size_t> &shape,
               DataType dtype = DataType::Float_t,
               mem_layout_t layout = mem_layout_t::layout_default)
        : dtype_(dtype), layout_(layout) {
        numDims_ = shape.size();
        for (size_t i = 0; i < shape.size(); i++) {
            shape_[i] = shape.at(i);
        }
    }

    ~TensorType() = default;

    int numDims() const { return numDims_; }
    size_t getDim(int idx) const { return shape_[idx]; }
    size_t size() const {
        size_t s = 1;
        for (int i = 0; i < numDims_; i++) {
            s *= shape_[i];
        }
        return s;
    }

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
    mem_layout_t layout_{mem_layout_t::layout_default};
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

    // TODO: to be depreciated
    DataType dataType_;
    TensorXXShape *shape_;
    mem_layout_t mem_layout_;

    TensorInitType initType_;
    TensorInitInfo initInfo_;

    int train_{0};

  public:
    Tensor() {
        shape_ = NULL;
        initType_ = TensorInitType::NONE;
    }

    // to be depreciated
    Tensor(TensorXXShape *shape, DataType dtype = DataType::Float_t,
           mem_layout_t layout = layout_default) {
        dataType_ = dtype;
        shape_ = shape;
        initType_ = TensorInitType::NONE;

        if (layout == layout_default) {
            // this framework use NHWC memory layout for 4D Tensors by default
            mem_layout_ =
                (shape_->getNDim() == 4) ? layout_nhwc : layout_default;
        } else {
            mem_layout_ = layout;
        }

        type_ = TensorType(shape_->getShape());
    }

    Tensor(const std::initializer_list<size_t> &shape,
           DataType dtype = DataType::Float_t,
           mem_layout_t layout = layout_default)
        : type_(shape, dtype, layout) {

        dataType_ = dtype;
        std::vector<size_t> *vec = new std::vector<size_t>();
        for (auto i : shape) {
            int v = i;
            vec->push_back(v);
        }
        shape_ = new TensorXXShape(vec);
        initType_ = TensorInitType::NONE;

        if (layout == layout_default) {
            // this framework use NHWC memory layout for 4D Tensors by default
            mem_layout_ =
                (shape_->getNDim() == 4) ? layout_nhwc : layout_default;
        } else {
            mem_layout_ = layout;
        }
    }

    Tensor(const std::vector<size_t> &dims, DataType dtype = DataType::Float_t,
           mem_layout_t layout = layout_default)
        : type_(dims, dtype, layout) {

        initType_ = TensorInitType::NONE;

        dataType_ = dtype;

        std::vector<size_t> *vec = new std::vector<size_t>(dims);
        shape_ = new TensorXXShape(vec);

        if (layout == layout_default) {
            // this framework use NHWC memory layout for 4D Tensors by default
            mem_layout_ =
                (shape_->getNDim() == 4) ? layout_nhwc : layout_default;
        } else {
            mem_layout_ = layout;
        }
    }

    Tensor(const TensorType &type) : type_(type) {

        dataType_ = type.getDataType();
        std::vector<size_t> *vec = new std::vector<size_t>();
        for (int i = 0; i < type.numDims(); i++) {
            vec->push_back(type.getDim(i));
        }
        shape_ = new TensorXXShape(vec);
        initType_ = TensorInitType::NONE;

        auto layout = type.getMemLayout();
        if (layout == layout_default) {
            // this framework use NHWC memory layout for 4D Tensors by default
            mem_layout_ =
                (shape_->getNDim() == 4) ? layout_nhwc : layout_default;
        } else {
            mem_layout_ = layout;
        }
    }

    Tensor(const Tensor &other) = delete;
    Tensor &operator=(const Tensor &other) = delete;

    ~Tensor() { destroy(); }

    void destroy() { getTensorXXShape()->destroy(); }

    void reset(TensorXXShape *shape, DataType dtype = DataType::Float_t,
               mem_layout_t layout = layout_default) {
        shape_ = shape;
        SWLOG_DEBUG(2) << "reset shape dims " << shape_->getNDim() << "\n";
        dataType_ = dtype;

        if (layout == layout_default) {
            // this framework use NHWC memory layout for 4D Tensors by default
            mem_layout_ =
                (shape_->getNDim() == 4) ? layout_nhwc : layout_default;
        } else {
            mem_layout_ = layout;
        }
    }
    void reset(const std::initializer_list<size_t> &shape,
               DataType dtype = DataType::Float_t,
               mem_layout_t layout = layout_default) {

        std::vector<size_t> *vec = new std::vector<size_t>();
        for (auto i : shape) {
            int v = i;
            vec->push_back(v);
        }
        shape_ = new TensorXXShape(vec);

        SWLOG_DEBUG(2) << "reset shape dims " << shape_->getNDim() << "\n";
        dataType_ = dtype;

        if (layout == layout_default) {
            // this framework use NHWC memory layout for 4D Tensors by default
            mem_layout_ =
                (shape_->getNDim() == 4) ? layout_nhwc : layout_default;
        } else {
            mem_layout_ = layout;
        }

        type_ = TensorType(shape, dtype, layout);
    }

    void reset(const TensorType &type) {

        type_ = type;

        dataType_ = type.getDataType();
        std::vector<size_t> *vec = new std::vector<size_t>();
        for (int i = 0; i < type.numDims(); i++) {
            vec->push_back(type.getDim(i));
        }
        shape_ = new TensorXXShape(vec);
        initType_ = TensorInitType::NONE;

        auto layout = type.getMemLayout();
        if (layout == layout_default) {
            // this framework use NHWC memory layout for 4D Tensors by default
            mem_layout_ =
                (shape_->getNDim() == 4) ? layout_nhwc : layout_default;
        } else {
            mem_layout_ = layout;
        }
    }

    Tensor *clone() const;
    TensorXXShape *
    getShuffledTensorXXShape(const std::vector<size_t> &shuffle) const;
    std::vector<size_t>
    getShuffledDims(const std::vector<size_t> &shuffle) const;
    TensorType getShuffledTensorType(const std::vector<size_t> &shuffle) const;

    TensorType getTiledTensorType(int idx, int n) {
        return type_.getTiledTensorType(idx, n);
    }

    DataType getDataType() const { return dataType_; }

    // int getNDim() const { return shape_->getNDim(); }
    // size_t getDim(int dim) const { return shape_->getDim(dim); }
    const std::vector<size_t> getDims() const {
        std::vector<size_t> dims(getNDim(), 0);
        for (int i = 0; i < getNDim(); i++)
            dims[i] = getDim(i);
        return dims;
    }
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

    TensorXXShape *getTensorXXShape() const { return shape_; }
    // size_t size() const { return shape_->size(); }
    // size_t getSizeInBytes() const;

    TensorType getType() const { return type_; }
    size_t size() const { return type_.size(); }
    size_t getSizeInBytes() const { return type_.getSizeInBytes(); }

    // void setMemLayout(mem_layout_t layout) { mem_layout_ = layout; }
    // mem_layout_t getMemLayout() const { return mem_layout_; }
    // std::string getMemLayoutTag() const;
    void setMemLayout(mem_layout_t layout) {
        mem_layout_ = layout;
        type_.setMemLayout(layout);
    }
    mem_layout_t getMemLayout() const { return type_.getMemLayout(); }
    std::string getMemLayoutTag() const { return type_.getMemLayoutTag(); }
};

} // namespace swc

#endif
