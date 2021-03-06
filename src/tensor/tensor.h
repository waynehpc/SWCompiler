/*************************************************************************
	> File Name: tensor.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tues. 12/ 4 15:53:19 2018
 ************************************************************************/

#ifndef _TENSOR_H
#define _TENSOR_H

#include "common.h"
#include "SWLOG.h"
#include <string>
#include <iostream>
#include <cassert>

namespace swc {

class TensorShape {
  private:
    int _ndim;
    std::vector<size_t> *shape_;

  public:
    TensorShape(std::vector<size_t> *shape);
    TensorShape(const std::initializer_list<size_t> &shape) {
        shape_ = new std::vector<size_t>();
        for (auto i : shape) {
            shape_->push_back(i);
        }
        _ndim = shape_->size();
    }

    ~TensorShape() {}
    void destroy() {}

    int getNDim() const;
    size_t getDim(int idx) const;
    size_t size() const;
    
    void setShape(const std::vector<size_t>& shape) {
        _ndim = shape.size();
        shape_->resize(_ndim);
        for (int i = 0; i < _ndim; i++)
            (*shape_)[i] = shape[i];
    };

    TensorShape *
    getShuffledTensorShape(const std::vector<size_t> &shuffle) const;
    
    TensorShape* getTiledShape( int index, int n);
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
    DataType dataType_;
    TensorShape *shape_;
    mem_layout_t mem_layout_;

    TensorInitType initType_;
    TensorInitInfo initInfo_;

    int train_{0};

  public:
    Tensor() {
        shape_ = NULL;
        initType_ = TensorInitType::NONE;
    }

    Tensor(TensorShape *shape, DataType dtype = DataType::Float_t, mem_layout_t layout = layout_default) {
        dataType_ = dtype;
        shape_ = shape;
        initType_ = TensorInitType::NONE;

        if(layout == layout_default) {
            // this framework use NHWC memory layout for 4D Tensors by default
            mem_layout_ = (shape_->getNDim() == 4) ? layout_nhwc : layout_default; 
        } else {
            mem_layout_ = layout; 
        }
    }

    Tensor(const std::initializer_list<size_t> &shape,
            DataType dtype = DataType::Float_t, mem_layout_t layout = layout_default) {
        dataType_ = dtype;
        std::vector<size_t> *vec = new std::vector<size_t>();
        for (auto i : shape) {
            int v = i;
            vec->push_back(v);
        }
        shape_ = new TensorShape(vec);
        initType_ = TensorInitType::NONE;

        if(layout == layout_default) {
            // this framework use NHWC memory layout for 4D Tensors by default
            mem_layout_ = (shape_->getNDim() == 4) ? layout_nhwc : layout_default; 
        } else {
            mem_layout_ = layout; 
        }
    }

    ~Tensor(){ destroy(); };

    void destroy() {
        getTensorShape()->destroy();
    }

    void reset(TensorShape *shape, DataType dtype = DataType::Float_t, mem_layout_t layout = layout_default) {
        shape_ = shape;
        SWLOG_DEBUG(2) << "reset shape dims " << shape_->getNDim() << "\n";
        dataType_ = dtype;

        if(layout == layout_default) {
            // this framework use NHWC memory layout for 4D Tensors by default
            mem_layout_ = (shape_->getNDim() == 4) ? layout_nhwc : layout_default; 
        } else {
            mem_layout_ = layout; 
        }
    }
    Tensor *clone() const;
    TensorShape *
    getShuffledTensorShape(const std::vector<size_t> &shuffle) const;
    std::vector<size_t> 
    getShuffledDims(const std::vector<size_t> &shuffle) const;

    DataType getDataType() { return dataType_; }

    int getNDim() const { return shape_->getNDim(); };
    size_t getDim(int dim) const { return shape_->getDim(dim); };
    const std::vector<size_t> getDims() const {
        std::vector<size_t> dims;
        for (int i = 0; i < getNDim(); i++)
            dims.push_back(getDim(i));
        return dims;
    }

    std::pair<size_t, size_t> viewAs2D(int n);

    void setTensorInit(TensorInitType type, float value = 0);
    void setTensorInit(TensorInitType type, std::string file,
                       size_t offset = 0);

    void setTensorInit(TensorInitType type, TensorInitInfo info);

    TensorInitType getTensorInitType() { return initType_; }
    TensorInitInfo getTensorInitInfo() const { return initInfo_; }

    void setTraining(int train) { train_ = train; }
    int getTraining() const { return train_; }

    TensorShape *getTensorShape() const { return shape_; }
    size_t size() const { return shape_->size(); }
    size_t getSizeInBytes() const;

    void setMemLayout(mem_layout_t layout) { mem_layout_ = layout; }
    mem_layout_t getMemLayout() const { return mem_layout_; }
    std::string getMemLayoutTag() const;
};

} // namespace swc

#endif
