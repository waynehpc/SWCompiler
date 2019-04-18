/*************************************************************************
	> File Name: common.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue 04 Dec 2018 08:09:21 AM UTC
 ************************************************************************/

#ifndef _COMMON_H
#define _COMMON_H

#include <memory>
#include <vector>

enum class DataType { Float_t, Double_t, Int8_t, Int32_t };

enum ParallelStrategy { SLICE, TILING };  

struct TrainingProfile {
    float lr{0.001};
    float decay{0.001};
    float momentum{0.9};
    size_t batch{1};
};

enum OpType { TENSOR_OP, BASIC_OP, DL_OP };

enum NodeType { TENSOR_NODE, OP_NODE };

enum TensorType {
    D5 = 5,
    D4 = 4,
    D3 = 3,
    D2 = 2,
    D1 = 1,
    D0 = 0,
    UNKNOWN = -1
};

enum class TensorInitType { NONE, CONSTANT, ZERO, XAVIER, FILE, PARENTOP };
enum class DeviceType : int { CPU, GPU };

enum class PrintStreamType { COUT, FILE };

struct Device {
    DeviceType type;
    int id;
    Device(DeviceType t = DeviceType::CPU, int i = 0) : type(t), id(i) {}
    friend bool operator==(const Device &x, const Device &y) {
        return x.type == y.type && x.id == y.id;
    }
};
namespace std {
template <> struct hash<Device> {
    size_t operator()(const Device &d) const {
        auto h1 = std::hash<int>{}(static_cast<int>(d.type));
        auto h2 = std::hash<int>{}(d.id);
        return h1 ^ h2;
    }
};
} // namespace std

#define NCHW2NHWC                                                              \
    { 0, 2, 3, 1 }
#define NHWC2NCHW                                                              \
    { 0, 3, 1, 2 }

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname)                                           \
    char gInstantiationGuard##classname;                                       \
    template class classname<float>;                                           \
    template class classname<double>

template <typename Dtype> class SWMem {
  private:
    size_t _len;
    Dtype *_data;

  public:
    SWMem(size_t len, Dtype *data);
    ~SWMem();
    Dtype *data();
    Dtype *mutable_data();
};

template <typename U, typename V>
int delVecMember(std::vector<U> &vec, V &del) {
    int delDone = 0;
    for (typename std::vector<U>::iterator it = vec.begin(); it != vec.end();) {
        if (*it == del) {
            it = vec.erase(it);
            delDone = 1;
            break;
        } else {
            ++it;
        }
    }
    return delDone;
}

#endif
