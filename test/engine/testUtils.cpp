/*************************************************************************
 *    > File Name: testUtils.cpp
 *    > Author:  wayne
 *    > mail:
 *    > Created Time: Sun 03 Jan 2021 01:00:45 PM UTC
 ************************************************************************/

#include <iostream>

#include "SWC.h"

using namespace swc;
using namespace swc::op;
using namespace std;

int main() {

    TENSOR(data0, 8, 50);
    data0_Tensor->setTensorInit(TensorInitType::FILE,
                                "input/mnist_images_8.bin");

    cout << data0_Tensor->size() << endl;
    cout << data0_Tensor->getSizeInBytes() << endl;
    cout << int(data0_Tensor->getDataType()) << endl;

    data0_Tensor->reset({8, 64}, DataType::Double_t,
                        mem_layout_t::layout_default);
    cout << data0_Tensor->size() << endl;
    cout << data0_Tensor->getSizeInBytes() << endl;
    cout << int(data0_Tensor->getDataType()) << endl;
    cout << data0_Tensor->getMemLayoutTag() << endl;

    TensorType tt = data0_Tensor->getType();
    cout << tt.size() << " " << tt.getSizeInBytes() << " "
         << tt.getMemLayoutTag() << endl;

    return 0;
}
