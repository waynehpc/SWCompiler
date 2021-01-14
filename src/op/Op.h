/*************************************************************************
        > File Name: Op.h
        > Author: cryinlaugh
        > Mail: cryinlaugh@gmail.com
        > Created Time: 二 12/ 4 15:57:08 2018
 ************************************************************************/

#ifndef _OP_H
#define _OP_H

#include <string>
#include <unordered_map>

#include "SWLOG.h"
#include "common.h"

namespace swc {

// Forward declarations
class Tensor;
class IRGraph;
class IRNode;
class OpNode;
class ParallelGen;

namespace op {

enum activation_type {
    SWC_ACTIVATION_RELU,
    SWC_ACTIVATION_TANH,
    SWC_ACTIVATION_SIGMOID
};

class Op {
  public:
    Op(OpType opType = BASIC_OP, int nInput = 0, int nOutput = 0,
       std::string opClassName = NULL)
        : _opType(opType), _nInput(nInput), _nOutput(nOutput),
          _opClassName(opClassName) {

        _nInputTensor = nInput;
        _nOutputTensor = nOutput;
    };

    ~Op(){};

    virtual void destroy(){};

    void addInputTensor(Tensor *inputTensor) {
        _inputTensors.push_back(inputTensor);
        _nInputTensor++;
    }

    void addOutputTensor(Tensor *outputTensor) {
        _outputTensors.push_back(outputTensor);
        _nOutputTensor++;
    }
    bool check();

    OpType getOpType() { return _opType; }

    // aggregate information for dotGen or Debug
    // calls getOpName, getnInput/Output etc.
    // some derived classes may override this
    virtual std::string getOpInfo();

    // Logically these function is better in OpNode, but need many if else
    // because different operators do not derive from OpNode, but are different
    // in Op member
    // if use virtual foo() = 0; then all Op must derive this
    virtual size_t getCost(OpNode *, Config &config) { return 0; }
    virtual std::string getCostTrace(OpNode *, Config &config) { return ""; }

    inline const std::string getOpName() { return _opClassName; }
    inline int getnInput() { return _nInput; }
    inline int getnOutput() { return _nOutput; }

    // for lowering
    virtual void lowering(IRGraph *graph, IRNode *node) {
        SWLOG_DEBUG(100) << "Lowering unimplemented in base Op class"
                         << std::endl;
    }

    virtual void checkValid(OpNode *node);

    virtual void autoDiff(IRGraph *graph, IRNode *opNode,
                          std::unordered_map<IRNode *, IRNode *> &gradNodeMap) {
        SWLOG_DEBUG(100) << "OpType [" << this->getOpName()
                         << "] autoDiff() unimplemented, pass" << std::endl;
        exit(0);
    }

    virtual void einsumLowering(IRGraph *graph, IRNode *node) {
        SWLOG_DEBUG(100) << "EinsumLowering unimplemented in base Op class"
                         << std::endl;
    }

    virtual void outTensorTypeGen(OpNode *node, size_t index, Tensor *tensor);
    /*
    Op *clone() const {
        return new Op(_opType, _nInput, _nOutput, _opClassName);
    }
    */

    virtual void setAttr(int ndim) {}
    virtual void setIONDims(std::initializer_list<int> indims,
                            std::initializer_list<int> ondims) {}
    inline int getInputDims(int n) { return _inputNDims[n]; }
    inline int getOutputDims(int n) { return _outputNDims[n]; }

    virtual void setEinReps(std::initializer_list<std::string> reps) {}
    inline int getEinOp() { return _einOp; }

  protected:
    /* The following variables are constant values in a specific Op Class
       indicating what kind of input/output tensors it should keep. */

    const OpType _opType;          // enum var, define the type of operation
    const int _nInput;             // nums of input  tensor
    const int _nOutput;            // nums of output tensor
    std::vector<int> _inputNDims;  // input  tensors
    std::vector<int> _outputNDims; // output tensors

    const std::string _opClassName;

    /* The following variables indicating the real input/output tensors
       that the Op really have, its useful in analyses or ref-code-generation.
     */

    int _nInputTensor;
    int _nOutputTensor;
    std::vector<Tensor *> _inputTensors;
    std::vector<Tensor *> _outputTensors;

    /* Edit by zwl @ 20190705
     * The following variables indicate the dimension-level relation shape of
     * input/output tensors, which is designed as a generalized abstraction for
     * tensors (similar to a Einsum expression), and is used for analyzing and
     * generating parallel strategies for a tensor graph.
     *
     * _einOp is a label variable indicats whether the Op can be discribed using
     * Einsum expression. _einRep is the vecter stores the Einsum expression for
     * input and output tensors seperately, _parallelDim is a vector of unsigned
     * integers which store the parallelizable label of all input tensors, used
     * for generate parallelization strategy, e.g. input tensor is expressed as
     * "ijk" and has a parallel label x, then dimension i is parallelizable <==>
     * ((x>>2) & 1 == 1) dimension j is parallelizable <==> ((x>>1) & 1 == 1)
     *      dimension k is parallelizable <==> ((x>>0) & 1 == 1)
     *
     */
    int _einOp{0};
    std::vector<std::string> _einRep;
    std::vector<int> _parallelDim;

    friend class swc::ParallelGen;
};

} // namespace op
} // namespace swc

#endif
