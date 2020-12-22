/*************************************************************************
	> File Name: Engine.h
	> Author: wayne
	> Mail:  
	> Created Time: Sat 14 Sep 2019 10:30:55 AM UTC
 ************************************************************************/
#ifndef ENGINE_H_
#define ENGINE_H_
#include <string>
#include "common.h"

namespace swc {
namespace codegen{
class Codegen;
}
class IRGraph;

class Engine {
public:
    Engine() = delete;
    Engine(IRGraph *graph) : graph_(graph) {}

    IRGraph* getGraph() { return graph_; }

    /// backend-specific transformations on graphs
    virtual void compile();
    
    // Infer
    void runInferPasses();
    // Train
    void runTrainPasses();

    // parallelization
    void runParallelPasses();

    virtual void transformForMKLDNN();
    virtual void transformForCUDA() {}
    virtual void optimize();
    std::string genCode(std::string out="Graph.cpp");
    std::string genCode(Config& config, std::string out="Graph.cpp");

    void setCodegenerator(codegen::Codegen *codegen) { generator_ = codegen; }
private:
    IRGraph *graph_; // if leverage leved-graph, we may change this entry graph to model to maintain globally information like sem_table.
    codegen::Codegen *generator_{nullptr}; 
};

} // swc
#endif
