/*************************************************************************
    > File Name: RenamingNodePass.h
    > Author: wayne
    > Mail:
    > Created Time: 一  4/ 8 16:32:26 2019
 ************************************************************************/
#ifndef _RENAMINGNODEPASS_H_
#define _RENAMINGNODEPASS_H_

#include "OptimizePass.h"
#include "SWLOG.h"

#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"

#include <cassert>
#include <sstream>
#include <unordered_map>

namespace swc {
namespace pass {

class UniqueName {
    std::unordered_map<std::string, int> names_map_;

  public:
    std::string operator()(const std::string &inputName) {
        assert(!inputName.empty() && "inputName empty");
        std::string name;
        for (const char c : inputName) {
            if (c == '/' || c == '.' || c == '-') {
                name.push_back('_');
            } else {
                name.push_back(c);
            }
        }

        auto iter = names_map_.find(name);
        if (iter != names_map_.end()) {
            std::string uname = name;
            while (names_map_.count(uname) != 0) {
                std::ostringstream os;
                os << name << (++iter->second);
                uname = os.str();
            }
            name = uname;
        }
        SWLOG_DEBUG(1) << "orig_name " << inputName << "--> uniq_name " << name << "\n";
        names_map_[name] = 0;
        return name;
    }

    void clear() { names_map_.clear(); }
};

class RenamingNodePass : public OptimizePass {
    using OptimizePass::_graph;
    UniqueName uniqueName;

  public:
    RenamingNodePass(IRGraph *graph) : OptimizePass(graph){};
    ~RenamingNodePass() {}
    void setGraph(IRGraph *graph) {
        _graph = graph;
        uniqueName.clear();
    }

    void run() {
        SWLOG_DEBUG(4) << "start RenamingNodePass ...\n";
        int nTensorNodes = _graph->tensorNodeNum();
        int nOpNodes = _graph->opNodeNum();
        for (int i = 0; i < nTensorNodes; i++) {
            TensorNode *node = _graph->getTensorNode(i);
            std::string uname = uniqueName(node->name());
            node->setName(uname);
        }

        for (int i = 0; i < nOpNodes; i++) {
            OpNode *node = _graph->getOpNode(i);
            std::string uname = uniqueName(node->name());
            node->setName(uname);
        }
        SWLOG_DEBUG(4) << "end RenamingNodePass ...\n\n";
    }
};

} // namespace pass
} // namespace swc
#endif
