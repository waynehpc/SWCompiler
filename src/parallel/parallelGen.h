/*************************************************************************
	> File Name: parallelGen.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue Jul  9 07:15:52 2019
 ************************************************************************/

#ifndef _PARALLELGEN_H
#define _PARALLELGEN_H

#include "op/Op.h"
#include <map>
#include <string>
#include <vector>
namespace swc {

class ParallelGen {

  public:
    static std::vector<std::vector<int>> generateStgy(OpNode *node);
    static std::vector<int> generateDataParStgy(OpNode *opnode);
};

} // namespace swc

#endif
