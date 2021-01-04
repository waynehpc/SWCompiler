/*************************************************************************
	> File Name: SWC.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Wed 05 Dec 2018 03:37:36 AM UTC
 ************************************************************************/

#ifndef _SWC_H
#define _SWC_H

#include "SWDSL.h"
#include "SWLOG.h"
#include "common.h"

#include "op/dlOp/dlOp.h"
#include "op/tensorOp/tensorOps.h"
#include "tensor/tensor.h"

#include "graphIR/IRGraph.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"

#include "pass/Label.h"
#include "pass/Optimizer.h"

#include "tool/dotGen.h"

#include "diff/AutoDiff.h"


#include "codegen/Codegen.h"

#include "engine/Engine.h"

#include "pass/OptimizePass.h"
#include "pass/LabelingPass.h"
#include "pass/LoweringPass.h"
#include "pass/EliminationPass.h"
#include "pass/RenamingNodePass.h"
// #include "pass/ParallelingPass.h"
#include "pass/ParallelLabelingPass.h"
#include "pass/ParallelLoweringPass.h"
#include "pass/AutodiffPass.h"

#endif
