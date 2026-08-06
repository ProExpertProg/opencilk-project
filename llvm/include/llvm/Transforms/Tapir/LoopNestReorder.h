//===- LoopNestReorder.h - Reorder nested parallel loops -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass reorders the iterations in a perfect loop nest of Tapir loops.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LOOPNESTREORDER_H
#define LLVM_LOOPNESTREORDER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class LoopNestReorderPass : public PassInfoMixin<LoopNestReorderPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_LOOPNESTREORDER_H
