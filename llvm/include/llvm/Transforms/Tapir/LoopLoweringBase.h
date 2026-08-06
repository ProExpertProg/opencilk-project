//===- LoopLoweringBase.h - Base class for Loop Lowering with common
// functionality -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class provides common functionality for passes that do any kind of loop
//// lowering.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LOOPLOWERINGBASE_H
#define LLVM_LOOPLOWERINGBASE_H

#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

namespace llvm {

class LoopLoweringBase {
  // Get the set of basic blocks within the task of Tapir loop \p TL.  The \p
  // TaskBlocks vector stores all of these basic blocks.  The \p ReattachBlocks
  // set identifies which blocks are terminated by a reattach instruction that
  // terminates the task.  The \p DetachedRethrowBlocks set identifies which
  // blocks are terminated by detached-rethrow instructions that terminate the
  // task.  Entry points to shared exception-handling code is stored in the
  // \p SharedEHEntries set.
  //
  // This method relies on being executed on the Tapir loops in a function in
  // post order.
  void
  getTapirLoopTaskBlocks(TapirLoopInfo *TL,
                         std::vector<BasicBlock *> &TaskBlocks,
                         SmallPtrSetImpl<BasicBlock *> &ReattachBlocks,
                         SmallPtrSetImpl<BasicBlock *> &DetachedRethrowBlocks,
                         SmallPtrSetImpl<BasicBlock *> &SharedEHEntries,
                         SmallPtrSetImpl<BasicBlock *> &UnreachableExits);

public:
  LoopLoweringBase(Function &F, DominatorTree &DT, LoopInfo &LI, TaskInfo &TI,
                   ScalarEvolution &SE, AssumptionCache &AC,
                   TargetTransformInfo &TTI, OptimizationRemarkEmitter &ORE)
      : AC(AC), DT(DT), F(F), LI(LI), ORE(ORE), SE(SE), TI(TI), TTI(TTI) {}

  ~LoopLoweringBase() {
    for (TapirLoopInfo *TL : TapirLoops)
      delete TL;
    TapirLoops.clear();
    TaskToTapirLoop.clear();
    LoopToTapirLoop.clear();
  }

protected:
  AssumptionCache &AC;
  DominatorTree &DT;
  Function &F;
  LoopInfo &LI;
  OptimizationRemarkEmitter &ORE;
  ScalarEvolution &SE;
  TaskInfo &TI;
  TargetTransformInfo &TTI;

  std::vector<TapirLoopInfo *> TapirLoops;

private:
  DenseMap<Loop *, TapirLoopInfo *> LoopToTapirLoop;
  DenseMap<Task *, TapirLoopInfo *> TaskToTapirLoop;

protected:
  // Outline Tapir loop \p TL into a helper function.  The \p Args set specified
  // the arguments to that helper function.  The map \p VMap will store the
  // mapping of values in the original function to values in the outlined
  // helper.
  Function *createHelperForTapirLoop(TapirLoopInfo *TL, ValueSet &Args,
                                     unsigned IVArgIndex,
                                     unsigned LimitArgIndex, Module *DestM,
                                     ValueToValueMapTy &VMap,
                                     ValueToValueMapTy &InputMap);

  // Record a Tapir loop defined by loop \p L and task \p T.
  TapirLoopInfo *createTapirLoop(Loop *L, Task *T) {
    TapirLoops.push_back(new TapirLoopInfo(L, T, ORE));
    TaskToTapirLoop[T] = TapirLoops.back();
    LoopToTapirLoop[L] = TapirLoops.back();
    return TapirLoops.back();
  }

  // Forget the specified Tapir loop \p TL.
  void forgetTapirLoop(TapirLoopInfo *TL) {
    TaskToTapirLoop.erase(TL->getTask());
    LoopToTapirLoop.erase(TL->getLoop());
  }

  // For all recorded Tapir loops, determine the function arguments and inputs
  // for the outlined helper functions for those loops.
  //
  // The \p LoopArgs map will store the function arguments for these outlined
  // loop helpers.  The \p LoopInputs map will store the corresponding arguments
  // for calling those outlined helpers from the parent function.  The \p
  // LoopArgStarts map will store the instruction in the parent where new code
  // for computing these outlined-helper-call arguments is first inserted.
  void getAllTapirLoopInputs(
      DenseMap<Loop *, ValueSet> &LoopInputSets,
      DenseMap<Loop *, SmallVector<Value *, 3>> &LoopCtlArgs,
      DenseMap<Loop *, SmallVector<Value *, 3>> &LoopCtlInputs);

  // If loop \p L defines a recorded Tapir loop, returns the Tapir loop info for
  // that Tapir loop.  Otherwise returns null.
  TapirLoopInfo *getTapirLoop(Loop *L) {
    if (!LoopToTapirLoop.count(L))
      return nullptr;
    return LoopToTapirLoop[L];
  }

  // If task \p T defines a recorded Tapir loop, returns the Tapir loop info for
  // that Tapir loop.  Otherwise returns null.
  TapirLoopInfo *getTapirLoop(Task *T) {
    if (!TaskToTapirLoop.count(T))
      return nullptr;
    return TaskToTapirLoop[T];
  }

  // Gets the Tapir loop that contains basic block \p B, i.e., the Tapir loop
  // for the loop associated with \p B.
  TapirLoopInfo *getTapirLoop(const BasicBlock *B) {
    return getTapirLoop(LI.getLoopFor(B));
  }
};

} // namespace llvm

#endif // LLVM_LOOPLOWERINGBASE_H
