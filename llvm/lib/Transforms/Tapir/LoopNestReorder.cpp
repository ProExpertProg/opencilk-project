//===- LoopNestReorder.cpp - Reorder nested parallel loops -------------===//
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

#include "llvm/Transforms/Tapir/LoopNestReorder.h"
#include "llvm/Analysis/AssumptionCache.h"

#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Tapir/LoopLoweringBase.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/LoopRotationUtils.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

#define LNR_NAME "LoopNestReorder"
#define DEBUG_TYPE LNR_NAME

using namespace llvm;

class LoopNestReorder : protected LoopLoweringBase {
public:
  LoopNestReorder(Function &F, DominatorTree &DT, LoopInfo &LI, TaskInfo &TI,
                  ScalarEvolution &SE, AssumptionCache &AC,
                  TargetTransformInfo &TTI, OptimizationRemarkEmitter &ORE)
      : LoopLoweringBase(F, DT, LI, TI, SE, AC, TTI, ORE) {}

  bool run();

  // If loop \p L defines a recorded Tapir loop, returns the Tapir loop info for
  // that Tapir loop. Otherwise, it constructs new TLI for a Tapir loop or
  // returns nullptr for a loop that isn't that.
  TapirLoopInfo *getOrCreateTapirLoopInfo(Loop *L) {
    auto *TL = getTapirLoop(L);

    if (TL != nullptr)
      return TL;

    auto *T = getTaskIfTapirLoopStructure(L, &TI);
    if (T == nullptr)
      return nullptr;

    return createTapirLoop(L, T);
  }

  void collectLoopsToTransform(
      SmallVectorImpl<std::unique_ptr<LoopNest>> &CollectedNests);
  void
  collectLoopRecursively(SmallVectorImpl<std::unique_ptr<LoopNest>> &Collected,
                         Loop *L);
  void
  filterAndPrepareLoopNests(const SmallVector<std::unique_ptr<LoopNest>> &Nests,
                            SmallVector<SmallVector<Loop *>> &FilteredNests);
};

// TODO get rid of recursion
void LoopNestReorder::collectLoopRecursively(
    SmallVectorImpl<std::unique_ptr<LoopNest>> &Collected, Loop *L) {

  // We always want to try and capture the largest nest possible. That's why we
  // start at the top and recursively look at subloops that aren't a part of a
  // perfect nest.

  auto LN = LoopNest::getLoopNest(*L, SE, TI, true);
  auto Depth = LN->getMaxPerfectDepth();
  if (Depth > 1) {
    // Loop has perfect nest
    LLVM_DEBUG(dbgs() << "Collecting loop " << L->getName()
                      << " with nest depth " << Depth << "\n");

    // Skip all loops in the nest
    assert(LN->getLoops().size() >= Depth);
    L = LN->getLoops()[Depth - 1];

    Collected.push_back(std::move(LN));
  }

  // LN is no longer valid as it might have been moved to Collected.
  // L should not have any perfectly nested loops anymore.
  assert(LoopNest::getMaxPerfectDepth(*L, SE, TI, true) == 1);

  // Check all subloops recursively.
  LLVM_DEBUG(dbgs() << "Checking subloops of loop " << L->getName() << "\n");
  for (Loop *Sub : *L) {
    collectLoopRecursively(Collected, Sub);
  }
}

void LoopNestReorder::collectLoopsToTransform(
    SmallVectorImpl<std::unique_ptr<LoopNest>> &CollectedNests) {
  for (Loop *TopLevelLoop : LI) {
    collectLoopRecursively(CollectedNests, TopLevelLoop);
  }
}

void LoopNestReorder::filterAndPrepareLoopNests(
    const SmallVector<std::unique_ptr<LoopNest>> &Nests,
    SmallVector<SmallVector<Loop *>> &FilteredNests) {

  // Check that the loop in the nest is appropriate and prepare it for
  // outlining.

  // Add accumulation to FilteredNests
  auto AddToFilteredNests = [&](SmallVector<Loop *> &&Acc) {
    if (Acc.size() > 1)
      FilteredNests.push_back(std::move(Acc));
  };

  for (auto &Nest : Nests) {
    // Used to collect appropriate loops and for splitting Nest in case some
    // loop in the middle of Nest isn't appropriate
    SmallVector<Loop *> CurrentNest{};
    for (size_t I = 0; I < Nest->getNumLoops(); ++I) {
      // The loop must be parallel.
      auto *L = Nest->getLoop(I);
      TapirLoopInfo *TL = getOrCreateTapirLoopInfo(L);
      if (TL != nullptr) {
        PredicatedScalarEvolution PSE(SE, *L);
        bool const CanOutline =
            TL->prepareForOutlining(DT, LI, TI, PSE, AC, LNR_NAME, ORE, TTI);

        if (CanOutline) {
          CurrentNest.push_back(L);
          continue;
        }

        // Loop ignored
        forgetTapirLoop(TL);
      }

      // Loop is not appropriate. Add the loops in the nest so far.
      AddToFilteredNests(std::move(CurrentNest));
      LLVM_DEBUG(dbgs() << "Loop " << L->getName() << " is not appropriate\n");

      CurrentNest.clear();
    }

    // The remaining part of the nest gets added too.
    AddToFilteredNests(std::move(CurrentNest));
  }
}

bool LoopNestReorder::run() {
  // check
  LLVM_DEBUG(dbgs() << "LoopNestReorder on function " << F.getName() << "\n");

  // Collect perfect loop nests for transformation
  SmallVector<std::unique_ptr<LoopNest>> Nests{};
  collectLoopsToTransform(Nests);

  LLVM_DEBUG(dbgs() << "Collected nests:\n");
  for (auto &LN : Nests) {
    LLVM_DEBUG(dbgs() << *LN << "\n");
  }

  // Let's filter out the nests we don't want to transform, and prepare the
  // loops for outlining. For now, those are nests where:
  //  - all loops are parallel - not necessary for DAC lowering TODO(lgovedic)
  // If there's a large nest that doesn't fulfill the criteria, we might split
  // it up into nests that do.
  SmallVector<SmallVector<Loop *>> FilteredNests;
  filterAndPrepareLoopNests(Nests, FilteredNests);
  LLVM_DEBUG({
    dbgs() << "Filtered nests:\n";
    for (auto &LN : FilteredNests) {
      for (auto *L : LN) {
        dbgs() << L->getName() << " ";
      }
      dbgs() << "\n";
    }
  });

  if (FilteredNests.empty())
    return false;

  DenseMap<Loop *, ValueSet> LoopInputSets;
  DenseMap<Loop *, SmallVector<Value *, 3>> LoopCtlArgs;
  DenseMap<Loop *, SmallVector<Value *, 3>> LoopCtlInputs;

  getAllTapirLoopInputs(LoopInputSets, LoopCtlArgs, LoopCtlInputs);

  for (auto &Nest : FilteredNests) {
    /// This variable contains all of the inputs to inner loops that are defined
    /// in their parents and hence aren't inputs to the parent.
    /// The key is the address of the topmost loop that has this input
    /// (its children might use it too).
    DenseMap<Loop *, SmallVector<Value *>> NestedInputs;

    // Check that there are no weird loop inputs
    for (unsigned I = 1; I < Nest.size(); I++) {
      // We want to collect all inputs to the current loop that are defined in
      // Prev. For that, we exclude:
      //  - all inputs for the parent loop
      //  - the parent loop IV

      Loop *Prev = Nest[I - 1];
      auto *PrevTLI = getTapirLoop(Prev);
      unsigned MatchedInputs = 0;

      for (auto *V : LoopInputSets[Nest[I]]) {
        if (LoopInputSets[Prev].count(V)) {
          MatchedInputs++;
          continue;
        }

        if (PrevTLI->getPrimaryInduction().first == V) {
          continue;
        }

        LLVM_DEBUG(dbgs() << "In loop " << Nest[I]->getName()
                          << ", value not an input to parent nor "
                             "is it the parent IV: "
                          << *V << "\n");

        NestedInputs[Nest[I]].push_back(V);
      }

      if (MatchedInputs < LoopInputSets[Prev].size()) {
        // Parent has extra inputs that aren't in the current loop
        for (auto *V : LoopInputSets[Prev]) {
          if (LoopInputSets[Nest[I]].count(V)) {
            continue;
          }

          auto *CtlIn = std::find(LoopCtlInputs[Prev].begin(),
                                  LoopCtlInputs[Prev].end(), V);
          if (CtlIn != LoopCtlInputs[Prev].end()) {
            continue;
          }

          LLVM_DEBUG(dbgs() << "In loop " << Nest[I]->getName()
                            << ", value not an input to current loop nor is it "
                               "a LoopCtlInput: "
                            << *V << "\n");
          // TODO this might actually not be a problem,
          //  std::terminate();
        }
      }
    }

    LLVM_DEBUG({
    for (Loop *L : Nest) {
      if (NestedInputs[L].empty())
        continue;

      dbgs() << "Loop " << L->getName() << " NestedInputs\n";
      for (Value *V : NestedInputs[L])
        dbgs() << "\t" << *V << "\n";
    }
    });


  }

  // TODO fixupHelperInputs and createHelperForTapirLoop()
  //  for inner loop

  // TODO
  //  TB:
  //  - IVs for loops in the loop nest
  //  - IVs are phi nodes in the header
  //  - pass individual arguments

  // TODO:
  //  - Create new function: LoopSpawningTI
  //  - Extract body BBs into said function
  //    - will need to add dimension args/params
  //  - Replace BBs with a function call
  // TB notes:
  // - needs to be a module pass
  // - either simplify & rotate, or make sure pass order is right
  // - find all loop inputs
  // - fixupHelperInputs: loop inputs and IV args
  // - rewrite uses of trip count not used for IV condition (n inside the loop)
  // - serialize: courtesy of Tapir
  // - post process calls implementDACIter...

  return true;
}

PreservedAnalyses LoopNestReorderPass::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  const SimplifyQuery SQ(M.getDataLayout());

  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetDT = [&FAM](Function &F) -> DominatorTree & {
    return FAM.getResult<DominatorTreeAnalysis>(F);
  };
  auto GetLI = [&FAM](Function &F) -> LoopInfo & {
    return FAM.getResult<LoopAnalysis>(F);
  };
  auto GetTI = [&FAM](Function &F) -> TaskInfo & {
    return FAM.getResult<TaskAnalysis>(F);
  };
  auto GetSE = [&FAM](Function &F) -> ScalarEvolution & {
    return FAM.getResult<ScalarEvolutionAnalysis>(F);
  };
  auto GetAC = [&FAM](Function &F) -> AssumptionCache & {
    return FAM.getResult<AssumptionAnalysis>(F);
  };
  auto GetTTI = [&FAM](Function &F) -> TargetTransformInfo & {
    return FAM.getResult<TargetIRAnalysis>(F);
  };
  auto GetTLI = [&FAM](Function &F) -> TargetLibraryInfo & {
    return FAM.getResult<TargetLibraryAnalysis>(F);
  };
  auto GetORE = [&FAM](Function &F) -> OptimizationRemarkEmitter & {
    return FAM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  };

  SmallVector<Function *, 8> WorkList;
  bool Changed = false;
  for (Function &F : M)
    if (!F.empty())
      WorkList.push_back(&F);

  // Transform all loops into simplified, rotated, and LCSSA form before we
  // process them.
  for (Function *F : WorkList) {
    LoopInfo &LI = GetLI(*F);
    DominatorTree &DT = GetDT(*F);
    ScalarEvolution &SE = GetSE(*F);
    const TargetTransformInfo &TTI = GetTTI(*F);
    SmallVector<Loop *, 8> LoopWorkList;
    for (Loop *L : LI) {
      Changed |= simplifyLoop(L, &DT, &LI, &SE, &GetAC(*F), nullptr,
                              /* PreserveLCSSA */ false);

      // TODO is this the right order for rotating loops?
      for (Loop *SubLoop : post_order(L)) {
        Changed |= LoopRotation(SubLoop, &LI, &TTI, &GetAC(*F), &DT, &SE,
                                /* TODO MSSAUpdater */ nullptr, &GetTI(*F), SQ,
                                false, -1, true);
      }
      Changed |= LoopRotation(L, &LI, &TTI, &GetAC(*F), &DT, &SE,
                              /* TODO MSSAUpdater */ nullptr, &GetTI(*F), SQ,
                              false, -1, true);
      LoopWorkList.push_back(L);
    }

    for (Loop *L : LoopWorkList)
      Changed |= formLCSSARecursively(*L, DT, &LI, &SE);
  }

  // Now process each loop.
  for (Function *F : WorkList) {

    //    TapirTargetID TargetID = GetTLI(*F).getTapirTarget();
    //    std::unique_ptr<TapirTarget> Target(getTapirTargetFromID(M,
    //    TargetID));
    //
    Changed |= LoopNestReorder(*F, GetDT(*F), GetLI(*F), GetTI(*F), GetSE(*F),
                               GetAC(*F), GetTTI(*F), GetORE(*F))
                   .run();
  }

  if (Changed)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
