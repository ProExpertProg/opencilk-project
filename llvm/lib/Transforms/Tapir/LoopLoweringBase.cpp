//
// Created by Luka on 4/27/2023.
//

#include "llvm/Transforms/Tapir/LoopLoweringBase.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Tapir.h"
#include "llvm/Transforms/Tapir/LoopLoweringBase.h"
#include "llvm/Transforms/Tapir/LoopSpawningTI.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/TapirUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "tapirlooplowering"

static const char TimerGroupName[] = DEBUG_TYPE;
static const char TimerGroupDescription[] = "Tapir Loop Lowering Base";

namespace {
// Helper test to see if the given basic block is the placeholder normal
// destination of a detached.rethrow or taskframe.resume intrinsic.
bool isUnreachablePlaceholder(const BasicBlock *B) {
  for (const BasicBlock *Pred : predecessors(B)) {
    if (!isDetachedRethrow(Pred->getTerminator()) &&
        !isTaskFrameResume(Pred->getTerminator()))
      return false;
    if (B != cast<InvokeInst>(Pred->getTerminator())->getNormalDest())
      return false;
  }
  return true;
}
} // namespace

/// Get the set of basic blocks within the task of Tapir loop \p TL.  The \p
/// TaskBlocks vector stores all of these basic blocks.  The \p ReattachBlocks
/// set identifies which blocks are terminated by a reattach instruction that
/// terminates the task.  The \p DetachedRethrowBlocks set identifies which
/// blocks are terminated by detached-rethrow instructions that terminate the
/// task.  Entry points to shared exception-handling code is stored in the
/// \p SharedEHEntries set.
///
/// This method relies on being executed on the Tapir loops in a function in
/// post order.
void LoopLoweringBase::getTapirLoopTaskBlocks(
    TapirLoopInfo *TL, std::vector<BasicBlock *> &TaskBlocks,
    SmallPtrSetImpl<BasicBlock *> &ReattachBlocks,
    SmallPtrSetImpl<BasicBlock *> &DetachedRethrowBlocks,
    SmallPtrSetImpl<BasicBlock *> &SharedEHEntries,
    SmallPtrSetImpl<BasicBlock *> &UnreachableExits) {
  NamedRegionTimer NRT("getTapirLoopTaskBlocks",
                       "Get basic blocks for Tapir loop", TimerGroupName,
                       TimerGroupDescription, TimePassesIsEnabled);

  Task *T = TL->getTask();
  SmallVector<Task *, 4> EnclosedTasks;
  TL->getEnclosedTasks(EnclosedTasks);
  SmallPtrSet<Spindle *, 4> VisitedSharedEH;

  // Get the header and loop-latch blocks of all Tapir subloops.
  SmallPtrSet<BasicBlock *, 4> SubloopControlToExclude;
  for (Task *EncT : EnclosedTasks) {
    for (Task *SubT : EncT->subtasks()) {
      if (TapirLoopInfo *SubTL = getTapirLoop(SubT)) {
        SubloopControlToExclude.insert(SubTL->getLoop()->getHeader());
        SubloopControlToExclude.insert(SubTL->getLoop()->getLoopLatch());
        // Mark the unwind destination of this subloop's detach as a
        // "SharedEHEntry," meaning it needs its Phi nodes updated after
        // cloning.
        DetachInst *SubDI =
            cast<DetachInst>(SubTL->getLoop()->getHeader()->getTerminator());
        if (SubDI->hasUnwindDest())
          SharedEHEntries.insert(SubDI->getUnwindDest());
      }
    }
  }

  for (Task *EncT : EnclosedTasks) {
    for (Spindle *S : depth_first<InTask<Spindle *>>(EncT->getEntrySpindle())) {
      // Record the entry blocks of any shared-EH spindles.
      if (S->isSharedEH()) {
        SharedEHEntries.insert(S->getEntry());
        if (!VisitedSharedEH.insert(S).second)
          continue;
      }

      bool TopLevelTaskSpindle = T->contains(S) || T->isSharedEHExit(S);
      for (BasicBlock *B : S->blocks()) {
        // Don't clone header and loop-latch blocks for Tapir subloops.
        if (SubloopControlToExclude.count(B))
          continue;

        // Skip basic blocks that are successors of detached rethrows in T.
        // They're dead anyway.
        if (TopLevelTaskSpindle && isSuccessorOfDetachedRethrow(B))
          continue;

        // Skip unreachable placeholder blocks, namely, the normal destinations
        // of detached.rethrow and taskframe.resume instructions.
        if (isUnreachablePlaceholder(B))
          continue;

        LLVM_DEBUG(dbgs() << "Adding block " << B->getName() << "\n");
        TaskBlocks.push_back(B);

        if (TopLevelTaskSpindle) {
          // Record the blocks terminated by reattaches and detached rethrows.
          if (isa<ReattachInst>(B->getTerminator()))
            ReattachBlocks.insert(B);
          if (isDetachedRethrow(B->getTerminator()))
            DetachedRethrowBlocks.insert(B);
          if (isTaskFrameResume(B->getTerminator()))
            UnreachableExits.insert(
                cast<InvokeInst>(B->getTerminator())->getNormalDest());
        } else if (isDetachedRethrow(B->getTerminator()) ||
                   isTaskFrameResume(B->getTerminator())) {
          UnreachableExits.insert(
              cast<InvokeInst>(B->getTerminator())->getNormalDest());
        }
      }
    }
  }
}

namespace {
/// Compute the grainsize of the loop, based on the limit.  Currently this
/// routine injects a call to the tapir_loop_grainsize intrinsic, which is
/// handled in a target-specific way by subsequent lowering passes.
Value *computeGrainsize(TapirLoopInfo *TL) {
  Value *TripCount = TL->getTripCount();
  assert(TripCount &&
         "No trip count found for computing grainsize of Tapir loop.");
  Type *IdxTy = TripCount->getType();
  BasicBlock *Preheader = TL->getLoop()->getLoopPreheader();
  Module *M = Preheader->getModule();
  IRBuilder<> B(Preheader->getTerminator());
  B.SetCurrentDebugLocation(TL->getDebugLoc());
  return B.CreateCall(
      Intrinsic::getDeclaration(M, Intrinsic::tapir_loop_grainsize, {IdxTy}),
      {TripCount});
}

/// Get the grainsize of this loop either from metadata or by computing the
/// grainsize.
Value *getGrainsizeVal(TapirLoopInfo *TL) {
  Value *GrainVal;
  if (unsigned Grainsize = TL->getGrainsize())
    GrainVal = ConstantInt::get(TL->getTripCount()->getType(), Grainsize);
  else
    GrainVal = computeGrainsize(TL);

  LLVM_DEBUG(dbgs() << "Grainsize value: " << *GrainVal << "\n");
  return GrainVal;
}

/// Determine the inputs to Tapir loop \p TL for the loop control.
void getLoopControlInputs(TapirLoopInfo *TL, SmallVectorImpl<Value *> &LCArgs,
                          SmallVectorImpl<Value *> &LCInputs) {
  // Add an argument for the primary induction variable.
  auto &PrimaryInduction = TL->getPrimaryInduction();
  PHINode *PrimaryPhi = PrimaryInduction.first;
  TL->StartIterArg =
      new Argument(PrimaryPhi->getType(), PrimaryPhi->getName() + ".start");
  LCArgs.push_back(TL->StartIterArg);
  LCInputs.push_back(PrimaryInduction.second.getStartValue());

  // Add an argument for the trip count.
  Value *TripCount = TL->getTripCount();
  assert(TripCount && "No trip count found for Tapir loop end argument.");
  TL->EndIterArg = new Argument(TripCount->getType(), "end");
  LCArgs.push_back(TL->EndIterArg);
  LCInputs.push_back(TripCount);

  // Add an argument for the grainsize.
  Value *GrainsizeVal = getGrainsizeVal(TL);
  TL->GrainsizeArg = new Argument(GrainsizeVal->getType(), "grainsize");
  LCArgs.push_back(TL->GrainsizeArg);
  LCInputs.push_back(GrainsizeVal);

  assert(TL->getInductionVars()->size() == 1 &&
         "Induction vars to process for arguments.");
  // // Add arguments for the other IV's.
  // for (auto &InductionEntry : *TL->getInductionVars()) {
  //   PHINode *Phi = InductionEntry.first;
  //   InductionDescriptor II = InductionEntry.second;
  //   if (Phi == PrimaryInduction.first) continue;
  //   LCArgs.push_back(new Argument(Phi->getType(),
  //                                 Phi->getName() + ".start"));
  //   LCInputs.push_back(II.getStartValue());
  // }
}
} // namespace

/// For all recorded Tapir loops, determine the function arguments and inputs
/// for the outlined helper functions for those loops.
///
/// The \p LoopArgs map will store the function arguments for these outlined
/// loop helpers.  The \p LoopInputs map will store the corresponding arguments
/// for calling those outlined helpers from the parent function.  The \p
/// LoopArgStarts map will store the instruction in the parent where new code
/// for computing these outlined-helper-call arguments is first inserted.
void LoopLoweringBase::getAllTapirLoopInputs(
    DenseMap<Loop *, ValueSet> &LoopInputSets,
    DenseMap<Loop *, SmallVector<Value *, 3>> &LoopCtlArgs,
    DenseMap<Loop *, SmallVector<Value *, 3>> &LoopCtlInputs) {
  NamedRegionTimer NRT("getAllTapirLoopInputs",
                       "Determine inputs for all Tapir loops", TimerGroupName,
                       TimerGroupDescription, TimePassesIsEnabled);

  // Determine the inputs for all tasks.
  TaskValueSetMap TaskInputs = findAllTaskInputs(F, DT, TI);

  // Combine these sets of inputs to determine inputs for each Tapir loop.
  DenseMap<Loop *, ValueSet> TapirLoopInputs;
  for (Task *T : post_order(TI.getRootTask())) {
    if (TapirLoopInfo *TL = getTapirLoop(T)) {
      Loop *L = TL->getLoop();

      // Convert inputs for task T to Tapir-loop inputs.
      ValueSet TLInputs = getTapirLoopInputs(TL, TaskInputs[T]);
      LoopInputSets[L] = TLInputs;
      LLVM_DEBUG({
        dbgs() << "Loop: " << L->getName() << "\nTLInputs\n";
        for (Value *V : TLInputs)
          dbgs() << "\t" << *V << "\n";
      });

      // Determine loop-control inputs.
      getLoopControlInputs(TL, LoopCtlArgs[L], LoopCtlInputs[L]);

      LLVM_DEBUG({
        dbgs() << "LoopCtlArgs:\n";
        for (Value *V : LoopCtlArgs[L])
          dbgs() << "\t" << *V << "\n";
        dbgs() << "LoopCtlInputs:\n";
        for (Value *V : LoopCtlInputs[L])
          dbgs() << "\t" << *V << "\n";
      });
    }
  }
}

namespace {
void updateClonedIVs(TapirLoopInfo *TL, BasicBlock *OrigPreheader,
                     ValueSet &Args, ValueToValueMapTy &VMap,
                     unsigned IVArgIndex, unsigned NextIVArgOffset = 3) {
  NamedRegionTimer NRT("updateClonedIVs", "Updated IVs in Tapir-loop helper",
                       TimerGroupName, TimerGroupDescription,
                       TimePassesIsEnabled);

  auto &PrimaryInduction = TL->getPrimaryInduction();
  PHINode *PrimaryPhi = PrimaryInduction.first;

  Value *PrimaryArg = Args[IVArgIndex];

  // TODO: This assertion implies that the following loop should only run once,
  // for the primary induction variable.  However, the loop is provided in case
  // we decide to handle more complicated sets of induction variables in the
  // future.
  assert(TL->getInductionVars()->size() == 1 &&
         "updateClonedIVs to process multiple inductions.");

  // Get the next argument that provides an input to an IV, which is typically 3
  // after the input for the primary induction variable, after the end-teration
  // and grainsize arguments.
  unsigned ArgIdx = IVArgIndex + NextIVArgOffset;
  for (auto &InductionEntry : *TL->getInductionVars()) {
    PHINode *OrigPhi = InductionEntry.first;
    InductionDescriptor II = InductionEntry.second;
    assert(II.getKind() == InductionDescriptor::IK_IntInduction &&
           "Non-integer induction found.");
    assert((II.getConstIntStepValue()->isOne() ||
            II.getConstIntStepValue()->isMinusOne()) &&
           "Non-canonical induction found: non-unit step.");
    assert(isa<Constant>(II.getStartValue()) &&
           "Non-canonical induction found: non-constant start.");
    assert(cast<Constant>(II.getStartValue())->isNullValue() &&
           "Non-canonical induction found: non-zero start.");

    // Get the remapped PHI node and preheader
    PHINode *NewPhi = cast<PHINode>(VMap[OrigPhi]);
    BasicBlock *NewPreheader = cast<BasicBlock>(VMap[OrigPreheader]);

    // Replace the input for the remapped PHI node from the preheader with the
    // input argument.
    unsigned BBIdx = NewPhi->getBasicBlockIndex(NewPreheader);
    if (OrigPhi == PrimaryPhi)
      NewPhi->setIncomingValue(BBIdx, VMap[PrimaryArg]);
    else
      // TODO: Because of the assertion above, this line should never run.
      NewPhi->setIncomingValue(BBIdx, VMap[Args[ArgIdx++]]);
  }
}

// ValueMaterializer to manage remapping uses of the tripcount in the helper
// function for the loop, when the only uses of tripcount occur in the condition
// for the loop backedge and, possibly, in metadata.
class ArgEndMaterializer final : public OutlineMaterializer {
private:
  Value *TripCount;
  Value *ArgEnd;

public:
  ArgEndMaterializer(const Instruction *SrcSyncRegion, Value *TripCount,
                     Value *ArgEnd)
      : OutlineMaterializer(SrcSyncRegion), TripCount(TripCount),
        ArgEnd(ArgEnd) {}

  Value *materialize(Value *V) final {
    // If we're materializing metadata for TripCount, materialize empty metadata
    // instead.
    if (auto *MDV = dyn_cast<MetadataAsValue>(V)) {
      Metadata *MD = MDV->getMetadata();
      if (auto *LAM = dyn_cast<LocalAsMetadata>(MD))
        if (LAM->getValue() == TripCount)
          return MetadataAsValue::get(V->getContext(),
                                      MDTuple::get(V->getContext(), None));
    }

    // Materialize TripCount with ArgEnd.  This should only occur in the loop
    // latch, and we'll overwrite the use of ArgEnd later.
    if (V == TripCount)
      return ArgEnd;

    // Otherwise go with the default behavior.
    return OutlineMaterializer::materialize(V);
  }
};
} // namespace

/// Outline Tapir loop \p TL into a helper function.  The \p Args set
/// specified
/// the arguments to that helper function.  The map \p VMap will store the
/// mapping of values in the original function to values in the outlined helper.
Function *LoopLoweringBase::createHelperForTapirLoop(
    TapirLoopInfo *TL, ValueSet &Args, unsigned IVArgIndex,
    unsigned LimitArgIndex, Module *DestM, ValueToValueMapTy &VMap,
    ValueToValueMapTy &InputMap) {
  Task *T = TL->getTask();
  Loop *L = TL->getLoop();
  BasicBlock *Header = L->getHeader();
  BasicBlock *Preheader = L->getLoopPreheader();

  // Collect all basic blocks in the Tapir loop.
  std::vector<BasicBlock *> TLBlocks;
  TLBlocks.push_back(L->getHeader());
  // Entry blocks of shared-EH spindles may contain PHI nodes that need to be
  // rewritten in the cloned helper.
  SmallPtrSet<BasicBlock *, 4> SharedEHEntries;
  SmallPtrSet<BasicBlock *, 4> DetachedRethrowBlocks;
  SmallPtrSet<BasicBlock *, 4> UnreachableExits;
  // Reattach instructions and detached rethrows in this task might need special
  // handling.
  SmallPtrSet<BasicBlock *, 4> ReattachBlocks;
  getTapirLoopTaskBlocks(TL, TLBlocks, ReattachBlocks, DetachedRethrowBlocks,
                         SharedEHEntries, UnreachableExits);
  TLBlocks.push_back(L->getLoopLatch());

  DetachInst *DI = T->getDetach();
  const Instruction *InputSyncRegion =
      dyn_cast<Instruction>(DI->getSyncRegion());

  OutlineMaterializer *Mat = nullptr;
  if (!isa<Constant>(TL->getTripCount()) && !Args.count(TL->getTripCount()))
    // Create an ArgEndMaterializer to handle uses of TL->getTripCount().
    Mat = new ArgEndMaterializer(InputSyncRegion, TL->getTripCount(),
                                 Args[LimitArgIndex]);
  else
    Mat = new OutlineMaterializer(InputSyncRegion);

  Twine NameSuffix = ".ls" + Twine(TL->getLoop()->getLoopDepth());
  SmallVector<ReturnInst *, 4> Returns; // Ignore returns cloned.
  ValueSet Outputs;                     // Outputs must be empty.
  Function *Helper;
  {
    NamedRegionTimer NRT("CreateHelper", "Create helper for Tapir loop",
                         TimerGroupName, TimerGroupDescription,
                         TimePassesIsEnabled);
    Helper = CreateHelper(
        Args, Outputs, TLBlocks, Header, Preheader, TL->getExitBlock(), VMap,
        DestM, F.getSubprogram() != nullptr, Returns, NameSuffix.str(), nullptr,
        &DetachedRethrowBlocks, &SharedEHEntries, TL->getUnwindDest(),
        &UnreachableExits, nullptr, nullptr, nullptr, Mat);
  } // end timed region

  assert(Returns.empty() && "Returns cloned when cloning detached CFG.");
  // If the Tapir loop has no unwind destination, then the outlined function
  // cannot throw.
  if (F.doesNotThrow() && !TL->getUnwindDest())
    Helper->setDoesNotThrow();
  // Don't inherit the noreturn attribute from the caller.
  if (F.doesNotReturn())
    Helper->removeFnAttr(Attribute::NoReturn);

  // Update cloned loop condition to use the end-iteration argument.
  unsigned TripCountIdx = 0;
  Value *TripCount = TL->getTripCount();
  if (InputMap[TripCount])
    TripCount = InputMap[TripCount];
  if (TL->getCondition()->getOperand(0) != TripCount)
    ++TripCountIdx;
  assert(TL->getCondition()->getOperand(TripCountIdx) == TripCount &&
         "Trip count not used in condition");
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TL->getCondition()]);
  ClonedCond->setOperand(TripCountIdx, VMap[Args[LimitArgIndex]]);

  // If the trip count is variable and we're not passing the trip count as an
  // argument, undo the eariler temporarily mapping.
  if (!isa<Constant>(TL->getTripCount()) && !Args.count(TL->getTripCount())) {
    VMap.erase(TL->getTripCount());
    // Delete the ArgEndMaterializer.
    delete Mat;
  }

  // Rewrite cloned IV's to start at their start-iteration arguments.
  updateClonedIVs(TL, Preheader, Args, VMap, IVArgIndex);

  // Add alignment assumptions to arguments of helper, based on alignment of
  // values in old function.
  {
    NamedRegionTimer NRT("AddAlignmentAssumptions",
                         "Add alignment assumptions to Tapir-loop helper",
                         TimerGroupName, TimerGroupDescription,
                         TimePassesIsEnabled);
    AddAlignmentAssumptions(&F, Args, VMap, Preheader->getTerminator(), &AC,
                            &DT);
  } // end timed region

  // CreateHelper partially serializes the cloned copy of the loop by converting
  // detached-rethrows into resumes.  We now finish the job of serializing the
  // cloned Tapir loop.

  // Move allocas in the newly cloned detached CFG to the entry block of the
  // helper.
  {
    NamedRegionTimer NRT("updateAllocas", "Update allocas in Tapir-loop helper",
                         TimerGroupName, TimerGroupDescription,
                         TimePassesIsEnabled);
    // Collect the end instructions of the task.
    SmallVector<Instruction *, 4> TaskEnds;
    for (BasicBlock *EndBlock : ReattachBlocks)
      TaskEnds.push_back(cast<BasicBlock>(VMap[EndBlock])->getTerminator());
    for (BasicBlock *EndBlock : DetachedRethrowBlocks)
      TaskEnds.push_back(cast<BasicBlock>(VMap[EndBlock])->getTerminator());

    // Move allocas in cloned detached block to entry of helper function.
    BasicBlock *ClonedTaskEntry = cast<BasicBlock>(VMap[T->getEntry()]);
    bool ContainsDynamicAllocas = MoveStaticAllocasInBlock(
        &Helper->getEntryBlock(), ClonedTaskEntry, TaskEnds);

    // If this task uses a taskframe, move allocas in cloned taskframe entry to
    // entry of helper function.
    if (Spindle *TFCreate = T->getTaskFrameCreateSpindle()) {
      BasicBlock *ClonedTFEntry = cast<BasicBlock>(VMap[TFCreate->getEntry()]);
      ContainsDynamicAllocas |= MoveStaticAllocasInBlock(
          &Helper->getEntryBlock(), ClonedTFEntry, TaskEnds);
    }
    // If the cloned loop contained dynamic alloca instructions, wrap the cloned
    // loop with llvm.stacksave/llvm.stackrestore intrinsics.
    if (ContainsDynamicAllocas) {
      Module *M = Helper->getParent();
      // Get the two intrinsics we care about.
      Function *StackSave = Intrinsic::getDeclaration(M, Intrinsic::stacksave);
      Function *StackRestore =
          Intrinsic::getDeclaration(M, Intrinsic::stackrestore);

      // Insert the llvm.stacksave.
      CallInst *SavedPtr =
          IRBuilder<>(&*ClonedTaskEntry, ClonedTaskEntry->begin())
              .CreateCall(StackSave, {}, "savedstack");

      // Insert a call to llvm.stackrestore before the reattaches in the
      // original Tapir loop.
      for (Instruction *ExitPoint : TaskEnds)
        IRBuilder<>(ExitPoint).CreateCall(StackRestore, SavedPtr);
    }
  }

  // Convert the cloned detach and reattaches into unconditional branches.
  {
    NamedRegionTimer NRT("serializeClonedLoop", "Serialize cloned Tapir loop",
                         TimerGroupName, TimerGroupDescription,
                         TimePassesIsEnabled);
    DetachInst *ClonedDI = cast<DetachInst>(VMap[DI]);
    BasicBlock *ClonedDetacher = ClonedDI->getParent();
    BasicBlock *ClonedContinue = ClonedDI->getContinue();
    for (BasicBlock *RB : ReattachBlocks) {
      ReattachInst *ClonedRI = cast<ReattachInst>(VMap[RB->getTerminator()]);
      ReplaceInstWithInst(ClonedRI, BranchInst::Create(ClonedContinue));
    }
    ClonedContinue->removePredecessor(ClonedDetacher);
    BranchInst *DetachRepl = BranchInst::Create(ClonedDI->getDetached());
    ReplaceInstWithInst(ClonedDI, DetachRepl);
    VMap[DI] = DetachRepl;
  } // end timed region

  return Helper;
}