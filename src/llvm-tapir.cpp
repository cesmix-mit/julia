// This file is a part of Julia. License is MIT: https://julialang.org/license

#include "llvm-version.h"

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Value.h>
#ifdef JL_DEBUG_BUILD
#include <llvm/IR/Verifier.h>
#endif
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <llvm/Support/Debug.h>

// clang-format off
#include "codegen_shared.h"
#include "julia.h"
#include "julia_internal.h"
#include "jitlayers.h"
#include "llvm-pass-helpers.h"
// clang-format on

// `llvm/ExecutionEngine/Orc/Layer.h` included via `jitlayers.h` sets
// `DEBUG_TYPE`. So, undef-ing it first:
#undef DEBUG_TYPE
#define DEBUG_TYPE "julia_tapir"

/**
 * JuliaTapir lowers Tapir constructs through outlining to Julia Task's.
 * After lowering the code should be equivalent to the Julia code below:
 *
 * ```julia
 *   llvmf = ... # outlined function
 *   tasklist = Task[]
 *   t = Task(llvmf)
 *   push!(tasklist, t)
 *   schedule(t)
 *   sync_end(tasklist)
 * ```
 **/

namespace llvm {

// Extract out the index value used by GEP created in LLVM/Tapir's
// `createTaskArgsStruct`:
uint64_t structFieldIndex(GetElementPtrInst *GEP)
{
    auto LastIdx = cast<ConstantInt>(*(GEP->op_end() - 1));
    return LastIdx->getValue().getLimitedValue();
}

static PointerType *needAddrSpaceCast(Type *T)
{
    auto PT = dyn_cast<PointerType>(T);
    if (PT && (PT->getAddressSpace() == AddressSpace::CalleeRooted ||
               PT->getAddressSpace() == AddressSpace::Derived)) {
        return PT;
    }
    else {
        return nullptr;
    }
}

static PointerType *castTypeForGC(PointerType *T)
{
    return PointerType::get(T->getElementType(), AddressSpace::Tracked);
}

class JuliaTapir : public TapirTarget, private JuliaPassContext {
    ValueToValueMapTy DetachBlockToTaskGroup;
    ValueToValueMapTy SyncRegionToTaskGroup;
    Type *SpawnFTy = nullptr;

    // Opaque Julia runtime functions
    FunctionCallee JlTapirTaskGroup;
    FunctionCallee JlTapirSpawn;
    FunctionCallee JlTapirSync;

    // Accessors for opaque Julia runtime functions
    FunctionCallee get_jl_tapir_taskgroup();
    FunctionCallee get_jl_tapir_spawn();
    FunctionCallee get_jl_tapir_sync();

    void markDecayedPointerInArgStruct(TaskOutlineInfo &);
    void replaceDecayedPointerInOutline(TaskOutlineInfo &);
    std::tuple<AllocaInst *, size_t> insertJuliaValueSources(TaskOutlineInfo &);
    void collectJuliaValues(SmallVectorImpl<Value *> &, IRBuilder<> &, Value *);

    DenseMap<Type *, Function *> PtrTypeToLoadNonNull;
    Function *getLoadNonNullFunction(Type *);

    void insertGCPreserve(Function &);
    void insertPtls(Function &);

    Type *T_uint64;
    Type *T_uint32;
    Type *T_size;
    Type *T_psize;
    Value *getWorldAge(Instruction *);

public:
    JuliaTapir(Module &M);
    ~JuliaTapir() {}

    ArgStructMode getArgStructMode() const override final
    {
        return ArgStructMode::Static;
        // return ArgStructMode::Dynamic;
    }

    Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
    void lowerSync(SyncInst &inst) override final;

    void preProcessFunction(Function &F, TaskInfo &TI,
                            bool OutliningTapirLoops) override final;
    void postProcessFunction(Function &F, bool OutliningTapirLoops) override final;
    void postProcessHelper(Function &F) override final;

    void preProcessOutlinedTask(Function &F, Instruction *DetachPt,
                                Instruction *TaskFrameCreate,
                                bool IsSpawner) override final;
    void postProcessOutlinedTask(Function &F, Instruction *DetachPt,
                                 Instruction *TaskFrameCreate,
                                 bool IsSpawner) override final;
    void preProcessRootSpawner(Function &F) override final;
    void postProcessRootSpawner(Function &F) override final;
    void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) override final;
};

JuliaTapir::JuliaTapir(Module &M) : TapirTarget(M)
{
    LLVMContext &C = M.getContext();
    // Initialize any types we need for lowering.
    SpawnFTy = PointerType::getUnqual(
        FunctionType::get(Type::getVoidTy(C), {Type::getInt8PtrTy(C)}, false));

    initAll(M);

    // Ideally, we can rely on `initAll(M)` to set up all runtime functions.
    // However, it does not set up the functions; i.e., it relies on that they
    // are created in `emit_function`. Since we are adding *new* GC roots, we
    // need to manually make sure that these functions are set up, even though
    // these functions are not inserted in the emission phase. The function
    // signatures have to be matched with the ones in `codegen.cpp` (see
    // `gc_preserve_begin_func = new JuliaFunction{...}` etc.).
    gc_preserve_begin_func =
        cast<Function>(M.getOrInsertFunction("llvm.julia.gc_preserve_begin",
                                             FunctionType::get(Type::getTokenTy(C), true))
                           .getCallee());
    gc_preserve_end_func = cast<Function>(
        M.getOrInsertFunction("llvm.julia.gc_preserve_end",
                              FunctionType::get(Type::getVoidTy(C), {Type::getTokenTy(C)},
                                                false))
            .getCallee());

    // See `init_julia_llvm_env`
    T_uint64 = Type::getInt64Ty(C);
    T_uint32 = Type::getInt32Ty(C);
    if (sizeof(size_t) == 8)
        T_size = T_uint64;
    else
        T_size = T_uint32;
    T_psize = PointerType::get(T_size, 0);
}

FunctionCallee JuliaTapir::get_jl_tapir_taskgroup()
{
    if (JlTapirTaskGroup)
        return JlTapirTaskGroup;

    AttributeList AL;
    FunctionType *FTy = FunctionType::get(T_prjlvalue, {}, false);

    JlTapirTaskGroup = M.getOrInsertFunction("jl_tapir_taskgroup", FTy, AL);
    return JlTapirTaskGroup;
}

FunctionCallee JuliaTapir::get_jl_tapir_spawn()
{
    if (JlTapirSpawn)
        return JlTapirSpawn;

    LLVMContext &C = M.getContext();
    const DataLayout &DL = M.getDataLayout();
    AttributeList AL;
    FunctionType *FTy = FunctionType::get(Type::getVoidTy(C),
                                          {
                                              T_prjlvalue, // jl_value_t *tasks
                                              SpawnFTy, // void *f
                                              Type::getInt8PtrTy(C), // void *arg
                                              DL.getIntPtrType(C), // size_t arg_size
                                              T_ppjlvalue, // jl_value_t **jvs
                                              DL.getIntPtrType(C), // size_t jvs_size
                                              T_size, // size_t world_age
                                          },
                                          false);

    JlTapirSpawn = M.getOrInsertFunction("jl_tapir_spawn", FTy, AL);
    return JlTapirSpawn;
}

FunctionCallee JuliaTapir::get_jl_tapir_sync()
{
    if (JlTapirSync)
        return JlTapirSync;

    LLVMContext &C = M.getContext();
    AttributeList AL;
    FunctionType *FTy = FunctionType::get(Type::getVoidTy(C), {T_prjlvalue}, false);

    JlTapirSync = M.getOrInsertFunction("jl_tapir_sync", FTy, AL);
    return JlTapirSync;
}

Value *JuliaTapir::lowerGrainsizeCall(CallInst *GrainsizeCall)
{
    Value *Limit = GrainsizeCall->getArgOperand(0);
    Module *M = GrainsizeCall->getModule();
    IRBuilder<> Builder(GrainsizeCall);

    // get jl_n_threads (extern global variable)
    Constant *proto =
        M->getOrInsertGlobal("jl_n_threads", Type::getInt32Ty(M->getContext()));

    Value *Workers = Builder.CreateLoad(proto);

    // Choose 8xWorkers as grainsize
    Value *WorkersX8 = Builder.CreateIntCast(
        Builder.CreateMul(Workers, ConstantInt::get(Workers->getType(), 8)),
        Limit->getType(), false);

    // Compute ceil(limit / 8 * workers) =
    //           (limit + 8 * workers - 1) / (8 * workers)
    Value *SmallLoopVal =
        Builder.CreateUDiv(Builder.CreateSub(Builder.CreateAdd(Limit, WorkersX8),
                                             ConstantInt::get(Limit->getType(), 1)),
                           WorkersX8);
    // Compute min
    Value *LargeLoopVal = ConstantInt::get(Limit->getType(), 2048);
    Value *Cmp = Builder.CreateICmpULT(LargeLoopVal, SmallLoopVal);
    Value *Grainsize = Builder.CreateSelect(Cmp, LargeLoopVal, SmallLoopVal);

    // Replace uses of grainsize intrinsic call with this grainsize value.
    GrainsizeCall->replaceAllUsesWith(Grainsize);
    return Grainsize;
}

void JuliaTapir::lowerSync(SyncInst &SI)
{
    IRBuilder<> builder(&SI);
    Value *SR = SI.getSyncRegion();
    auto TG = SyncRegionToTaskGroup[SR];
    builder.CreateCall(get_jl_tapir_sync(), {TG});
    BranchInst *PostSync = BranchInst::Create(SI.getSuccessor(0));
    ReplaceInstWithInst(&SI, PostSync);
}

void JuliaTapir::preProcessFunction(Function &F, TaskInfo &TI, bool OutliningTapirLoops)
{
#ifdef JL_DEBUG_BUILD
    assert(!verifyFunction(F, &dbgs()));
#endif
    if (OutliningTapirLoops) // TODO: figure out if we need to do something
        return;

    for (Task *T : post_order(TI.getRootTask())) {
        if (T->isRootTask())
            continue;
        DetachInst *Detach = T->getDetach();
        BasicBlock *detB = Detach->getParent();
        auto SR = cast<Instruction>(Detach->getSyncRegion());

        // Sync regions and task groups are one-to-one. However, since multiple
        // detach instructions can be invoked in a single sync region, we check
        // if a corresponding task group is created.
        Value *TG = SyncRegionToTaskGroup[SR];
        if (!TG) {
            TG = CallInst::Create(get_jl_tapir_taskgroup(), {}, "", SR);
            SyncRegionToTaskGroup[SR] = TG;
        }
        // TODO: don't look up the map twice
        if (!DetachBlockToTaskGroup[detB]) {
            DetachBlockToTaskGroup[detB] = TG;
        }
    }
#ifdef JL_DEBUG_BUILD
    assert(!verifyFunction(F, &dbgs()));
#endif
}

void JuliaTapir::postProcessFunction(Function &F, bool OutliningTapirLoops)
{
    // nothing
}

// Since `LateLowerGCFrame::LocalScan` expect the output argument to use alloca,
// we replace struct captured in the outlined task argument with alloca here.
void JuliaTapir::postProcessHelper(Function &F)
{
    // TODO: Check if we need to collect `Calls` for iterator invalidation.
    SmallVector<CallInst *, 8> Calls;
    for (BasicBlock &BB : F)
        for (Instruction &I : BB)
            if (auto CI = dyn_cast<CallInst>(&I)) {
                // See `LateLowerGCFrame::LocalScan`
                Value *Arg = CI->arg_begin()[0];
                if (CI->hasStructRetAttr())
                    if (isa<PointerType>(Arg->getType()) &&
                        !isa<AllocaInst>(Arg->stripInBoundsOffsets()))
                        Calls.push_back(CI);
            }

    const DataLayout &DL = M.getDataLayout();
    for (auto CI : Calls) {
        auto AfterCall = cast<Instruction>(++CI->getIterator());
        for (size_t i = 0; i < std::min(2u, CI->getNumArgOperands()); i++) {
            Value *Arg = CI->arg_begin()[i];
            auto T = dyn_cast<PointerType>(Arg->getType());
            if (!T)
                continue;
            auto SRet =
                new AllocaInst(T->getElementType(), DL.getAllocaAddrSpace(), "jltapir.sret",
                               &*F.getEntryBlock().getFirstInsertionPt());
            if (T == SRet->getType()) {
                CI->setArgOperand(i, SRet);
            }
            else {
                auto Cast = CastInst::Create(Instruction::AddrSpaceCast, SRet, T,
                                             "jltapir.sret", CI);
                CI->setArgOperand(i, Cast);
            }
            auto V = new LoadInst(SRet, "jltapir.sret", AfterCall);
            new StoreInst(V, Arg, AfterCall);
        }
    }
#ifdef JL_DEBUG_BUILD
    assert(!verifyFunction(F, &dbgs()));
#endif
}

void JuliaTapir::preProcessOutlinedTask(Function &F, Instruction *DetachPt,
                                        Instruction *TaskFrameCreate, bool IsSpawner)
{
    insertPtls(F);
}

void JuliaTapir::insertGCPreserve(Function &F)
{
#ifdef JL_DEBUG_BUILD
    assert(!verifyFunction(F, &dbgs()));
#endif
    auto DT = DominatorTree(F);
    for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
            if (CallInst *TG = dyn_cast<CallInst>(&I)) {
                if (TG->getCalledFunction() == get_jl_tapir_taskgroup().getCallee()) {
                    // "Put `TG` in `GC.@preserve`":
                    auto Next = cast<Instruction>(++TG->getIterator());
                    Value *gctoken =
                        CallInst::Create(gc_preserve_begin_func, {TG}, "", Next);
                    // Make sure we "close `GC.@preserve`":
                    SmallVector<BasicBlock *, 4> Blocks;
                    DT.getDescendants(&BB, Blocks);
                    for (auto BB2 : Blocks) {
                        if (isa<ReturnInst>(BB2->getTerminator())) {
                            CallInst::Create(gc_preserve_end_func, {gctoken}, "",
                                             BB2->getTerminator());
                        }
                    }
                }
            }
        }
    }
#ifdef JL_DEBUG_BUILD
    assert(!verifyFunction(F, &dbgs()));
#endif
}

void JuliaTapir::insertPtls(Function &F)
{
#ifdef JL_DEBUG_BUILD
    assert(!verifyFunction(F, &dbgs()));
#endif
    if (getPtls(F))
        return;
    // Do what `allocate_gc_frame` (`codegen.cpp`) does:
    CallInst::Create(ptls_getter, {}, "", F.getEntryBlock().getFirstNonPHI());
    assert(getPtls(F));
    // TODO: other things in `allocate_gc_frame`
#ifdef JL_DEBUG_BUILD
    assert(!verifyFunction(F, &dbgs()));
#endif
}

// See `emit_last_age_field`
Value *JuliaTapir::getWorldAge(Instruction *IP)
{
    auto F = IP->getParent()->getParent();
    auto ptls = getPtls(*F);
    IRBuilder<> builder(IP);
    auto GEP = builder.CreateInBoundsGEP(
        T_size, builder.CreateBitCast(ptls, T_psize),
        ConstantInt::get(T_size, offsetof(jl_tls_states_t, world_age) / sizeof(size_t)));
    return builder.CreateAlignedLoad(GEP, Align(sizeof(size_t)), "age");
}

void JuliaTapir::postProcessOutlinedTask(Function &F, Instruction *DetachPt,
                                         Instruction *TaskFrameCreate, bool IsSpawner)
{
    insertGCPreserve(F);
}

void JuliaTapir::preProcessRootSpawner(Function &F)
{
    insertPtls(F);
}

void JuliaTapir::postProcessRootSpawner(Function &F)
{
    insertGCPreserve(F);
}

static bool isJuliaValue(Type *T)
{
    auto PT = dyn_cast<PointerType>(T);
    if (!PT)
        return false;
    auto AS = PT->getAddressSpace();
    return AS == AddressSpace::Tracked;
    // TODO: check if this is enough
}

// Null-out pointer values in aggregates.
// Note that this has to handle nested aggregates like `[2 x [2 x {} addrspace(10)*]]`.
static void initializeJuliaValueWithNull(Instruction *I)
{
    auto E = cast<PointerType>(I->getType())->getElementType();
    auto Next = cast<Instruction>(++I->getIterator());
    IRBuilder<> Builder(Next);
    if (auto T = dyn_cast<PointerType>(E))
        // We can also check `hasJuliaValue(E)` here instead to minimize
        // stores (but a bit of additional stores are probably harmless?).
        Builder.CreateStore(ConstantPointerNull::get(T), I);
    else if (auto T = dyn_cast<SequentialType>(E)) {
        for (size_t i = 0; i < T->getNumElements(); i++) {
            auto Name = I->getName() + ".jlnullout";
            auto V = Builder.CreateConstGEP2_32(T, I, 0, i, Name);
            initializeJuliaValueWithNull(cast<Instruction>(V));
        }
    }
    else if (auto T = dyn_cast<StructType>(E)) {
        for (size_t i = 0; i < T->getNumElements(); i++) {
            auto Name = I->getName() + ".jlnullout";
            auto V = Builder.CreateConstGEP2_32(T, I, 0, i, Name);
            initializeJuliaValueWithNull(cast<Instruction>(V));
        }
    }
}

// Create a function of type `**T -> *T` that propagates null.
Function *JuliaTapir::getLoadNonNullFunction(Type *T)
{
    auto PTy = cast<PointerType>(T);
    auto ETy = cast<PointerType>(PTy->getElementType());
    if (auto F = PtrTypeToLoadNonNull[T])
        return F;
    LLVMContext &C = M.getContext();
    const DataLayout &DL = M.getDataLayout();
    auto IntTy = DL.getIntPtrType(PTy);
    auto FTy = FunctionType::get(ETy, {PTy}, false);
    auto F = Function::Create(FTy, GlobalValue::InternalLinkage,
                              DL.getProgramAddressSpace(), "julia_tapir_load_nonnull", &M);
    auto Arg = F->args().begin();
    auto CheckBlock = BasicBlock::Create(C, "check", F);
    auto IfTrue = BasicBlock::Create(C, "if.true", F);
    auto IfFalse = BasicBlock::Create(C, "if.false", F);
    auto Int = CastInst::Create(Instruction::PtrToInt, Arg, IntTy, "", CheckBlock);
    auto NullP = CastInst::Create(Instruction::PtrToInt, ConstantPointerNull::get(PTy),
                                  IntTy, "", CheckBlock);
    auto IsNull =
        CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_EQ, Int, NullP, "", CheckBlock);
    BranchInst::Create(IfTrue, IfFalse, IsNull, CheckBlock);
    ReturnInst::Create(C, ConstantPointerNull::get(ETy), IfTrue);
    auto E = new LoadInst(Arg, "", IfFalse);
    ReturnInst::Create(C, E, IfFalse);
    F->addFnAttr(Attribute::AlwaysInline);
#ifdef JL_DEBUG_BUILD
    assert(!verifyFunction(*F, &dbgs()));
#endif
    return F;
}

// Collect Julia objects accesible through `J` into `JuliaValues`. Create load
// instructions with `Builder` if required. Single SSA value may result in
// multiple Julia values if it points to a struct.
void JuliaTapir::collectJuliaValues(SmallVectorImpl<Value *> &JuliaValues,
                                    IRBuilder<> &Builder, Value *J)
{
    auto T = cast<PointerType>(J->getType());
    auto E = T->getElementType();
    if (isJuliaValue(T)) {
        JuliaValues.push_back(J);
    }
    else if (isa<PointerType>(E)) {
        auto Name = J->getName() + ".jlv";
        collectJuliaValues(JuliaValues, Builder,
                           Builder.CreateCall(getLoadNonNullFunction(T), J, Name));
    }
    else if (auto S = dyn_cast<SequentialType>(E)) {
        // TODO: handle scalable vector? (is it used in Julia?)
        for (size_t i = 0; i < S->getNumElements(); i++) {
            auto GEP = Builder.CreateConstGEP2_32(S, J, 0, i);
            collectJuliaValues(JuliaValues, Builder, GEP);
        }
    }
    else if (auto S = dyn_cast<StructType>(E)) {
        for (size_t i = 0; i < S->getNumElements(); i++) {
            auto GEP = Builder.CreateConstGEP2_32(S, J, 0, i);
            collectJuliaValues(JuliaValues, Builder, GEP);
        }
    }
}
// TODO: combine `hasJuliaValue` and `collectJuliaValues`

// Check if the type contain pointer to a tracked Julia object.
static bool hasJuliaValue(Type *AnyT, DenseSet<Type *> &Seen)
{
    if (isJuliaValue(AnyT))
        return true;
    if (!Seen.insert(AnyT).second)
        return false;
    if (auto T = dyn_cast<PointerType>(AnyT)) {
        return hasJuliaValue(T->getElementType(), Seen);
    }
    else if (auto T = dyn_cast<ArrayType>(AnyT)) {
        return hasJuliaValue(T->getElementType(), Seen);
    }
    else if (auto T = dyn_cast<StructType>(AnyT)) {
        for (auto E : T->elements())
            if (hasJuliaValue(E, Seen))
                return true;
    }
    return false;
}

static bool hasJuliaValue(AllocaInst *I)
{
    DenseSet<Type *> Seen;
    return hasJuliaValue(I->getAllocatedType(), Seen);
}

static Value *getJuliaValueSourceImpl(Value *To, DenseMap<Value *, Value *> &Seen)
{
    if (auto From = Seen[To])
        return From;
    auto PT = cast<PointerType>(To->getType());
    Value *From = nullptr;
    auto AS = PT->getAddressSpace();
    if (isJuliaValue(PT)) {
        From = To;
    }
    else {
        if (auto I = dyn_cast<CastInst>(To)) {
            return getJuliaValueSourceImpl(I->getOperand(0), Seen);
        }
        else if (auto I = dyn_cast<GetElementPtrInst>(To)) {
            return getJuliaValueSourceImpl(I->getPointerOperand(), Seen);
        }
        else if (auto I = dyn_cast<PHINode>(To)) {
            auto Y = PHINode::Create(castTypeForGC(PT), I->getNumIncomingValues(),
                                     I->getName() + ".src", I);
            bool hasSource = false;
            for (auto BB : I->blocks()) {
                auto X = I->getIncomingValueForBlock(BB);
                auto S = getJuliaValueSourceImpl(X, Seen);
                if (S) {
                    Y->addIncoming(S, BB);
                    hasSource = true;
                }
                else {
                    Y->addIncoming(ConstantPointerNull::get(castTypeForGC(PT)), BB);
                }
            }
            if (hasSource) {
                From = Y;
            }
            else {
                Y->eraseFromParent();
                From = nullptr;
            }
        }
        else if (auto I = dyn_cast<AllocaInst>(To)) {
            if (hasJuliaValue(I)) {
                From = To;
                initializeJuliaValueWithNull(I);
            }
        }
        else if (AddressSpace::FirstSpecial <= AS && AS <= AddressSpace::LastSpecial) {
            // TODO: Should be an assert?
            LLVM_DEBUG({
                dbgs() << "T2T: Special address space " << AS << " not handled for:\n"
                       << "T2T: ";
                llvm_dump(To);
            });
        }
        else {
            LLVM_DEBUG({
                dbgs() << "T2T: Not following pointer: ";
                llvm_dump(To);
            });
        }
        // TODO: What else?
    }
    Seen[To] = From;
    return From;
}

static Value *getJuliaValueSource(Value *To)
{
    auto PT = dyn_cast<PointerType>(To->getType());
    if (!PT)
        return nullptr;
    DenseMap<Value *, Value *> Seen;
    return getJuliaValueSourceImpl(To, Seen);
}

// Tapir's outliner may capture the decayed pointers in the argument struct. The
// Julia objects from which these decayed pointers are derived are discovered
// and rooted by `insertJuliaValueSources`. However, `GCInvariantVerifier` would
// still raise some concerns due to that we store decayed pointers. So, we
// attach a metadata to the stores of them and ask GC to ignore them.
// See: `GCInvariantVerifier::visitStoreInst`.
void JuliaTapir::markDecayedPointerInArgStruct(TaskOutlineInfo &TOI)
{
    LLVMContext &C = M.getContext();
    CallBase *ReplCall = cast<CallBase>(TOI.ReplCall);
    AllocaInst *CallerArgStruct = cast<AllocaInst>(ReplCall->getArgOperand(0));
    BasicBlock *CallBlock = TOI.ReplStart->getParent();
    for (Instruction &V : *CallBlock) {
        auto Store = dyn_cast<StoreInst>(&V);
        if (!Store)
            continue;
        auto GEP = dyn_cast<GetElementPtrInst>(Store->getOperand(1));
        if (!(GEP && GEP->getPointerOperand() == CallerArgStruct))
            continue;
        auto T_int1 = Type::getInt1Ty(C);
        auto MD = MDNode::get(C, ConstantAsMetadata::get(ConstantInt::get(T_int1, true)));
        V.setMetadata("julia.tapir.store", MD);
    }
}

// Replace the address space of fields of the argument struct.
void JuliaTapir::replaceDecayedPointerInOutline(TaskOutlineInfo &TOI)
{
    LLVMContext &C = M.getContext();
    Function *F = TOI.Outline;
    FunctionType *FTy = F->getFunctionType();
    assert(FTy->getNumParams() == 1);
    auto STy = cast<StructType>(cast<PointerType>(FTy->getParamType(0))->getElementType());

    bool need_change = false;
    SmallVector<Type *, 8> FieldTypes;
    auto NumFields = STy->getStructNumElements();
    FieldTypes.resize(NumFields);
    for (size_t i = 0; i < NumFields; i++) {
        auto T0 = STy->getStructElementType(i);
        if (auto PT0 = needAddrSpaceCast(T0)) {
            FieldTypes[i] = castTypeForGC(PT0);
            need_change = true;
        }
        else {
            FieldTypes[i] = T0;
        }
    }
    StructType *NSTy;
    if (STy->isLiteral())
        NSTy = StructType::get(C, FieldTypes, STy->isPacked());
    else
        NSTy = StructType::create(C, FieldTypes, STy->getName(), STy->isPacked());
    PointerType *NPTy = PointerType::getUnqual(NSTy);
    FunctionType *NFTy = FunctionType::get(FTy->getReturnType(), {NPTy}, FTy->isVarArg());

    // TODO: enable this
    // if (!need_change)
    //     return;
    (void)need_change; // avoid unused variable warning until enabling early return

    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    F->getAllMetadata(MDs);

    Function *NF = Function::Create(NFTy, F->getLinkage(), F->getAddressSpace(),
                                    // F->getName() + ".jltapir",
                                    F->getName(), // maybe add suffix?
                                    F->getParent());

    // See `CloneFunction` for how `CloneFunctionInto` is used.
    ValueToValueMapTy VMap;
    Argument *NArg = NF->args().begin();
    Argument *Arg = F->args().begin();
    NArg->setName(Arg->getName());
    VMap[Arg] = NArg;
    SmallVector<ReturnInst *, 8> Returns; // Ignore returns cloned.
    bool ModuleLevelChanges = F->getSubprogram() != nullptr;
    CloneFunctionInto(NF, F, VMap, ModuleLevelChanges, Returns, "", nullptr);
#ifdef JL_DEBUG_BUILD
    assert(!verifyFunction(*NF, &dbgs()));
#endif

    SmallVector<GetElementPtrInst *, 8> GEPs;
    for (Value *I : NArg->users()) {
        auto GEP = dyn_cast<GetElementPtrInst>(I);
        if (GEP)
            GEPs.push_back(GEP);
        else {
            // TODO: Should be an assert?
            LLVM_DEBUG({
                dbgs() << "T2T: Ignoring non-GEP use of argument: ";
                llvm_dump(I);
            });
        }
    }
    for (auto GEP : GEPs) {
        // Extract IdxList from GEP:
        SmallVector<Value *, 8> IdxList;
        IdxList.reserve(GEP->getNumIndices());
        for (size_t i = 0; i < GEP->getNumIndices(); i++) {
            IdxList.push_back(GEP->getOperand(i + 1));
        }

        auto NGEP = GetElementPtrInst::Create(NSTy, GEP->getPointerOperand(), IdxList,
                                              GEP->getName() + "redecay.gep", GEP);
        NGEP->copyMetadata(*GEP);
        assert(NSTy->getTypeAtIndex(structFieldIndex(GEP)) == NGEP->getResultElementType());
        assert(!needAddrSpaceCast(NGEP->getResultElementType()));
        if (GEP->getType() == NGEP->getType()) {
            GEP->replaceAllUsesWith(NGEP);
            GEP->eraseFromParent();
            continue;
        }

        // Load as generic and then decay to the actually used address space.
        // First creating a copy `GEPUsers` as we are going to mutate the instructions.
        auto GEPUsers = SmallVector<Value *, 8>(GEP->users());
        for (Value *I : GEPUsers) {
            auto LI = dyn_cast<LoadInst>(I);
            if (LI) {
                // if (LI->getPointerOperandType() == NGEP->getResultElementType())
                //     continue;
                auto NLI = new LoadInst(NGEP->getResultElementType(), NGEP, "redecay.tmp",
                                        LI->isVolatile(), LI->getAlignment(),
                                        LI->getOrdering(), LI->getSyncScopeID(), LI);
                auto Decay = CastInst::Create(Instruction::AddrSpaceCast, NLI,
                                              LI->getType(), "redecay", LI);
                NLI->copyMetadata(*LI);
                LI->replaceAllUsesWith(Decay);
                LI->eraseFromParent();
                assert(!needAddrSpaceCast(NLI->getType()));
            }
            else {
                LLVM_DEBUG({
                    dbgs() << "T2T: Ignoring non-load use of GEP: ";
                    llvm_dump(I);
                });
            }
        }

        // TODO: Should we remove the original GEPs? How to make sure it's safe?
        // GEP->eraseFromParent();
        // ^-- It might be better to remove the original GEPs. However, since we
        // are not making sure all the usages are covered, this is not a safe
        // thing to do.
    }
#ifdef JL_DEBUG_BUILD
    for (BasicBlock &BB : *NF) {
        for (Instruction &I : BB) {
            auto Load = dyn_cast<LoadInst>(&I);
            if (!Load)
                continue;
            assert(!needAddrSpaceCast(Load->getType()));
        }
    }
#endif

    TOI.Outline = NF;
#ifdef JL_DEBUG_BUILD
    assert(!verifyFunction(*NF, &dbgs()));
#endif
}

// Collect Julia values that we need to capture and create an array of it.
std::tuple<AllocaInst *, size_t> JuliaTapir::insertJuliaValueSources(TaskOutlineInfo &TOI)
{
    LLVMContext &C = M.getContext();
    auto InsertPoint = cast<CallBase>(TOI.ReplCall);
    IRBuilder<> Builder(InsertPoint);
    SmallVector<Value *, 8> JuliaValues;
    SmallVector<Type *, 8> FieldTypes;
    for (auto V : TOI.InputSet) {
        auto J = getJuliaValueSource(V);
        if (!J)
            continue;
        collectJuliaValues(JuliaValues, Builder, J);
    }
    for (auto J : JuliaValues)
        FieldTypes.push_back(J->getType());
    auto *JVsTy = StructType::create(C, FieldTypes);
    auto JVs = Builder.CreateAlloca(JVsTy);

    for (size_t i = 0; i < JuliaValues.size(); i++) {
        auto V = JuliaValues[i];
        Builder.CreateStore(V, Builder.CreateConstGEP2_32(JVsTy, JVs, 0, i));
    }

    return {JVs, JuliaValues.size()};
}

// Based on QthreadsABI
void JuliaTapir::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT)
{
    Function *OldOutline = TOI.Outline; // to be deleted
#ifdef JL_DEBUG_BUILD
    assert(!verifyFunction(*TOI.Outline, &dbgs()));
#endif
    markDecayedPointerInArgStruct(TOI);
    replaceDecayedPointerInOutline(TOI);

    Function *Outlined = TOI.Outline;
    Instruction *ReplStart = TOI.ReplStart;
    CallBase *ReplCall = cast<CallBase>(TOI.ReplCall);
    BasicBlock *CallBlock = ReplStart->getParent();

    LLVMContext &C = M.getContext();
    const DataLayout &DL = M.getDataLayout();

    // At this point, we have a call in the parent to a function containing the
    // task body.  That function takes as its argument a pointer to a structure
    // containing the inputs to the task body.  This structure is initialized in
    // the parent immediately before the call.

    // Construct a call to jl_tapir_spawn:
    IRBuilder<> CallerIRBuilder(ReplCall);
    Value *OutlinedFnPtr =
        CallerIRBuilder.CreatePointerBitCastOrAddrSpaceCast(Outlined, SpawnFTy);
    AllocaInst *CallerArgStruct = cast<AllocaInst>(ReplCall->getArgOperand(0));
    Type *ArgsTy = CallerArgStruct->getAllocatedType();
    Value *ArgStructPtr =
        CallerIRBuilder.CreateBitCast(CallerArgStruct, Type::getInt8PtrTy(C));
    ConstantInt *ArgSize =
        ConstantInt::get(DL.getIntPtrType(C), DL.getTypeAllocSize(ArgsTy));

    AllocaInst *JVs;
    size_t NJVs;
    std::tie(JVs, NJVs) = insertJuliaValueSources(TOI);
    auto JVsPtr = CallerIRBuilder.CreateBitCast(JVs, T_ppjlvalue);
    auto JVsSize = ConstantInt::get(DL.getIntPtrType(C), NJVs);

    // Get the task group handle associatated with this detach instruction.
    // (NOTE: Since detach instruction is a terminator, we can use the basic
    // block containing it to identify the detach.)
    // TODO: Do I have access to task group inside nested tasks?
    // TODO: (If so, when are unneeded task groups cleaned up?)
    Value *TaskGroupPtr = DetachBlockToTaskGroup[TOI.ReplCall->getParent()];

    CallInst *Call =
        CallerIRBuilder.CreateCall(get_jl_tapir_spawn(),
                                   {
                                       TaskGroupPtr, // jl_value_t *tasks
                                       OutlinedFnPtr, // void *f
                                       ArgStructPtr, // void *arg
                                       ArgSize, // size_t arg_size
                                       JVsPtr, // jl_value_t **jvs
                                       JVsSize, // size_t jvs_size
                                       getWorldAge(ReplCall), // size_t world_age
                                   });
    Call->setDebugLoc(ReplCall->getDebugLoc());
    TOI.replaceReplCall(Call);
    ReplCall->eraseFromParent();
    // Now that `ReplCall` (that references `OldOutline`) is gone, we can remove
    // `OldOutline`:
    if (OldOutline != Outlined)
        OldOutline->eraseFromParent();

    CallerIRBuilder.SetInsertPoint(Call);
    CallerIRBuilder.CreateLifetimeStart(CallerArgStruct, ArgSize);
    CallerIRBuilder.CreateLifetimeStart(JVs, JVsSize);
    CallerIRBuilder.SetInsertPoint(CallBlock, ++Call->getIterator());
    CallerIRBuilder.CreateLifetimeEnd(CallerArgStruct, ArgSize);
    CallerIRBuilder.CreateLifetimeEnd(JVs, JVsSize);

    if (TOI.ReplUnwind)
        // TODO: Copied from Qthread; do we still need this?
        BranchInst::Create(TOI.ReplRet, CallBlock);
}

} // namespace LLVM

llvm::TapirTarget *jl_tapir_target_factory(llvm::Module &M)
{
    return new llvm::JuliaTapir(M);
}
