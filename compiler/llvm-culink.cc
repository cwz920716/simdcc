//===- llvm-culink.cc - LLVM linker For Cuda ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility may be invoked in the following manner:
//  llvm-culink a.bc b.bc c.bc -o x.bc
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Transforms/IPO/FunctionImport.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Utils/FunctionImportUtils.h"

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>

#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace boost::algorithm;

#include "llvm_utils.h"

static cl::list<std::string>
InputFilenames(cl::Positional, cl::OneOrMore,
               cl::desc("<input bitcode files>"));

static cl::list<std::string> OverridingInputs(
    "override", cl::ZeroOrMore, cl::value_desc("filename"),
    cl::desc(
        "input bitcode file which can override previously defined symbol(s)"));

// Option to simulate function importing for testing. This enables using
// llvm-link to simulate ThinLTO backend processes.
static cl::list<std::string> Imports(
    "import", cl::ZeroOrMore, cl::value_desc("function:filename"),
    cl::desc("Pair of function name and filename, where function should be "
             "imported from bitcode in filename"));

// Option to support testing of function importing. The module summary
// must be specified in the case were we request imports via the -import
// option, as well as when compiling any module with functions that may be
// exported (imported by a different llvm-link -import invocation), to ensure
// consistent promotion and renaming of locals.
static cl::opt<std::string>
    SummaryIndex("summary-index", cl::desc("Module summary index filename"),
                 cl::init(""), cl::value_desc("filename"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Override output filename"), cl::init("-"),
               cl::value_desc("filename"));

static cl::opt<bool>
Internalize("internalize", cl::desc("Internalize linked symbols"));

static cl::opt<bool>
    DisableDITypeMap("disable-debug-info-type-map",
                     cl::desc("Don't use a uniquing type map for debug info"));

static cl::opt<bool>
OnlyNeeded("only-needed", cl::desc("Link only needed symbols"));

static cl::opt<bool>
Force("f", cl::desc("Enable binary output on terminals"));

static cl::opt<bool>
    DisableLazyLoad("disable-lazy-loading",
                    cl::desc("Disable lazy module loading"));

static cl::opt<bool>
    DisableOutput("disable-output",
                    cl::desc("Disable output file"));

static cl::opt<bool>
    OutputAssembly("S", cl::desc("Write output as LLVM assembly"), cl::Hidden);

static cl::opt<bool>
Verbose("v", cl::desc("Print information about actions taken"));

static cl::opt<bool>
DumpAsm("d", cl::desc("Print assembly as linked"), cl::Hidden);

static cl::opt<bool>
SuppressWarnings("suppress-warnings", cl::desc("Suppress all linking warnings"),
                 cl::init(false));

static cl::opt<bool> PreserveBitcodeUseListOrder(
    "preserve-bc-uselistorder",
    cl::desc("Preserve use-list order when writing LLVM bitcode."),
    cl::init(true), cl::Hidden);

static cl::opt<bool> PreserveAssemblyUseListOrder(
    "preserve-ll-uselistorder",
    cl::desc("Preserve use-list order when writing LLVM assembly."),
    cl::init(false), cl::Hidden);

static ExitOnError ExitOnErr;

// Read the specified bitcode file in and return it. This routine searches the
// link path for the specified file to try to find it...
//
static std::unique_ptr<Module> loadFile(const char *argv0,
                                        const std::string &FN,
                                        LLVMContext &Context,
                                        bool MaterializeMetadata = true) {
  SMDiagnostic Err;
  if (Verbose) errs() << "Loading '" << FN << "'\n";
  std::unique_ptr<Module> Result;
  if (DisableLazyLoad)
    Result = parseIRFile(FN, Err, Context);
  else
    Result = getLazyIRFileModule(FN, Err, Context, !MaterializeMetadata);

  if (!Result) {
    Err.print(argv0, errs());
    return nullptr;
  }

  if (MaterializeMetadata) {
    ExitOnErr(Result->materializeMetadata());
    UpgradeDebugInfo(*Result);
  }

  return Result;
}

namespace {

/// Helper to load on demand a Module from file and cache it for subsequent
/// queries during function importing.
class ModuleLazyLoaderCache {
  /// Cache of lazily loaded module for import.
  StringMap<std::unique_ptr<Module>> ModuleMap;

  /// Retrieve a Module from the cache or lazily load it on demand.
  std::function<std::unique_ptr<Module>(const char *argv0,
                                        const std::string &FileName)>
      createLazyModule;

public:
  /// Create the loader, Module will be initialized in \p Context.
  ModuleLazyLoaderCache(std::function<std::unique_ptr<Module>(
                            const char *argv0, const std::string &FileName)>
                            createLazyModule)
      : createLazyModule(std::move(createLazyModule)) {}

  /// Retrieve a Module from the cache or lazily load it on demand.
  Module &operator()(const char *argv0, const std::string &FileName);

  std::unique_ptr<Module> takeModule(const std::string &FileName) {
    auto I = ModuleMap.find(FileName);
    assert(I != ModuleMap.end());
    std::unique_ptr<Module> Ret = std::move(I->second);
    ModuleMap.erase(I);
    return Ret;
  }
};

// Get a Module for \p FileName from the cache, or load it lazily.
Module &ModuleLazyLoaderCache::operator()(const char *argv0,
                                          const std::string &Identifier) {
  auto &Module = ModuleMap[Identifier];
  if (!Module)
    Module = createLazyModule(argv0, Identifier);
  return *Module;
}
} // anonymous namespace

namespace {
struct LLVMLinkDiagnosticHandler : public DiagnosticHandler {
  bool handleDiagnostics(const DiagnosticInfo &DI) override {
    unsigned Severity = DI.getSeverity();
    switch (Severity) {
    case DS_Error:
      errs() << "ERROR: ";
      break;
    case DS_Warning:
      if (SuppressWarnings)
        return true;
      errs() << "WARNING: ";
      break;
    case DS_Remark:
    case DS_Note:
      llvm_unreachable("Only expecting warnings and errors");
    }

    DiagnosticPrinterRawOStream DP(errs());
    DI.print(DP);
    errs() << '\n';
    return true;
  }
};
}

/// Import any functions requested via the -import option.
static bool importFunctions(const char *argv0, Module &DestModule) {
  if (SummaryIndex.empty())
    return true;
  std::unique_ptr<ModuleSummaryIndex> Index =
      ExitOnErr(llvm::getModuleSummaryIndexForFile(SummaryIndex));

  // Map of Module -> List of globals to import from the Module
  FunctionImporter::ImportMapTy ImportList;

  auto ModuleLoader = [&DestModule](const char *argv0,
                                    const std::string &Identifier) {
    return loadFile(argv0, Identifier, DestModule.getContext(), false);
  };

  ModuleLazyLoaderCache ModuleLoaderCache(ModuleLoader);
  for (const auto &Import : Imports) {
    // Identify the requested function and its bitcode source file.
    size_t Idx = Import.find(':');
    if (Idx == std::string::npos) {
      errs() << "Import parameter bad format: " << Import << "\n";
      return false;
    }
    std::string FunctionName = Import.substr(0, Idx);
    std::string FileName = Import.substr(Idx + 1, std::string::npos);

    // Load the specified source module.
    auto &SrcModule = ModuleLoaderCache(argv0, FileName);

    if (verifyModule(SrcModule, &errs())) {
      errs() << argv0 << ": " << FileName
             << ": error: input module is broken!\n";
      return false;
    }

    Function *F = SrcModule.getFunction(FunctionName);
    if (!F) {
      errs() << "Ignoring import request for non-existent function "
             << FunctionName << " from " << FileName << "\n";
      continue;
    }
    // We cannot import weak_any functions without possibly affecting the
    // order they are seen and selected by the linker, changing program
    // semantics.
    if (F->hasWeakAnyLinkage()) {
      errs() << "Ignoring import request for weak-any function " << FunctionName
             << " from " << FileName << "\n";
      continue;
    }

    if (Verbose)
      errs() << "Importing " << FunctionName << " from " << FileName << "\n";

    auto &Entry = ImportList[FileName];
    Entry.insert(std::make_pair(F->getGUID(), /* (Unused) threshold */ 1.0));
  }
  auto CachedModuleLoader = [&](StringRef Identifier) {
    return ModuleLoaderCache.takeModule(Identifier);
  };
  FunctionImporter Importer(*Index, CachedModuleLoader);
  ExitOnErr(Importer.importFunctions(DestModule, ImportList));

  return true;
}

static bool linkFiles(const char *argv0, LLVMContext &Context, Linker &L,
                      const cl::list<std::string> &Files,
                      unsigned Flags) {
  // Filter out flags that don't apply to the first file we load.
  unsigned ApplicableFlags = Flags & Linker::Flags::OverrideFromSrc;
  // Similar to some flags, internalization doesn't apply to the first file.
  bool InternalizeLinkedSymbols = false;
  for (const auto &File : Files) {
    std::unique_ptr<Module> M = loadFile(argv0, File, Context);
    if (!M.get()) {
      errs() << argv0 << ": error loading file '" << File << "'\n";
      return false;
    }

    // Note that when ODR merging types cannot verify input files in here When
    // doing that debug metadata in the src module might already be pointing to
    // the destination.
    if (DisableDITypeMap && verifyModule(*M, &errs())) {
      errs() << argv0 << ": " << File << ": error: input module is broken!\n";
      return false;
    }

    // If a module summary index is supplied, load it so linkInModule can treat
    // local functions/variables as exported and promote if necessary.
    if (!SummaryIndex.empty()) {
      std::unique_ptr<ModuleSummaryIndex> Index =
          ExitOnErr(llvm::getModuleSummaryIndexForFile(SummaryIndex));

      // Conservatively mark all internal values as promoted, since this tool
      // does not do the ThinLink that would normally determine what values to
      // promote.
      for (auto &I : *Index) {
        for (auto &S : I.second.SummaryList) {
          if (GlobalValue::isLocalLinkage(S->linkage()))
            S->setLinkage(GlobalValue::ExternalLinkage);
        }
      }

      // Promotion
      if (renameModuleForThinLTO(*M, *Index))
        return true;
    }

    if (Verbose)
      errs() << "Linking in '" << File << "'\n";

    bool Err = false;
    if (InternalizeLinkedSymbols) {
      Err = L.linkInModule(
          std::move(M), ApplicableFlags, [](Module &M, const StringSet<> &GVS) {
            internalizeModule(M, [&GVS](const GlobalValue &GV) {
              return !GV.hasName() || (GVS.count(GV.getName()) == 0);
            });
          });
    } else {
      Err = L.linkInModule(std::move(M), ApplicableFlags);
    }

    if (Err)
      return false;

    // Internalization applies to linking of subsequent files.
    InternalizeLinkedSymbols = Internalize;

    // All linker flags apply to linking of subsequent files.
    ApplicableFlags = Flags;
  }

  return true;
}

class FatbinRegisterInstVisitor:
  public llvm::InstVisitor<FatbinRegisterInstVisitor> {
 public:
  std::vector<llvm::Value *> registry;

  void visitCallInst(llvm::CallInst &call) {
    LLVM_STRING(__cudaRegisterFatBinary);
    if (call.getCalledFunction()->getName() == __cudaRegisterFatBinary) {
      registry.push_back(&call);
    }
  }
};

enum VarFlags {
  kDefault = 0,
  kExtern = 1,
  kConst = 2,
};

struct DeviceVar {
 public:
  DeviceVar(const std::string &n, int f = kDefault):
      name(n), flag(f), llvm_global_var(nullptr) {}

  std::string name;
  int flag;
  llvm::GlobalVariable *llvm_global_var;
};

static std::vector<DeviceVar> ParseDeviceVarsFromFile(void) {
  std::vector<DeviceVar> vars;
  std::ifstream device_vars_file("deviceVars.decl");
  if (!device_vars_file) {
    errs() << "WARNING: Cannot open deviceVars.decl. File may not exist.\n";
    return vars;
  }

  std::string decl;
  while (std::getline(device_vars_file, decl)) {
    std::vector<std::string> tokens;
    const char *sep = "\n\t ;*()";
    boost::trim_if(decl, boost::is_any_of(sep));
    boost::split(tokens, decl, is_any_of(sep));

    std::string name = tokens[tokens.size() - 1];
    int flag = kDefault;
    for (auto &token : tokens) {
      if (token == "extern") {
        flag |= kExtern;
      }
      if (token == "const") {
        flag |= kExtern;
      }
    }
    DeviceVar var(name, flag);
    vars.push_back(var);
  }

  return vars;
}

static void FixCUDARegistry(Module &Module, LLVMContext &Context) {
  if (Verbose) errs() << "Fix CUDA Registry...\n";
  LLVM_STRING(__cuda_register_globals);
  LLVM_STRING(__cuda_module_ctor);

  auto register_kernel_func = Module.getFunction(__cuda_register_globals);
  register_kernel_func = Module.getFunction(__cuda_module_ctor);
  if (!register_kernel_func) {
    register_kernel_func = Module.getFunction(__cuda_module_ctor);
  }

  if (!register_kernel_func) {
    if (Verbose) errs() << "Skip non-cuda-host modules.\n";
  }

  // get the fatBinary handle.
  std::vector<llvm::Value *> gpu_binary_handles;
  if (register_kernel_func->getName() == __cuda_register_globals) {
    gpu_binary_handles.push_back(register_kernel_func->arg_begin());
  } else {
    FatbinRegisterInstVisitor fatbin_register_ivisitor;
    fatbin_register_ivisitor.visit(*register_kernel_func);
    gpu_binary_handles = std::move(fatbin_register_ivisitor.registry);
  }
  if (Verbose) errs() << "List Gpu Binary Handles...\n";
  for (auto handle : gpu_binary_handles) {
    if (Verbose) errs() << *handle << "\n";
  }

  // parse device-wide variables, i.e., __device__;
  auto device_vars = ParseDeviceVarsFromFile();
  for (auto &v : device_vars) {
    auto gv = Module.getNamedGlobal(v.name);
    if (!gv) {
      errs() << "WARNING: Variabe " << v.name << " not found!\n";
    }
    v.llvm_global_var = gv;
  }
  
  llvm::Type *type_int32 = llvm::IntegerType::get(Context, 32);
  llvm::Type *type_int8 = llvm::IntegerType::get(Context, 8);
  llvm::Type *type_void_ptr = llvm::PointerType::get(type_int8, 0);
  llvm::Type *type_char_ptr = llvm::PointerType::get(type_int8, 0);
  llvm::Type *type_void_ptr_ptr = llvm::PointerType::get(type_void_ptr, 0);

  LLVM_STRING(__cudaRegisterVar);
  llvm::Constant *register_var_func = Module.getFunction(__cudaRegisterVar);
  if (!register_var_func) {
    // void __cudaRegisterVar(void **, char *, char *, const char *,
    //                        int, int, int, int)
    llvm::Type *register_var_params[] = {type_void_ptr_ptr, type_char_ptr, type_char_ptr,
                                       type_char_ptr,    type_int32,     type_int32,
                                       type_int32,        type_int32};
    auto func_type =
        llvm::FunctionType::get(type_int32, register_var_params, false);
    register_var_func =
        Module.getOrInsertFunction(__cudaRegisterVar, func_type);
  }

  for (auto &BB : *register_kernel_func) {
    auto terminator = BB.getTerminator();
    if (ReturnInst *ret = dyn_cast<ReturnInst>(terminator)) {
      // locate the return instruction
      llvm::IRBuilder<> builder(ret);
      // void __cudaRegisterVar(void **, char *, char *, const char *,
      //                        int, int, int, int)
      for (auto gpu_binary_handle : gpu_binary_handles) {
        for (auto &dv : device_vars) {
          llvm::GlobalVariable *var = dv.llvm_global_var;
          if (var == nullptr) {
            continue;
          }

          uint64_t var_size =
              Module.getDataLayout().getTypeAllocSize(var->getValueType());
          auto var_name =
              builder.CreateGlobalStringPtr(var->getName());
          auto bitcast = builder.CreateBitCast(var, type_void_ptr);
          llvm::Value *args[] = {
            gpu_binary_handle,
            bitcast,
            var_name,
            var_name,
            /* ExternDeviceVar */
            llvm::ConstantInt::get(type_int32, (dv.flag & kExtern) ? 1 : 0),
            /* Var Size */
            llvm::ConstantInt::get(type_int32, var_size),
            /* ConstantDeviceVar */
            llvm::ConstantInt::get(type_int32, (dv.flag & kConst) ? 1 : 0),
            /* 0 */
            llvm::ConstantInt::get(type_int32, 0)
          };
          builder.CreateCall(register_var_func, args);
          if (Verbose) errs() << "Register Var " << dv.name << ".\n";
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);

  ExitOnErr.setBanner(std::string(argv[0]) + ": ");

  LLVMContext Context;
  Context.setDiagnosticHandler(
      llvm::make_unique<LLVMLinkDiagnosticHandler>(), true);

  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.
  cl::ParseCommandLineOptions(argc, argv, "llvm linker\n");

  if (!DisableDITypeMap)
    Context.enableDebugTypeODRUniquing();

  auto Composite = make_unique<Module>("llvm-nvlink", Context);
  Linker L(*Composite);

  unsigned Flags = Linker::Flags::None;
  if (OnlyNeeded)
    Flags |= Linker::Flags::LinkOnlyNeeded;

  // First add all the regular input files
  if (!linkFiles(argv[0], Context, L, InputFilenames, Flags))
    return 1;

  // Next the -override ones.
  if (!linkFiles(argv[0], Context, L, OverridingInputs,
                 Flags | Linker::Flags::OverrideFromSrc))
    return 1;

  // Import any functions requested via -import
  if (!importFunctions(argv[0], *Composite))
    return 1;

  if (DumpAsm) errs() << "Here's the assembly:\n" << *Composite;

  std::error_code EC;
  ToolOutputFile Out(OutputFilename, EC, sys::fs::F_None);
  if (EC) {
    errs() << EC.message() << '\n';
    return 1;
  }

  if (verifyModule(*Composite, &errs())) {
    errs() << argv[0] << ": error: linked module is broken!\n";
    return 1;
  }

  FixCUDARegistry(*Composite, Context);

  if (DisableOutput) {
    return 0;
  }

  if (Verbose) errs() << "Writing bitcode...\n";
  if (OutputAssembly) {
    Composite->print(Out.os(), nullptr, PreserveAssemblyUseListOrder);
  } else if (Force || !CheckBitcodeOutputToConsole(Out.os(), true))
    WriteBitcodeToFile(Composite.get(), Out.os(), PreserveBitcodeUseListOrder);

  // Declare success.
  Out.keep();

  return 0;
}
