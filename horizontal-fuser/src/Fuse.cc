
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "clang/Tooling/CommonOptionsParser.h"
#include "llvm/Support/CommandLine.h"

#include "fuse/Fuse.h"
#include "fuse/Contexts.h"
#include "fuse/Utils.h"
#include "fuse/Tools.h"

using namespace std;

using namespace clang::tooling;

using namespace fuse::contexts;
using namespace fuse::tools;
//---------------------------------------------------------------------------
// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("my-tool options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static llvm::cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static llvm::cl::extrahelp MoreHelp("\nMore help text...\n");
//---------------------------------------------------------------------------
namespace fuse {
//---------------------------------------------------------------------------
Arguments::Arguments(const std::string &ProgName, const std::string &CompileCommandsPath, const std::string &FilePath)
{
  CompileCommands_ = CompileCommandsPath;
  File_            = FilePath;

  Argc_ = 4;

  Argv_    = (const char**)malloc(sizeof(char *) * Argc_);
  Argv_[0] = ProgName.c_str();
  Argv_[1] = "-p";
  Argv_[2] = CompileCommandsPath.c_str();
  Argv_[3] = FilePath.c_str();
}
//---------------------------------------------------------------------------
Arguments::~Arguments() { free(Argv_); }
//---------------------------------------------------------------------------
void Arguments::print() const
{
  cout << "================= OptionsParserArguments =================\n";
  cout << "argc: " << Argc_ << "\n";
  cout << "argv: ";
  for (int i = 0; i < Argc_; ++i) {
    cout << Argv_[i] << " ";
  }
  cout << "\n";
}
//---------------------------------------------------------------------------
void bfuse(const string ProgName, const string FusionConfigPath,
           const string KernelConfigPath, const string CompileCommandsPath, const string OutputPath)
{
  // Extract information from yaml files
  auto FusionYAML = utils::readYAMLInfo<vector<FusionInfo>>(FusionConfigPath);
  auto KernelYAML = utils::readYAMLInfo<map<string, KernelInfo>>(KernelConfigPath);

  // Results & Output YAML
  string Results = "";
  map<string, KernelInfo> FusedKernelYAML;

  // Run block-level fusion
  // for (auto& Info : FusionYAML) {
  for (long unsigned I = 0; I < FusionYAML.size(); ++I) {
    auto& Info = FusionYAML[I];

    // Create compilation database
    string CodePath = CompileCommandsPath + "/" + utils::extractFilePath(Info);
    Arguments Args{ProgName, CompileCommandsPath, CodePath};

    auto [argc, argv]   = Args.getArguments();
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
    if (!ExpectedParser) {
      // Fail gracefully for unsupported options
      llvm::errs() << ExpectedParser.takeError();
      exit(0);
    }
    CommonOptionsParser &OptionsParser = ExpectedParser.get();

    /*
     * The Fusion-Flow
     *
     * First,  analysis the kernels and extract data to be used
     * Second, rewrite kernels (Parameters, Synchronization, ThreadIdx, BlockIdx, etc...)
     *         this rewriting rule needs to add  "Union Shared Memory Variable" Algorithm
     * Third,  create new fused function and save it into the disk
     * THE END
     */

    FusionContext FContext = FusionContext::create(Info, KernelYAML, /**bfuse=*/true);
    FusionTool Tool{OptionsParser, FContext};

    cout << "[" << I + 1 << "/" << FusionYAML.size() << "]"
         << " Start to bfuse \"" << FContext.FusedKernelName_ << "\" kernel...\n";

    // 0. Backup files first
    // -----------------------------------------------------------------
    // Backup the kernel codes to be fused.
    // Because Clang libTooling is not suitable for creating new AST,
    // so rather than creating new AST from code, backup existing files,
    // and then refactoring the files. (Later recovery them)

    cout << "Backup files...\n";
    for (auto& S : OptionsParser.getSourcePathList()) {
      utils::backUpFile(S);
    }

    // 1. Initially rewrite code to analyze and rewrite easily
    // -----------------------------------------------------------------
    // First, We need to rewrite whole body because when using libTooling
    // rewriting shared memory declarations may occur error. (Maybe bug?)
    // Then, this will append compound statement
    // so that we can be free from semantic error,
    // which can be occured during analyze and rewrite some variables.

    cout << "Initializing codes at first...\n";
    if (Tool.initiallyRewriteKernels()) {
      ERROR_MESSAGE("error occur while initialzing codes");
      exit(1);
    }
    if (Tool.rewriteCompStmt()) {
      ERROR_MESSAGE("error occur while rewriting compound statment");
      exit(1);
    }

    // 2. Renaming parameters
    // -----------------------------------------------------------------
    // Renaming kernels' parameters with kernels' name.
    // Because when fusing kernels, each parameters' name can be duplicated
    // in fused kernel.
    // i.e. ParmName -> KernelName + "_" + ParmName + "_";

    cout << "Renaming parameters...\n";
    if (Tool.renameParameters()) {
      ERROR_MESSAGE("error occur while renaming parameters");
      exit(1);
    }

    // 3. Rewrite pre-built variables
    // -----------------------------------------------------------------
    // Rewrite blockIdx and gridDim variables.
    // Because when fusing kernels, the semantics of blockIdx and gridDim
    // are changed.

    cout << "Rewriting CUDA pre-built variables...\n";
    if (Tool.rewriteCUDAVariables()) {
      ERROR_MESSAGE("error occur while rewriting CUDA pre-built variables");
      exit(1);
    }

    // 4. Hoisting & Renaming shared memory variables
    // -----------------------------------------------------------------
    // TODO: add comments

    cout << "Hoisting & Renaming shared memory variables...\n";
    if (Tool.hoistSharedDecls()) {
      ERROR_MESSAGE("error occur while extracting shared memory declarations");
      exit(1);
    }
    if (Tool.renameSharedVariables()) {
      ERROR_MESSAGE("error occur while renaming shared memory variables");
      exit(1);
    }

    // 5. Create fused kerenl
    // -----------------------------------------------------------------
    // Fuse two different kernels and
    // save it in the result path directory.

    cout << "Creating fused kernel...\n";
    if (Tool.createBFuseKernel()) {
      ERROR_MESSAGE("error occur while creating new function");
      exit(1);
    }

    // 6. Recover files
    // -----------------------------------------------------------------
    // Recover files.

    cout << "Recovering files...\n";
    for (auto &S : OptionsParser.getSourcePathList()) {
      utils::recoverFile(S);
    }

    // 6. Save fused kernel's information
    // -----------------------------------------------------------------
    // Save fused kernel's information.
    Results += Tool.getFuncStr();
    FusedKernelYAML[FContext.FusedKernelName_] = KernelInfo(FContext.FusedKernelName_,
                                                            true,
                                                            FContext.FusedGridDim_,
                                                            FContext.FusedBlockDim_,
                                                            32,
                                                            -1);
  }

  utils::writeFile(OutputPath, "bfuse_kernels.cu", Results);
  utils::writeYAMLInfo<map<string, KernelInfo>>(OutputPath, "bfuse_kernels.yaml", FusedKernelYAML);
}
//---------------------------------------------------------------------------
void hfuse(const string ProgName, const string FusionConfigPath,
           const string KernelConfigPath, const string CompileCommandsPath, const string OutputPath)
{
  // Extract information from yaml files
  auto FusionYAML = utils::readYAMLInfo<vector<FusionInfo>>(FusionConfigPath);
  auto KernelYAML = utils::readYAMLInfo<map<string, KernelInfo>>(KernelConfigPath);

  // Results & Output YAML
  string Results = "";
  map<string, KernelInfo> FusedKernelYAML;

  // Run block-level fusion
  // for (auto& Info : FusionYAML) {
  for (long unsigned I = 0; I < FusionYAML.size(); ++I) {
    auto& Info = FusionYAML[I];

    // Create compilation database
    string CodePath = CompileCommandsPath + "/" + utils::extractFilePath(Info);
    Arguments Args{ProgName, CompileCommandsPath, CodePath};

    auto [argc, argv]   = Args.getArguments();
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
    if (!ExpectedParser) {
      // Fail gracefully for unsupported options
      llvm::errs() << ExpectedParser.takeError();
      exit(0);
    }
    CommonOptionsParser &OptionsParser = ExpectedParser.get();

    /*
     * The Fusion-Flow
     *
     * First,  analysis the kernels and extract data to be used
     * Second, rewrite kernels (Parameters, Synchronization, ThreadIdx, BlockIdx, etc...)
     *         this rewriting rule needs to add  "Union Shared Memory Variable" Algorithm
     * Third,  create new fused function and save it into the disk
     * THE END
     */

    FusionContext FContext = FusionContext::create(Info, KernelYAML, /*bfuse=*/false);
    FusionTool Tool{OptionsParser, FContext};

    cout << "[" << I + 1 << "/" << FusionYAML.size() << "]"
         << " Start to hfuse \"" << FContext.FusedKernelName_ << "\" kernel...\n";

    // 0. Backup files first
    // -----------------------------------------------------------------
    // Backup the kernel codes to be fused.
    // Because Clang libTooling is not suitable for creating new AST,
    // so rather than creating new AST from code, backup existing files,
    // and then refactoring the files. (Later recovery them)

    cout << "Backup files...\n";
    for (auto& S : OptionsParser.getSourcePathList()) {
      utils::backUpFile(S);
    }

    // 1. Initially rewrite code to analyze and rewrite easily
    // -----------------------------------------------------------------
    // First, We need to rewrite whole body because when using libTooling
    // rewriting shared memory declarations may occur error. (Maybe bug?)
    // Then, this will append compound statement
    // so that we can be free from semantic error,
    // which can be occured during analyze and rewrite some variables.

    cout << "Initializing codes at first...\n";
    if (Tool.initiallyRewriteKernels()) {
      ERROR_MESSAGE("error occur while initialzing codes");
      exit(1);
    }
    if (Tool.rewriteCompStmt()) {
      ERROR_MESSAGE("error occur while rewriting compound statment");
      exit(1);
    }

    // 2. Renaming parameters
    // -----------------------------------------------------------------
    // Renaming kernels' parameters with kernels' name.
    // Because when fusing kernels, each parameters' name can be duplicated
    // in fused kernel.
    // i.e. ParmName -> KernelName + "_" + ParmName + "_";

    cout << "Renaming parameters...\n";
    if (Tool.renameParameters()) {
      ERROR_MESSAGE("error occur while renaming parameters");
      exit(1);
    }

    // 3. Rewrite pre-built variables
    // -----------------------------------------------------------------
    // Rewrite blockIdx and gridDim variables.
    // Because when fusing kernels, the semantics of blockIdx and gridDim
    // are changed.

    cout << "Rewriting CUDA pre-built variables...\n";
    if (Tool.rewriteCUDAVariables()) {
      ERROR_MESSAGE("error occur while rewriting CUDA pre-built variables");
      exit(1);
    }

    // 4. Rewrite __syncthreads() functions
    // -----------------------------------------------------------------
    // TODO: add comments

    cout << "Rewriting CUDA __syncthreads() functions...\n";
    if (Tool.rewriteCUDASynchronize()) {
      ERROR_MESSAGE("error occur while rewriting CUDA __syncthreads() functions");
      exit(1);
    }

    // 5. Create fused kerenl
    // -----------------------------------------------------------------
    // Fuse two different kernels and
    // save it in the result path directory.

    cout << "Creating fused kernel...\n";
    if (Tool.createHFuseKernel()) {
      ERROR_MESSAGE("error occur while creating new function");
      exit(1);
    }

    // 6. Recover files
    // -----------------------------------------------------------------
    // Recover files.

    cout << "Recovering files...\n";
    for (auto &S : OptionsParser.getSourcePathList()) {
      utils::recoverFile(S);
    }

    // 6. Save fused kernel's information
    // -----------------------------------------------------------------
    // Save fused kernel's information.
    Results += Tool.getFuncStr();
    FusedKernelYAML[FContext.FusedKernelName_] = KernelInfo(FContext.FusedKernelName_,
                                                            true,
                                                            FContext.FusedGridDim_,
                                                            FContext.FusedBlockDim_,
                                                            32,
                                                            -1);
  }

  utils::writeFile(OutputPath, "hfuse_kernels.cu", Results);
  utils::writeYAMLInfo<map<string, KernelInfo>>(OutputPath, "hfuse_kernels.yaml", FusedKernelYAML);
}
//---------------------------------------------------------------------------
} // namespace fuse
//---------------------------------------------------------------------------