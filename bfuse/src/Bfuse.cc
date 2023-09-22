
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "clang/Tooling/CommonOptionsParser.h"
#include "llvm/Support/CommandLine.h"

#include "bfuse/Contexts.h"
#include "bfuse/Tools.h"
#include "bfuse/Utils.h"
#include "bfuse/Bfuse.h"

using namespace std;

using namespace clang::tooling;

using namespace bfuse::contexts;
using namespace bfuse::tools;
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
namespace bfuse {
//---------------------------------------------------------------------------
OptionsParserArguments::OptionsParserArguments(const string& ProgName, const string& CompileCommandsPath,
                                               const string& FilePath)
{
  compileCommandsPath = CompileCommandsPath;
  filePath            = FilePath;

  argv    = (const char**)malloc(sizeof(char *) * argc);
  argv[0] = ProgName.c_str();
  argv[1] = "-p";
  argv[2] = compileCommandsPath.c_str();
  argv[3] = filePath.c_str();
}
//---------------------------------------------------------------------------
OptionsParserArguments::~OptionsParserArguments() { free(argv); }
//---------------------------------------------------------------------------
void OptionsParserArguments::print() const
{
  cout << "================= OptionsParserArguments =================\n";
  cout << "argc: " << argc << "\n";
  cout << "argv: ";
  for (int i = 0; i < argc; ++i) {
    cout << argv[i] << " ";
  }
  cout << "\n";
}
//---------------------------------------------------------------------------
void bfuse(const string ProgName, const string CompileCommandsPath, 
           const string ConfigFilePath, const string ResultPath)
{
  string FusionInfoPath = ConfigFilePath + "/fusions.yaml";
  string KernelInfoPath = ConfigFilePath + "/kernels.yaml";

  // Extract information from yaml files
  auto FusionYAML = utils::readYAMLInfo<vector<FusionInfo>>(FusionInfoPath);
  auto KernelYAML = utils::readYAMLInfo<map<string, KernelInfo>>(KernelInfoPath);

  // Run block-level fusion
  for (auto& Info : FusionYAML) {
    if (!utils::checkFusionValid(Info, KernelYAML)) {
      ERROR_MESSAGE("invalid fusion definition exist");
      exit(0);
    }

    // Create compilation database
    string CodePath = CompileCommandsPath + "/" + utils::extractFilePath(Info, KernelYAML);
    OptionsParserArguments Args{ProgName, CompileCommandsPath, CodePath};

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

    FusionTool Tool{OptionsParser};

    FusionContext FContext   = FusionContext::create(Info, KernelYAML);
    AnalysisContext AContext = AnalysisContext::create(FContext);

    // 0. Backup files first
    // -----------------------------------------------------------------
    // Backup the kernel codes to be fused.
    // Because Clang libTooling is not suitable for creating new AST,
    // so rather than creating new AST from code, backup existing files,
    // and then refactoring the files. (Later recovery them)

    cout << "Backup files...\n";
    for (auto& S : OptionsParser.getSourcePathList()) {
      utils::backUpFiles(S);
    }

    // 1. Initially rewrite code to analyze and rewrite easily
    // -----------------------------------------------------------------
    // First, We need to rewrite whole body because when using libTooling
    // rewriting shared memory declarations may occur error. (Maybe bug?)
    // Then, this will append compound statement
    // so that we can be free from semantic error,
    // which can be occured during analyze and rewrite some variables.

    cout << "Initializing codes at first...\n";
    if (Tool.initiallyRewriteKernels(AContext)) {
      ERROR_MESSAGE("error occur while initialzing codes");
      exit(0);
    }
    if (Tool.rewriteCompStmt(AContext)) {
      ERROR_MESSAGE("error occur while rewriting compound statment");
      exit(0);
    }

    // 2. Renaming parameters
    // -----------------------------------------------------------------
    // Renaming kernels' parameters with kernels' name.
    // Because when fusing kernels, each parameters' name can be duplicated
    // in fused kernel.
    // i.e. ParmName -> KernelName + "_" + ParmName + "_";

    cout << "Renaming parameters...\n";
    if (Tool.analyzeParameters(AContext)) {
      ERROR_MESSAGE("error occur while analyzing parameters");
      exit(0);
    }
    if (Tool.renameParameters(AContext)) {
      ERROR_MESSAGE("error occur while renaming parameters");
      exit(0);
    }

    // 3. Rewrite pre-built variables
    // -----------------------------------------------------------------
    // Rewrite blockIdx and gridDim variables.
    // Because when fusing kernels, the semantics of blockIdx and gridDim
    // are changed.

    cout << "Rewriting pre-built variables...\n";
    if (Tool.rewriteCUDAVariables(AContext)) {
      ERROR_MESSAGE("error occur while rewriting CUDA pre-built variables");
      exit(0);
    }

    // 4. Renaming shared memory variables
    // -----------------------------------------------------------------
    // TODO: add comments

    cout << "Renaming shared memory declarations...\n";
    if (Tool.extractSharedDecls(AContext)) {
      ERROR_MESSAGE("error occur while extracting shared memory declarations");
      exit(0);
    }
    // if (Tool.hoistSharedDecls(AContext)) {
    //   ERROR_MESSAGE("error occur while rewriting shared memory declarations");
    //   exit(0);
    // }
    if (Tool.analyzeSharedVariables(AContext)) {
      ERROR_MESSAGE("error occur while renaming shared memory variables");
      exit(0);
    }
    if (Tool.renameSharedVariables(AContext)) {
      ERROR_MESSAGE("error occur while renaming shared memory variables");
      exit(0);
    }

    // 5. Create fused kerenl
    // -----------------------------------------------------------------
    // Fuse two different kernels and
    // save it into the result path directory.

    cout << "Creating fused kernel...\n";
    if (Tool.createFusedKernel(AContext)) {
      ERROR_MESSAGE("error occur while creating new function");
      exit(0);
    }
    if (Tool.saveFusedKernel(AContext, ResultPath)) {
      ERROR_MESSAGE("error occur while saving new function");
      exit(0);
    }

    // 6. Recover files
    // -----------------------------------------------------------------
    // Recover files.

    cout << "Recovering files...\n";
    for (auto &S : OptionsParser.getSourcePathList()) {
      utils::recoverFiles(S);
    }
  }
}
//---------------------------------------------------------------------------
} // namespace bfuse
//---------------------------------------------------------------------------