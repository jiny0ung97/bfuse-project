
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <map>

// Declares clang::SyntaxOnlyAction
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

// Declare llvm::cl::extrahelp
#include "llvm/Support/CommandLine.h"

#include "bfuse/Contexts.h"
#include "bfuse/Matchers.h"
#include "bfuse/Utils.h"
#include "bfuse/Bfuse.h"

using namespace std;

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

using namespace bfuse::contexts;
using namespace bfuse::matchers;
using namespace bfuse::utils;
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

// Clang AST Matchers
static DeclarationMatcher FunctionDeclMatcher
        = functionDecl(
            hasAttr(attr::CUDAGlobal)
          ).bind(CUDAFunctionDeclBindId);
//---------------------------------------------------------------------------
namespace bfuse {
//---------------------------------------------------------------------------
OptionsParserArguments::OptionsParserArguments(const char *ProgName,
                                               string& CompileCommandsPath, string& FilePath)
{
  compileCommandsPath = CompileCommandsPath;
  filePath            = FilePath;

  argv    = (const char**)malloc(sizeof(char *) * argc);
  argv[0] = ProgName;
  argv[1] = "-p";
  argv[2] = compileCommandsPath.c_str();
  argv[3] = filePath.c_str();

  // cout << "argc: " << argc << "\n";
  // cout << "argv: ";
  // for (int i = 0; i < argc; ++i) {
  //   cout << argv[i] << " ";
  // }
  // cout << "\n";
}
//---------------------------------------------------------------------------
OptionsParserArguments::~OptionsParserArguments() { free(argv); }
//---------------------------------------------------------------------------
void KernelInfo::print(const string& KName) const
{
  cout << "[KernelInfo]\n";
  cout << KName << "\n";
  cout << "  File: " << filePath << "\n";
  cout << "  Barriers: " << hasBarriers << "\n";
  cout << "  GridDim:\n";
  cout << "    X: " << gridDim.x << "\n";
  cout << "    Y: " << gridDim.y << "\n";
  cout << "    Z: " << gridDim.z << "\n";
  cout << "  BlockDim:\n";
  cout << "    X: " << blockDim.x << "\n";
  cout << "    Y: " << blockDim.y << "\n";
  cout << "    Z: " << blockDim.z << "\n\n";
}
//---------------------------------------------------------------------------
void FusionInfo::print() const
{
  cout << "[FusionInfo]\n";
  cout << "  - Kernels:\n";
  for (auto& KName : kernels) {
    cout << "    - " << KName << "\n";
  }
  cout << "\n";
}
//---------------------------------------------------------------------------
void bfuse(const char *ProgName, string ConfigFilePath, string CompileCommandsPath)
{
  string FusionInfoPath = ConfigFilePath + "/fusions.yaml";
  string KernelInfoPath = ConfigFilePath + "/kernels.yaml";

  // Extract information from yaml files
  auto FusionYAML = readYAMLInfo<vector<FusionInfo>>(FusionInfoPath);
  auto KernelYAML = readYAMLInfo<map<string, KernelInfo>>(KernelInfoPath);

  // Run block-level fusion
  for (auto& Info : FusionYAML) {
    if (!checkFusionValid(Info, KernelYAML)) {
      ERROR_MESSAGE("invalid fusion definition exist.");
      exit(0);
    }

    string CodePath = CompileCommandsPath + "/" + extractFilePath(Info, KernelYAML);

    OptionsParserArguments Args{ProgName, CompileCommandsPath, CodePath};
    FusionContext          Context{Info, KernelYAML};

    // [Tests]
    // Context.print();

    // 1. First, analysis the kernels and plan how to rewrite the kernels
    // 2. Second, rewrite kernels (Parameters, Synchronization, ThreadIdx, BlockIdx, etc...)
    // 3. Need to add share "Shared Memory Variable" Algorithm
    // 4. Third, temporally save rewrite to temp file and make fused kernel
    // %. THE END

    // 1. Get compilation database
    auto [argc, argv]   = Args.getArguments();
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
    if (!ExpectedParser) {
      // Fail gracefully for unsupported options
      llvm::errs() << ExpectedParser.takeError();
      exit(0);
    }
    CommonOptionsParser& OptionsParser = ExpectedParser.get();
    ClangTool Tool(OptionsParser.getCompilations(),
                   OptionsParser.getSourcePathList());


    // 2. Build my matcher (analysis) & Run
    // [Test]
    CUDAFunctionDeclPrinter Printer;
    MatchFinder Finder;
    Finder.addMatcher(FunctionDeclMatcher, &Printer);
    Tool.run(newFrontendActionFactory(&Finder).get());

    // 3. Build my matcher (rewriter) & Run
    //    Each matcher should include rewriter
    // ... My Something ...
    // MatchFinder Finder;
    // Finder.addMatcher(...);
    // Tool.run(...);

    // 4. Rewrite to temp llvm::raw_string_ostream
    // Rewriter.getEditBuffer(...).write(...);

    // 5. build AST from llvm::raw_string_ostream
    // unique_ptr<ASTUnit> Unit = buildASTFromCode(... .str());

    // 6. Extract bodies and Set it
    // TODO:
  }
}
//---------------------------------------------------------------------------
} // namespace bfuse
//---------------------------------------------------------------------------