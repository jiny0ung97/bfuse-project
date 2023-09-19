
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
OptionsParserArguments::OptionsParserArguments(const char* ProgName,
                                               string& CompileCommandsPath, string& FilePath)
{
  compileCommandsPath = CompileCommandsPath;
  filePath            = FilePath;

  argv    = (const char**)malloc(sizeof(char *) * argc);
  argv[0] = ProgName;
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
void bfuse(const char* ProgName, string ConfigFilePath, string CompileCommandsPath)
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
    CommonOptionsParser& OptionsParser = ExpectedParser.get();

    /*
     * The Fusion Flow
     *
     * First,  analysis the kernels and extract data to be used
     * Second, rewrite kernels (Parameters, Synchronization, ThreadIdx, BlockIdx, etc...)
     *         this rewriting rule needs to add  "Union Shared Memory Variable" Algorithm
     * Third,  create new fused function and save it into the disk
     * THE END
     */

    FusionContext   Context{Info, KernelYAML};
    FusionTool      Tool{OptionsParser, Context};
    AnalysisContext Analysis;

    // 0. Backup files first
    cout << "Backup files...\n";
    for (auto& S : OptionsParser.getSourcePathList()) {
      utils::backUpFiles(S);
    }

    // 1. Analyze kernel codes to be fused
    cout << "Analyzing parameters...\n";
    if (Tool.analyzeParameters(Analysis)) {
      ERROR_MESSAGE("error occur while analyzing parameters");
      exit(0);
    }
    cout << "Analyzing thread boundaries...\n";
    if (Tool.analyzeThreadBoundaries(Analysis)) {
      ERROR_MESSAGE("error occur while analyzing thread boundaries");
      exit(0);
    }

    // 2. Rename and rewrite kernel codes and write it back
    cout << "Renaming parameters...\n";
    if (Tool.renameParameters(Analysis)) {
      ERROR_MESSAGE("error occur while renaming");
      exit(0);
    }
    cout << "Rewriting cuda informations...\n";
    if (Tool.rewriteCUDAInfos(Analysis)) {
      ERROR_MESSAGE("error occur while rewriting");
      exit(0);
    }

    // 3. Create new fused function
    string FuncStr;
    cout << "Building new fused function...\n";
    if (Tool.createFunction(Analysis, FuncStr)) {
      ERROR_MESSAGE("error occur while creating new function");
      exit(0);
    }

    // 4. Write it back to file
    cout << "Save new fused function...\n";
    if (Tool.saveFunction(Analysis, FuncStr)) {
      ERROR_MESSAGE("error occur while saving new function");
      exit(0);
    }

    // 5. Recover files
    cout << "Recover files...\n";
    for (auto &S : OptionsParser.getSourcePathList()) {
      utils::recoverFiles(S);
    }
  }
}
//---------------------------------------------------------------------------
} // namespace bfuse
//---------------------------------------------------------------------------