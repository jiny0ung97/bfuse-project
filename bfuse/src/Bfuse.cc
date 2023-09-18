
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "clang/Tooling/CommonOptionsParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "bfuse/Contexts.h"
#include "bfuse/Instances.h"
#include "bfuse/Matchers.h"
#include "bfuse/Utils.h"
#include "bfuse/Bfuse.h"

using namespace std;

using namespace clang::tooling;

using namespace bfuse::contexts;
using namespace bfuse::tools;
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
}
//---------------------------------------------------------------------------
OptionsParserArguments::~OptionsParserArguments() { free(argv); }
//---------------------------------------------------------------------------
void OptionsParserArguments::print() const
{
  cout << "================= OptionsParserArguments =================\n";
  cout << "argc: " << argc << "\n";
  cout << "argv: "
  for (int i = 0; i < argc; ++i) {
    cout << argv[i] << " ";
  }
  cout << "\n";
}
//---------------------------------------------------------------------------
void bfuse(const char *ProgName, string ConfigFilePath, string CompileCommandsPath)
{
  string FusionInfoPath = ConfigFilePath + "/fusions.yaml";
  string KernelInfoPath = ConfigFilePath + "/kernels.yaml";

  // Extract information from yaml files
  auto FusionYAML = utils::readYAMLInfo<vector<FusionInfo>>(FusionInfoPath);
  auto KernelYAML = utils::readYAMLInfo<map<string, KernelInfo>>(KernelInfoPath);

  // Run block-level fusion
  for (auto& Info : FusionYAML) {
    if (!checkFusionValid(Info, KernelYAML)) {
      ERROR_MESSAGE("invalid fusion definition exist.");
      exit(0);
    }

    // Create compilation database
    string CodePath = CompileCommandsPath + "/" + extractFilePath(Info, KernelYAML);
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
     *         this rewriting rule need to add share "Shared Memory Variable" Algorithm
     * Third,  create new fused function and save it into the disk
     * THE END
     */

    FusionContext     Context{Info, KernelYAML};
    FusionRewriteTool FRTool{OptionsParser, Context};

    // 0. Backup files first
    cout << "Backup files...\n";
    for (auto &S : OptionsParser.getSourcePathList()) {
      utils::backUpFiles(S);
    }

    // 1. Analyze kernel codes to be fused
    AnalysisContext AContext;

    cout << "Analyze CUDA kernels...\n";
    if (FRTool.analyze(AContext)) {
      ERROR_MESSAGE("error occur while analyzing");
      exit(0);
    }

    // [Test]
    // AContext.print();

    // 2. Rewrite kernel codes and write it back
    cout << "Rewrite codes...\n";
    if (FRTool.rewrite(AContext)) {
      ERROR_MESSAGE("error occur while rewriting");
      exit(0);
    }

    // 3. Create new fused function
    FusionBuildTool FBTool;

    cout << "Create new fused function...\n";
    FBTool.createFunctionFromCode();

    // 4. Write it back to file
    string FilePath = "output.cu";

    cout << "Save fused function...\n";
    cout << "File: " << FilePath << "\n";
    FBTool.write(FilePath);
  }
}
//---------------------------------------------------------------------------
} // namespace bfuse
//---------------------------------------------------------------------------