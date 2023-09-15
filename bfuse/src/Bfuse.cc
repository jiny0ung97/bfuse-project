
#include <cstdlib>
// #include <iostream>
#include <string>
#include <vector>
#include <map>

#include "bfuse/Contexts.h"
#include "bfuse/Instances.h"
#include "bfuse/Matchers.h"
#include "bfuse/Utils.h"
#include "bfuse/Bfuse.h"

using namespace std;

using namespace bfuse::contexts;
using namespace bfuse::tools;
using namespace bfuse::matchers;
using namespace bfuse::utils;
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

    // 1. Build Fusion Instance
    FusionInstance FI{Args, Context};

    // [Test]
    if (FI.printFunctionDeclExample()) {
      ERROR_MESSAGE("error occur while running tool");
      exit(0);
    }

    // 2. Build my matcher (analysis) & Run

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