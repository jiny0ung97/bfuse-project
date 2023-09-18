
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "llvm/Support/raw_ostream.h"

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

    /*
     * The Fusion Flow
     *
     * First,  analysis the kernels and extract data to be used
     * Second, rewrite kernels (Parameters, Synchronization, ThreadIdx, BlockIdx, etc...)
     *         this rewriting rule need to add share "Shared Memory Variable" Algorithm
     * Third,  create new fused function and save it into the disk
     * THE END
     */

    FusionRewriteTool FRTool{Args, Context};

    // [Test]
    // if (FRTool.printFunctionDeclExample()) {
    //   ERROR_MESSAGE("error occur while testing tool");
    //   exit(0);
    // }

    // 1. Analyze kernel codes to be fused
    AnalysisContext AContext;

    cout << "Analyze CUDA kernels...\n";
    if (FRTool.analyze(AContext)) {
      ERROR_MESSAGE("error occur while analyzing");
      exit(0);
    }

    // [Test]
    AContext.print();

    // 2. Rewrite kernel codes and write it back
    string Str;
    llvm::raw_string_ostream RawStream{Str};

    cout << "Rewrite codes...\n";
    if (FRTool.rewrite(AContext, RawStream)) {
      ERROR_MESSAGE("error occur while rewriting");
      exit(0);
    }

    // [Test]
    // cout << RawStream.str() << "\n";

    // 3. Create new fused function
    FusionBuildTool FBTool;

    cout << "Create new fused function...\n";
    FBTool.createFunctionFromCode(RawStream);

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