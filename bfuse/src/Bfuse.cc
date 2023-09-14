
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "bfuse/Contexts.h"
#include "bfuse/Tools.h"
#include "bfuse/Utils.h"
#include "bfuse/Bfuse.h"

using namespace std;
using namespace bfuse::contexts;
using namespace bfuse::tools;
using namespace bfuse::utils;
//---------------------------------------------------------------------------
namespace bfuse {
//---------------------------------------------------------------------------
CommonParsersArguments::CommonParsersArguments(const char *ProgName,
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
CommonParsersArguments::~CommonParsersArguments() { free(argv); }
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

    CommonParsersArguments Args{ProgName, CompileCommandsPath, CodePath};
    FusionTool             Tool{Args};
    FusionContext          Context{Info, KernelYAML};

    // [Tests]
    Context.print();
    Tool.print(Info.kernels.front());
  }
}
//---------------------------------------------------------------------------
} // namespace bfuse
//---------------------------------------------------------------------------