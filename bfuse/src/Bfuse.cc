
#include <cstdlib>
#include <string>
#include <iostream>

#include "bfuse/Contexts.h"
#include "bfuse/Tools.h"
#include "bfuse/Utils.h"
#include "bfuse/Bfuse.h"

using namespace std;
using namespace bfuse::tools;
using namespace bfuse::contexts;
//---------------------------------------------------------------------------
static const string KerenlInfoFilePath = "kernels.yaml";
static const string FusionInfoFilePath = "fusion.yaml";
static const string KernelCodePath = "kernels.cu";
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
  cout << "    Z: " << blockDim.z << "\n";
}
//---------------------------------------------------------------------------
void FusionInfo::print() const
{
  cout << "[FusionInfo]\n";
  cout << "  - Kernels:\n";
  for (auto& KName : kernels) {
    cout << "    - " << KName << "\n";
  }
}
//---------------------------------------------------------------------------
void bfuse(const char *ProgName, string ConfigFilePath, string CompileCommandsPath)
{
  string FusionInfoPath = ConfigFilePath + "/" + FusionInfoFilePath;
  string KernelInfoPath = ConfigFilePath + "/" + KerenlInfoFilePath;
  string CodePath       = CompileCommandsPath + "/" + KernelCodePath;

  // Extract information from yaml files
  auto FusionYAML = utils::readYAMLInfo<vector<FusionInfo>>(FusionInfoPath);
  auto KernelYAML = utils::readYAMLInfo<map<string, KernelInfo>>(KernelInfoPath);

  CommonParsersArguments Args{ProgName, CompileCommandsPath, CodePath};

  // Run block-level fusion
  for (auto& Info : FusionYAML) {
    FusionContext Context{Info, KernelYAML};
    FusionTool    Tool{Args};

    // [Tests]
    Context.print();
    // Tool.print();

  }
}
//---------------------------------------------------------------------------
} // namespace bfuse
//---------------------------------------------------------------------------