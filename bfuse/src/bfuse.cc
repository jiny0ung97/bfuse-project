
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

// #include "clang/ASTMatchers/ASTMatchers.h"
// #include "clang/ASTMatchers/ASTMatchFinder.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

// #include "llvm/Support/CommandLine.h"

#include "bfuse.h"
#include "utils.h"

using namespace std;
using namespace bfuse;
using namespace clang::tooling;
//---------------------------------------------------------------------------
static llvm::cl::OptionCategory FusionToolCategory("fusion-tool options");
//---------------------------------------------------------------------------
namespace bfuse {

string KernelInfo::getStringInfo()
{
  stringstream text;

  text << "\n========= Kernel Info =========\n";
  text << "- Name: " << kernelName << "\n";
  text << "- Barriers: " << hasBarriers << "\n";
  text << "- GridDim: " << gridDim.size() << "\n";
  text << "- BlockDim: " << blockDim.size() << "\n";
  text << "- Registers: " << reg << "\n";
  text << "- ExecTime: " << execTime << "\n";

  return text.str();
}
//---------------------------------------------------------------------------
string FusionInfo::getStringInfo()
{
  stringstream text;

  text << "\n========= Fusion Info =========\n";
  text << "- File: " << filePath << "\n";
  text << "- Kernels:\n";
  for (auto& kernel : kernels) {
    text << "+- " << kernel << "\n";
  }

  return text.str();
}
//---------------------------------------------------------------------------
void bfuse(const char *ProgName, std::string FusionInfoPath,
           std::string KernelInfoPath, std::string BasePath)
{
  auto FusionYAML =
      utils::readYAMLInfo<vector<FusionInfo>>(FusionInfoPath);
  auto KernelYAML =
      utils::readYAMLInfo<map<string, KernelInfo>>(KernelInfoPath);

  // Fuse kernels in horizontal.
  // Fusion is done by block-level.
  const char **argv = (const char**)malloc(sizeof(char *) * 2);
  int argc = 2;
  string NewPath;

  argv[0] = ProgName;

  for (auto& info : FusionYAML) {
    NewPath  = BasePath + "/" + info.filePath;
    argv[1]  = NewPath.c_str();

    // TODO:
    // auto ExpectedParser = CommonOptionsParser OptionsParser(argc, argv, FusionToolCategory);

    vector<KernelInfo> KernelVector;
    for (auto &name : info.kernels) {
      KernelVector.push_back(KernelYAML.at(name));
    }
  }

  free(argv);
}

} // namespace bfuse
//---------------------------------------------------------------------------