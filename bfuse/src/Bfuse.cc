
#include <cstdlib>
#include <string>
#include <vector>
#include <map>

// #include "clang/ASTMatchers/ASTMatchers.h"
// #include "clang/ASTMatchers/ASTMatchFinder.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

#include "bfuse/Bfuse.h"
#include "bfuse/Utils.h"
#include "bfuse/Tools.h"

using namespace std;
using namespace clang::tooling;
using namespace bfuse::tools;
using namespace bfuse::utils;
//---------------------------------------------------------------------------
namespace bfuse {
//---------------------------------------------------------------------------
void bfuse(const char *ProgName, string FusionInfoPath, string KernelInfoPath, string BasePath)
{
  // Extract information from yaml files
  auto FusionYAML = readYAMLInfo<vector<FusionInfo>>(FusionInfoPath);
  auto KernelYAML = readYAMLInfo<map<string, KernelInfo>>(KernelInfoPath);

  // Run block-level fusion
  for (auto& Info : FusionYAML) {
    // Create fusion tools object
    FusionTools Tools{Info, KernelYAML};

    // [Tests]
    printFusionTools(Tools, Info);
  }
}
//---------------------------------------------------------------------------
} // namespace bfuse
//---------------------------------------------------------------------------