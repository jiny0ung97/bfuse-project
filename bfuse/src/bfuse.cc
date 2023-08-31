
#include <iostream>
#include <string>
#include <vector>
#include <map>

// #include "clang/ASTMatchers/ASTMatchers.h"
// #include "clang/ASTMatchers/ASTMatchFinder.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

#include "bfuse.h"
#include "utils.h"

using namespace std;
using namespace bfuse;
using namespace clang::tooling;
//---------------------------------------------------------------------------
static llvm::cl::OptionCategory FusionToolCategory{"fusion-tool options"};
static llvm::cl::extrahelp      CommonHelp{CommonOptionsParser::HelpMessage};
static llvm::cl::extrahelp      MoreHelp{"\nMore help text...\n"};
//---------------------------------------------------------------------------
namespace bfuse {
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
  int    argc = 2;
  string NewPath;
  vector<KernelInfo> KernelVector;

  argv[0] = ProgName;
  for (auto& info : FusionYAML) {
    NewPath  = BasePath + "/" + info.filePath;
    argv[1]  = NewPath.c_str();

    auto ExpectedParser = CommonOptionsParser::create(argc, argv, FusionToolCategory);
    if (!ExpectedParser) {
      llvm::errs() << ExpectedParser.takeError();
    }

    for (auto &name : info.kernels) {
      KernelVector.push_back(KernelYAML.at(name));
    }

    KernelVector.clear();
  }

  // tests
  utils::printFusionYAML(FusionYAML);
  utils::printKernelYAML(KernelYAML);

  free(argv);
}
//---------------------------------------------------------------------------
} // namespace bfuse
//---------------------------------------------------------------------------