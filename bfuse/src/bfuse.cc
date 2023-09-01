
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

#include "bfuse.h"
#include "utils.h"
#include "tools.h"

using namespace std;
using namespace clang::tooling;
using namespace bfuse::tools;
using namespace bfuse::utils;
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
  // Extract information from yaml files
  auto FusionYAML = readYAMLInfo<vector<FusionInfo>>(FusionInfoPath);
  auto KernelYAML = readYAMLInfo<map<string, KernelInfo>>(KernelInfoPath);

  // Run block-level fusion
  for (auto& info : FusionYAML) {
    string NewPath {BasePath + "/" + info.filePath};
    Arguments Args {ProgName, BasePath};

    // Create clang parser
    auto [argc, argv]   = Args.getArguments();
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, FusionToolCategory);
    if (!ExpectedParser) {
      llvm::errs() << ExpectedParser.takeError();
      exit(0);
    }
    CommonOptionsParser& OptionsParser = ExpectedParser.get();

    // Do fuse using clang tools
  }

  // tests
  printFusionYAML(FusionYAML);
  printKernelYAML(KernelYAML);
}
//---------------------------------------------------------------------------
} // namespace bfuse
//---------------------------------------------------------------------------