
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
static llvm::cl::OptionCategory FusionToolCategory{"fusion-tool options"};
static llvm::cl::extrahelp      CommonHelp{CommonOptionsParser::HelpMessage};
static llvm::cl::extrahelp      MoreHelp{"\nMore help text...\n"};
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
    string NewPath {BasePath + "/" + Info.filePath};
    Arguments Args {ProgName, NewPath};

    // Create clang parser object
    auto [argc, argv]   = Args.getArguments();
    auto ExpectedParser = CommonOptionsParser::create(argc, argv, FusionToolCategory);
    if (!ExpectedParser) {
      llvm::errs() << ExpectedParser.takeError();
      exit(0);
    }
    CommonOptionsParser& OptionsParser = ExpectedParser.get();

    // Create fusion tools object
    auto Tools = FusionTools::create(Info, KernelYAML);

    // [Tests]
    // printKernelContexts(Tools.getKernelContexts());

    // Do fuse using clang tools
    Tools.expandMacros(OptionsParser);
    Tools.renameParameters(OptionsParser);
    Tools.rewriteThreadInfo(OptionsParser);
    Tools.barrierRewriter(OptionsParser);
  }

  // [Tests]
  // printFusionYAML(FusionYAML);
  // printKernelYAML(KernelYAML);
}
//---------------------------------------------------------------------------
} // namespace bfuse
//---------------------------------------------------------------------------