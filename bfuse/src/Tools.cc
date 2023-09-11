
#include <cstdlib>
#include <iostream>

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

#include "bfuse/Bfuse.h"
#include "bfuse/Tools.h"

using namespace std;
using namespace clang::tooling;
//---------------------------------------------------------------------------
static llvm::cl::OptionCategory MyToolCategory{"my-tool options"};
static llvm::cl::extrahelp      CommonHelp{CommonOptionsParser::HelpMessage};
static llvm::cl::extrahelp      MoreHelp{"\nMore help text...\n"};
//---------------------------------------------------------------------------
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
FusionTool::FusionTool(const Arguments& Arg) {
  auto [argc, argv] = Arg.getArguments();
  auto ExpectParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
  if (!ExpectParser) {
    llvm::errs() << ExpectParser.takeError();
    exit(0);
  }

  CommonOptionsParser& OptionsParser = ExpectParser.get();
  ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

  Tool.buildASTs(aSTs);
}
//---------------------------------------------------------------------------
void FusionTool::print() const
{
  cout << "\n================= FusionTools =================\n";
  for (auto& AST : aSTs) {
    auto* TU = AST->getASTContext().getTranslationUnitDecl();
    TU->dump();
    cout << "\n";
  }
}
//---------------------------------------------------------------------------
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------