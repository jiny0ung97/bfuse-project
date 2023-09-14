
#include <cstdlib>
#include <iostream>

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

#include "bfuse/Bfuse.h"
#include "bfuse/Contexts.h"
#include "bfuse/Database.h"

using namespace std;
using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;
//---------------------------------------------------------------------------
static llvm::cl::OptionCategory MyToolCategory{"my-tool options"};
static llvm::cl::extrahelp      CommonHelp{CommonOptionsParser::HelpMessage};
static llvm::cl::extrahelp      MoreHelp{"\nMore help text...\n"};
//---------------------------------------------------------------------------
namespace bfuse {
namespace database {
//---------------------------------------------------------------------------
FusionDatabase::FusionDatabase(const OptionsParserArguments& Arg) {
  auto [argc, argv] = Arg.getArguments();
  auto ExpectParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
  if (!ExpectParser) {
    llvm::errs() << ExpectParser.takeError();
    exit(0);
  }

  CommonOptionsParser& OptionsParser = ExpectParser.get();
  ClangTool Tool(OptionsParser.getCompilations(), OptionsParser.getSourcePathList());

  Tool.buildASTs(ASTs);

  // 1. Init matchers for each functions
}
//---------------------------------------------------------------------------
void FusionDatabase::print() const
{
  cout << "================= FusionTools =================\n";
  cout << "Size of total AST : " << ASTs.size() << "\n\n";
  // for (auto& AST : ASTs) {
  //   auto *TU = AST->getASTContext().getTranslationUnitDecl();
  //   TU->dump();
  //   cout << "\n";
  // }
  cout << "\n";
}
//---------------------------------------------------------------------------
void FusionDatabase::print(string& KName) const
{
  cout << "================= FusionTools =================\n";
  cout << "Kernel Name : " << KName << "\n\n";
  
  auto MB = (functionDecl(hasAttr(attr::CUDAGlobal), hasName(KName))).bind("bindStr");
  for (auto& AST : ASTs) {
    auto MatchRes = match(MB, AST->getASTContext());
    assert(MatchRes.size() >= 1);

    auto Result = MatchRes[0].getNodeAs<FunctionDecl>("bindStr");
    Result->dump();
  }
  cout << "\n";
}
//---------------------------------------------------------------------------
} // namespace database
} // namespace bfuse
//---------------------------------------------------------------------------