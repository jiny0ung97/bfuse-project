
#include <cstdlib>
#include <iostream>

#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

#include "bfuse/Bfuse.h"
#include "bfuse/Contexts.h"
#include "bfuse/Tools.h"

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
// void FusionTool::print(contexts::FusionContext& Context) const
// {
//   auto Matcher = functionDecl(hasAttr(attr::CUDAGlobal),
//                               hasName(Context.kernels[0])).bind("print-FusionTool-example");

//   cout << "\n================= FusionTools =================\n";
//   cout << "Size of total AST : " << aSTs.size() << "\n\n";
//   for (auto& AST : aSTs) {
//     auto MatchRes = match(Matcher, AST->getASTContext());
//     if (MatchRes.size() < 1)
//       continue;

//     auto *Result = MatchRes[0].getNodeAs<FunctionDecl>("print-FusionTool-example");
//     assert(Result);

//     Result->dump();
//   }
// }
//---------------------------------------------------------------------------
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------