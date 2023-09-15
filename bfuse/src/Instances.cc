
#include <cstdlib>

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "bfuse/Contexts.h"
#include "bfuse/Matchers.h"
#include "bfuse/Instances.h"

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

using namespace bfuse::contexts;
using namespace bfuse::matchers;
//---------------------------------------------------------------------------
// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("my-tool options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static llvm::cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static llvm::cl::extrahelp MoreHelp("\nMore help text...\n");
//---------------------------------------------------------------------------
static DeclarationMatcher FunctionDeclMatcher
        = functionDecl(
            hasAttr(attr::CUDAGlobal)
          ).bind(CUDAFunctionDeclBindId);
//---------------------------------------------------------------------------
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
int FusionInstance::analyze(/*maybe need AnalyzeContext?*/)
{
  return 0;
}
//---------------------------------------------------------------------------
int FusionInstance::rewrite(llvm::raw_ostream &RawOstream /*, maybe need AnalyzeContext?*/)
{
  // TODO: do something...

  Rewriter Writer;

  auto& SM = Writer.getSourceMgr();
  auto& WB = Writer.getEditBuffer(SM.getMainFileID());
  WB.write(RawOstream);

  return 0;
}
//---------------------------------------------------------------------------
int FusionInstance::printFunctionDeclExample() const
{
  auto [argc, argv]   = Args.getArguments();
  auto ExpectedParser = CommonOptionsParser::create(argc, argv, MyToolCategory);
  if (!ExpectedParser) {
    // Fail gracefully for unsupported options
    llvm::errs() << ExpectedParser.takeError();
    exit(0);
  }
  CommonOptionsParser& OptionsParser = ExpectedParser.get();
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  CUDAFunctionDeclPrinter Printer;
  MatchFinder Finder;
  Finder.addMatcher(FunctionDeclMatcher, &Printer);

  return Tool.run(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
} // tools
} // bfuse
//---------------------------------------------------------------------------