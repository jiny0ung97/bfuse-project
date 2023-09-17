
#include <cstdlib>
#include <memory>

#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "bfuse/Contexts.h"
#include "bfuse/Matchers.h"
#include "bfuse/Instances.h"

using namespace std;

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
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
int FusionRewriteTool::analyze(AnalyzeContext &Analysis)
{
  // Create compilation database
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

  // Add AST matchers


  return 0;
}
//---------------------------------------------------------------------------
int FusionRewriteTool::rewrite(AnalyzeContext &Analysis, llvm::raw_ostream &RawOstream)
{
  // Create compilation database
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

  // Add AST matchers
  CUDAFuncParamRewriter ParamRewriter;
  CUDABlockIdxRewriter  BlockIdxRewriter;
  CUDASyncRewriter      SyncRewriter;
  MatchFinder Finder;

  for (auto &KName : Context.kernels) {
    Finder.addMatcher(ParamRewriter.getFuncParamMatcher(KName),
                      &ParamRewriter);
    Finder.addMatcher(BlockIdxRewriter.getBlockIdxMatcher(KName),
                      &BlockIdxRewriter);
    Finder.addMatcher(SyncRewriter.getSyncMatcher(KName),
                      &SyncRewriter);
  }


  // TODO:
  // Rewriter Writer;

  // auto& SM = Writer.getSourceMgr();
  // auto& WB = Writer.getEditBuffer(SM.getMainFileID());
  // WB.write(RawOstream);

  return Tool.run(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionRewriteTool::printFunctionDeclExample() const
{
  // Create compilation database
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

  // Add AST matchers
  CUDAFuncDeclPrinter Printer;
  MatchFinder Finder;

  for (auto &KName : Context.kernels) {
    auto Matcher = Printer.getFuncDeclMatcher(KName);
    Finder.addMatcher(Matcher, &Printer);
  }

  return Tool.run(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionBuildTool::createFunctionFromCode(llvm::raw_string_ostream &RawString)
{
  // TODO:
  // unique_ptr<ASTUnit> Unit = buildASTFromCode(RawString);

  // auto &C  = Unit->getASTContext();
  // auto *TU = C.getTranslationUnitDecl();

  return 0;
}
//---------------------------------------------------------------------------
int FusionBuildTool::write(string &FilePath)
{
  // TODO:

  return 0;
}
//---------------------------------------------------------------------------
} // tools
} // bfuse
//---------------------------------------------------------------------------