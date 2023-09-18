
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <string>
#include <map>

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/LangOptions.h"

#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Refactoring/Rename/RenamingAction.h"

#include "clang/Rewrite/Core/Rewriter.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "bfuse/Contexts.h"
#include "bfuse/Matchers.h"
#include "bfuse/Instances.h"
#include "bfuse/Utils.h"

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
int FusionRewriteTool::analyze(AnalysisContext &Analysis)
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
  MatchFinder Finder;
  CUDAFuncParamAnalyzer ParamAnalyzer;

  for (auto &KName : Context.kernels) {
    Finder.addMatcher(ParamAnalyzer.getFuncParamMatcher(KName),
                      &ParamAnalyzer);
  }

  int Err = Tool.run(newFrontendActionFactory(&Finder).get());
  if (Err) {
    return Err;
  }

  Analysis.kernels      = Context.kernels;
  Analysis.ParamListMap = ParamAnalyzer.ParamListMap;
  Analysis.USRsListMap  = ParamAnalyzer.USRsListMap;

  return Err;
}
//---------------------------------------------------------------------------
int FusionRewriteTool::rewrite(AnalysisContext &Analysis, llvm::raw_ostream &RawOstream)
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
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // TODO:
  // Renaming function parameters
  vector<string>         NewParams;
  vector<string>         PrevParams;
  vector<vector<string>> USRs;

  for (auto &KName : Analysis.kernels) {
    auto &PrevParamList = Analysis.ParamListMap.at(KName);
    auto &USRsList      = Analysis.USRsListMap.at(KName);

    vector<string> NewParamList{PrevParamList.size()};
    transform(PrevParamList.begin(), PrevParamList.end(),
              NewParamList.begin(),
              [&KName](string &PName) {
                return "__" + KName + "_" + PName;
              });

    NewParams.insert(NewParams.end(),
                     NewParamList.begin(), NewParamList.end());
    PrevParams.insert(PrevParams.end(),
                      PrevParamList.begin(), PrevParamList.end());
    USRs.insert(USRs.end(),
                USRsList.begin(), USRsList.end());
  }

  for (auto &S : Tool.getSourcePaths()) {
    utils::backUpFiles(S);
  }

  RenamingAction Renaming{NewParams, PrevParams,
                          USRs, Tool.getReplacements()};

  int Err = Tool.runAndSave(newFrontendActionFactory(&Renaming).get());
  if (Err)
    return Err;

  // Add AST matchers
  // MatchFinder Finder;
  // CUDABlockIdxRewriter BlockIdxRewriter{Tool.getReplacements()};
  // CUDASyncRewriter     SyncRewriter{Tool.getReplacements()};

  // for (auto &KName : Context.kernels) {
  //   Finder.addMatcher(BlockIdxRewriter.getBlockIdxMatcher(KName),
  //                     &BlockIdxRewriter);
  //   Finder.addMatcher(SyncRewriter.getSyncMatcher(KName),
  //                     &SyncRewriter);
  // }

  // Err = Tool.run(newFrontendActionFactory(&Finder).get());
  // if (Err)
  //   return Err;

  // TODO:
  // IntrusiveRefCntPtr<DiagnosticIDs> DiagIDs      = new DiagnosticIDs();
  // IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();

  // // TextDiagnosticPrinter DiagPrinter{llvm::errs(), DiagOpts.get()};
  // DiagnosticsEngine DiagEngine{IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()),
  //                              &*DiagOpts,
  //                              new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts)};

  // SourceManager SourceMgr{DiagEngine, Tool.getFiles()};
  // LangOptions   LangOpts;
  // Rewriter      Writer{SourceMgr, LangOpts};

  // if (!Tool.applyAllReplacements(Writer)) {
  //   return 0;
  // }
  // auto &WriteBuffer = Writer.getEditBuffer(SourceMgr.getMainFileID());
  // WriteBuffer.write(RawOstream);

  return Err;
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