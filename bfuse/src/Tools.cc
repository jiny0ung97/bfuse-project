
#include <algorithm>
#include <numeric>
#include <string>
#include <map>

#include "clang/Frontend/FrontendActions.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Refactoring/Rename/RenamingAction.h"

#include "llvm/Support/raw_ostream.h"

#include "bfuse/Contexts.h"
#include "bfuse/Matchers.h"
#include "bfuse/Tools.h"
#include "bfuse/Utils.h"

using namespace std;

using namespace clang::tooling;
using namespace clang::ast_matchers;

using namespace bfuse::contexts;
using namespace bfuse::matchers;
//---------------------------------------------------------------------------
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
int FusionTool::initiallyRewriteKernels(AnalysisContext &AContext)
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDADeclExtractor Extractor{Tool.getReplacements()};

  for (auto &KName : AContext.Kernels) {
    Finder.addMatcher(Extractor.getFuncDeclMatcher(KName), &Extractor);
  }
  return Tool.runAndSave(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionTool::analyzeParameters(AnalysisContext &AContext)
{
  // Clang Tool
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDAFuncParamAnalyzer ParamAnalyzer;

  for (auto &KName : AContext.Kernels) {
    Finder.addMatcher(ParamAnalyzer.getFuncParamMatcher(KName),
                      &ParamAnalyzer);
  }

  int Err = Tool.run(newFrontendActionFactory(&Finder).get());
  if (Err) {
    return Err;
  }

  AContext.ParamListMap = ParamAnalyzer.ParamListMap;
  AContext.USRsListMap  = ParamAnalyzer.USRsListMap;
  return Err;
}
//---------------------------------------------------------------------------
int FusionTool::renameParameters(AnalysisContext &AContext)
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // Collect informations of parameters to be renamed
  vector<string>         NewParams;
  vector<string>         PrevParams;
  vector<vector<string>> USRs;

  for (auto &KName : AContext.Kernels) {
    auto &PrevParamList = AContext.ParamListMap.at(KName);
    auto &USRsList      = AContext.USRsListMap.at(KName);

    vector<string> NewParamList{PrevParamList.size()};
    transform(PrevParamList.begin(), PrevParamList.end(),
              NewParamList.begin(),
              [&KName](string &PName) {
                return KName + "_" + PName + "_";
              });

    NewParams.insert(NewParams.end(),
                     NewParamList.begin(), NewParamList.end());
    PrevParams.insert(PrevParams.end(),
                      PrevParamList.begin(), PrevParamList.end());
    USRs.insert(USRs.end(),
                USRsList.begin(), USRsList.end());
  }

  // Run renaming frontend action
  RenamingAction Renaming{NewParams, PrevParams,
                          USRs, Tool.getReplacements()};

  return Tool.runAndSave(newFrontendActionFactory(&Renaming).get());
}
//---------------------------------------------------------------------------
int FusionTool::rewriteCUDAVariables(AnalysisContext &AContext)
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDABlockInfoRewriter BlockInfoRewriter{Tool.getReplacements(), AContext.TmpBlockInfoString};
  CUDASyncRewriter      SyncRewriter{Tool.getReplacements(), AContext.ThreadNumMap};

  for (auto &KName : AContext.Kernels) {
    Finder.addMatcher(BlockInfoRewriter.getBlockInfoMatcher(KName),
                      &BlockInfoRewriter);
    Finder.addMatcher(SyncRewriter.getSyncMatcher(KName),
                      &SyncRewriter);
  }
  return Tool.runAndSave(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionTool::createFusedKernel(AnalysisContext &AContext)
{
  // Clang Tool
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDAFuncBuilder Builder{AContext, FuncStr};

  for (auto &KName : AContext.Kernels) {
    auto Matcher = Builder.getFuncBuildMatcher(KName);
    Finder.addMatcher(Matcher, &Builder);
  }
  return Tool.run(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionTool::saveFusedKernel(AnalysisContext &AContext, const string &ResultPath)
{
  string FilePath = ResultPath + "/" + AContext.NewFuncName + ".cu";
  std::error_code EC;
  llvm::raw_fd_ostream FdStream{FilePath, EC};

  FdStream << FuncStr;
  FdStream.close();

  return 0;
}
//---------------------------------------------------------------------------
int FusionTool::printFuncDecl(AnalysisContext &AContext)
{
  // Clang Tool
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDAFuncDeclPrinter Printer;

  for (auto &KName : AContext.Kernels) {
    auto Matcher = Printer.getFuncDeclMatcher(KName);
    Finder.addMatcher(Matcher, &Printer);
  }
  return Tool.run(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
} // tools
} // bfuse
//---------------------------------------------------------------------------