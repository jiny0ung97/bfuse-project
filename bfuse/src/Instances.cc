
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

#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Refactoring/Rename/RenamingAction.h"

#include "clang/Rewrite/Core/Rewriter.h"

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
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
int FusionRewriteTool::analyze(AnalysisContext &Analysis)
{
  // Clang Tool
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
int FusionRewriteTool::rename(AnalysisContext &Analysis)
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // Collect informations of parameters to be renamed
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
int FusionRewriteTool::rewrite(AnalysisContext &Analysis)
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDABlockInfoRewriter BlockInfoRewriter{Tool.getReplacements()};
  CUDASyncRewriter      SyncRewriter{Tool.getReplacements(),
                                     Context.kernelContextMap};

  for (auto &KName : Context.kernels) {
    Finder.addMatcher(BlockInfoRewriter.getBlockInfoMatcher(KName),
                      &BlockInfoRewriter);
    Finder.addMatcher(SyncRewriter.getSyncMatcher(KName),
                      &SyncRewriter);
  }
  return Tool.runAndSave(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionRewriteTool::printFunctionDeclExample() const
{
  // Clang Tool
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
int FusionBuildTool::createFunctionFromCode()
{
  // TODO:

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