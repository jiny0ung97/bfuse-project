
#include <utility>
#include <algorithm>
#include <numeric>
#include <string>
#include <map>

#include "clang/Frontend/FrontendActions.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Refactoring/Rename/RenamingAction.h"

#include "fuse/Tools.h"
#include "fuse/Contexts.h"
#include "fuse/Matchers.h"
#include "fuse/Utils.h"

using namespace std;

using namespace clang::tooling;
using namespace clang::ast_matchers;

using namespace fuse::contexts;
using namespace fuse::matchers;
//---------------------------------------------------------------------------
namespace fuse {
namespace tools {
//---------------------------------------------------------------------------
int FusionTool::initiallyRewriteKernels()
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser_.getCompilations(),
                       OptionsParser_.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDAKernelRewriter Rewriter{Tool.getReplacements()};

  for (auto &KName : FContext_.Kernels_) {
    Finder.addMatcher(ASTPatternMatcher::getFuncDeclMatcher(KName), &Rewriter);
  }
  return Tool.runAndSave(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionTool::rewriteCompStmt()
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser_.getCompilations(),
                       OptionsParser_.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDACompStmtRewriter Rewriter{Tool.getReplacements()};

  for (auto &KName : FContext_.Kernels_) {
    Finder.addMatcher(ASTPatternMatcher::getFuncDeclMatcher(KName), &Rewriter);
  }
  return Tool.runAndSave(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionTool::renameParameters()
{
  // Clang Tool
  ClangTool Tool(OptionsParser_.getCompilations(),
                 OptionsParser_.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDAFuncParmAnalyzer Analyzer;

  for (auto &KName : FContext_.Kernels_) {
    Finder.addMatcher(ASTPatternMatcher::getFuncDeclMatcher(KName), &Analyzer);
  }

  int Err = Tool.run(newFrontendActionFactory(&Finder).get());
  if (Err) {
    return Err;
  }

  // Refactoring Tool
  RefactoringTool ReTool(OptionsParser_.getCompilations(),
                         OptionsParser_.getSourcePathList());

  // Collect parameters' information and return renamed parameter lists
  vector<string> NewParams;
  vector<string> PrevParams;
  vector<vector<string>> USRs;

  for (auto &KName : FContext_.Kernels_) {
    if (Analyzer.ParmListMap_.find(KName) == Analyzer.ParmListMap_.end()) {
      continue;
    }
    
    auto &PrevParmList = Analyzer.ParmListMap_.at(KName);
    auto &USRsList     = Analyzer.ParmUSRsListMap_.at(KName);

    vector<string> NewParmList{PrevParmList.size()};
    transform(PrevParmList.begin(), PrevParmList.end(),
              NewParmList.begin(),
              [&KName](const string &PName) {
                return KName + "_" + PName + "_";
              });

    NewParams.insert(NewParams.end(),
                     NewParmList.begin(), NewParmList.end());
    PrevParams.insert(PrevParams.end(),
                      PrevParmList.begin(), PrevParmList.end());
    USRs.insert(USRs.end(),
                USRsList.begin(), USRsList.end());
  }

  // Run renaming frontend action
  RenamingAction Renaming{NewParams, PrevParams,
                          USRs, ReTool.getReplacements()};

  return ReTool.runAndSave(newFrontendActionFactory(&Renaming).get());
}
//---------------------------------------------------------------------------
int FusionTool::rewriteCUDAVariables()
{
  map<string, int> ThreadNumMap;
  for (auto &KName : FContext_.Kernels_) {
    auto &KInfo = FContext_.KernelInfoMap_.at(KName);
    ThreadNumMap[KName] = KInfo.BlockDim_.size();
  }

  // Refactoring Tool
  RefactoringTool Tool(OptionsParser_.getCompilations(),
                       OptionsParser_.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDABlockInfoRewriter BInfoRewriter{Tool.getReplacements()};
  // CUDASyncRewriter      SyncRewriter{Tool.getReplacements(), ThreadNumMap};

  for (auto &KName : FContext_.Kernels_) {
    Finder.addMatcher(ASTPatternMatcher::getBlockIdxMatcher(KName),
                      &BInfoRewriter);
    // Finder.addMatcher(ASTPatternMatcher::getSyncMatcher(KName),
    //                   &SyncRewriter);
  }
  return Tool.runAndSave(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionTool::hoistSharedDecls()
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser_.getCompilations(),
                       OptionsParser_.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDASharedDeclExtractor Extractor{Tool.getReplacements()};
  CUDASharedDeclRewriter  Writer{Tool.getReplacements()};

  for (auto &KName : FContext_.Kernels_) {
    Finder.addMatcher(ASTPatternMatcher::getSharedDeclMatcher(KName), &Extractor);
    Finder.addMatcher(ASTPatternMatcher::getSharedDeclMatcher(KName), &Writer);
  }

  return Tool.runAndSave(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionTool::printFuncDecl()
{
  // Clang Tool
  ClangTool Tool(OptionsParser_.getCompilations(),
                 OptionsParser_.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDAFuncDeclPrinter Printer;

  for (auto &KName : FContext_.Kernels_) {
    Finder.addMatcher(ASTPatternMatcher::getFuncDeclMatcher(KName), &Printer);
  }
  return Tool.run(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
} // tools
} // fuse
//---------------------------------------------------------------------------