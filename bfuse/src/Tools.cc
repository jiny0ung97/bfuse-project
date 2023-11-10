
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

#include "llvm/Support/raw_ostream.h"

#include "bfuse/Contexts.h"
#include "bfuse/Algorithms.h"
#include "bfuse/Matchers.h"
#include "bfuse/Utils.h"
#include "bfuse/Tools.h"

using namespace std;

using namespace clang::tooling;
using namespace clang::ast_matchers;

using namespace bfuse::contexts;
using namespace bfuse::matchers;
//---------------------------------------------------------------------------
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
int FusionTool::initiallyRewriteKernels(const AnalysisContext &AContext)
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDAKernelRewriter Rewriter{Tool.getReplacements()};

  for (auto &KName : AContext.Kernels) {
    Finder.addMatcher(ASTPatternMatcher::getFuncDeclMatcher(KName), &Rewriter);
  }
  return Tool.runAndSave(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionTool::rewriteCompStmt(const contexts::AnalysisContext &AContext)
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDACompStmtRewriter Rewriter{Tool.getReplacements(), AContext.TmpBlockInfoString};

  for (auto &KName : AContext.Kernels) {
    Finder.addMatcher(ASTPatternMatcher::getFuncDeclMatcher(KName), &Rewriter);
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
  CUDAFuncParmAnalyzer Analyzer;

  for (auto &KName : AContext.Kernels) {
    Finder.addMatcher(ASTPatternMatcher::getFuncDeclMatcher(KName), &Analyzer);
  }

  int Err = Tool.run(newFrontendActionFactory(&Finder).get());
  if (Err) {
    return Err;
  }

  AContext.ParmListMap      = move(Analyzer.ParmListMap);
  AContext.ParmUSRsListMap  = move(Analyzer.ParmUSRsListMap);
  return Err;
}
//---------------------------------------------------------------------------
int FusionTool::renameParameters(const AnalysisContext &AContext)
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // Collect parameters' information and return renamed parameter lists
  auto [NewParams, PrevParams, USRs] = algorithms::getNewParmLists(AContext);

  // Run renaming frontend action
  RenamingAction Renaming{NewParams, PrevParams,
                          USRs, Tool.getReplacements()};

  return Tool.runAndSave(newFrontendActionFactory(&Renaming).get());
}
//---------------------------------------------------------------------------
int FusionTool::rewriteCUDAVariables(const AnalysisContext &AContext)
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDABlockInfoRewriter BInfoRewriter{Tool.getReplacements()};
  CUDASyncRewriter      SyncRewriter{Tool.getReplacements(), AContext.ThreadNumMap};

  for (auto &KName : AContext.Kernels) {
    Finder.addMatcher(ASTPatternMatcher::getBlockIdxMatcher(KName),
                      &BInfoRewriter);
    Finder.addMatcher(ASTPatternMatcher::getSyncMatcher(KName),
                      &SyncRewriter);
  }
  return Tool.runAndSave(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionTool::hoistSharedDecls(AnalysisContext &AContext)
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDASharedDeclExtractor Extractor{Tool.getReplacements()};
  CUDASharedDeclRewriter  Writer{Tool.getReplacements()};

  for (auto &KName : AContext.Kernels) {
    Finder.addMatcher(ASTPatternMatcher::getSharedDeclMatcher(KName), &Extractor);
    Finder.addMatcher(ASTPatternMatcher::getSharedDeclMatcher(KName), &Writer);
  }

  return Tool.runAndSave(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionTool::analyzeSharedVariables(AnalysisContext &AContext)
{
  // Clang Tool
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDASharedVarAnalyzer Analyzer;

  for (auto &KName : AContext.Kernels) {
    Finder.addMatcher(ASTPatternMatcher::getSharedDeclMatcher(KName), &Analyzer);
  }

  auto Err = Tool.run(newFrontendActionFactory(&Finder).get());
  if (Err) {
    return Err;
  }

  auto [ShrdVarListMap, ShrdVarUSRsListMap, ShrdVarSizeListMap, ShrdDeclStr]
       = algorithms::getShrdVarAnalysis(AContext, Analyzer.ShrdVarListMap,
                                        Analyzer.ShrdVarUSRsListMap, Analyzer.ShrdVarSizeListMap);

  AContext.ShrdVarListMap     = move(ShrdVarListMap);
  AContext.ShrdVarUSRsListMap = move(ShrdVarUSRsListMap);
  AContext.ShrdVarSizeListMap = move(ShrdVarSizeListMap);
  AContext.NewShrdDeclString  = move(ShrdDeclStr);

  return Err;
}
//---------------------------------------------------------------------------
int FusionTool::renameSharedVariables(const AnalysisContext &AContext)
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // Generate new names
  auto [NewShrdVars, PrevShrdVars, USRs] = algorithms::getNewShrdVarLists(AContext);

  // Run renaming frontend action
  RenamingAction Renaming{NewShrdVars, PrevShrdVars,
                          USRs, Tool.getReplacements()};

  return Tool.runAndSave(newFrontendActionFactory(&Renaming).get());
}
//---------------------------------------------------------------------------
int FusionTool::createFusedKernel(const AnalysisContext &AContext)
{
  // Clang Tool
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDAFuncBuilder Builder{AContext, FuncStr};

  for (auto &KName : AContext.Kernels) {
    Finder.addMatcher(ASTPatternMatcher::getFuncDeclMatcher(KName), &Builder);
  }
  return Tool.run(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionTool::saveFusedKernel(const AnalysisContext &AContext, const string &ResultPath)
{
  string FilePath = ResultPath + "/" + AContext.NewFuncName + ".cu";
  std::error_code EC;
  llvm::raw_fd_ostream FdStream{FilePath, EC};

  FdStream << FuncStr;
  FdStream.close();

  return 0;
}
//---------------------------------------------------------------------------
int FusionTool::printFuncDecl(const AnalysisContext &AContext)
{
  // Clang Tool
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDAFuncDeclPrinter Printer;

  for (auto &KName : AContext.Kernels) {
    Finder.addMatcher(ASTPatternMatcher::getFuncDeclMatcher(KName), &Printer);
  }
  return Tool.run(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
} // tools
} // bfuse
//---------------------------------------------------------------------------