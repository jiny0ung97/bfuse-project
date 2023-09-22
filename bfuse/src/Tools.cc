
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
int FusionTool::initiallyRewriteKernels(const AnalysisContext &AContext)
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDADeclRewriter Rewriter{Tool.getReplacements()};

  for (auto &KName : AContext.Kernels) {
    Finder.addMatcher(Rewriter.getFuncDeclMatcher(KName), &Rewriter);
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

  AContext.ParmListMap      = ParamAnalyzer.ParmListMap;
  AContext.ParmUSRsListMap  = ParamAnalyzer.ParmUSRsListMap;
  return Err;
}
//---------------------------------------------------------------------------
int FusionTool::renameParameters(const AnalysisContext &AContext)
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // Collect informations of parameters to be renamed
  vector<string>         NewParams;
  vector<string>         PrevParams;
  vector<vector<string>> USRs;

  for (auto &KName : AContext.Kernels) {
    auto &PrevParamList = AContext.ParmListMap.at(KName);
    auto &USRsList      = AContext.ParmUSRsListMap.at(KName);

    vector<string> NewParamList{PrevParamList.size()};
    transform(PrevParamList.begin(), PrevParamList.end(),
              NewParamList.begin(),
              [&KName](const string &PName) {
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
int FusionTool::rewriteCUDAVariables(const AnalysisContext &AContext)
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
int FusionTool::extractSharedDecls(AnalysisContext &AContext)
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDASharedDeclExtractor Extractor{Tool.getReplacements()};
  CUDASharedDeclRewriter Writer(Tool.getReplacements(),
                                AContext.SharedDeclStringMap,
                                AContext.Kernels);

  for (auto &KName : AContext.Kernels) {
    Finder.addMatcher(Extractor.getSharedDeclMatcher(KName), &Extractor);
    Finder.addMatcher(Writer.getSharedDeclMatcher(KName), &Writer);
  }

  auto Err = Tool.runAndSave(newFrontendActionFactory(&Finder).get());
  if (Err) {
    return Err;
  }

  AContext.SharedDeclStringMap = Extractor.SharedDeclStringMap;
  return Err;
}
//---------------------------------------------------------------------------
int FusionTool::hoistSharedDecls(const AnalysisContext &AContext)
{
  // // Refactoring Tool
  // RefactoringTool Tool(OptionsParser.getCompilations(),
  //                      OptionsParser.getSourcePathList());

  // // Add AST matchers
  // MatchFinder Finder;
  // CUDASharedDeclRewriter Writer(Tool.getReplacements(),
  //                               AContext.SharedDeclStringMap);

  // for (auto &KName : AContext.Kernels) {
  //   Finder.addMatcher(Writer.getFuncDeclMatcher(KName), &Writer);
  // }
  // return Tool.runAndSave(newFrontendActionFactory(&Finder).get());
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
    Finder.addMatcher(Analyzer.getSharedDeclMatcher(KName), &Analyzer);
  }

  auto Err = Tool.run(newFrontendActionFactory(&Finder).get());
  if (Err) {
    return Err;
  }

  // Sorting containers
  auto SortFunc = [](auto &Container, auto &Criteria) {
    return [&Container, &Criteria](auto &A, auto &B) {
      int AIdx = find(Container.begin(), Container.end(), A) - Container.begin();
      int BIdx = find(Container.begin(), Container.end(), B) - Container.begin();
      return Criteria[AIdx] > Criteria[BIdx];
    };
  };

  auto ShrdVarListMap     = Analyzer.ShrdVarListMap;
  auto ShrdVarUSRsListMap = Analyzer.ShrdVarUSRsListMap;
  auto ShrdVarSizeListMap = Analyzer.ShrdVarSizeListMap;

  for (auto &KName : AContext.Kernels) {
    auto &ShrdVarList     = ShrdVarListMap.at(KName);
    auto &ShrdVarUSRsList = ShrdVarUSRsListMap.at(KName);
    auto &ShrdVarSizeList = ShrdVarSizeListMap.at(KName);

    auto &OldShrdVarList     = Analyzer.ShrdVarListMap.at(KName);
    auto &OldShrdVarUSRsList = Analyzer.ShrdVarUSRsListMap.at(KName);
    auto &OldShrdVarSizeList = Analyzer.ShrdVarSizeListMap.at(KName);

    sort(ShrdVarList.begin(), ShrdVarList.end(),
         SortFunc(OldShrdVarList, OldShrdVarSizeList));
    sort(ShrdVarUSRsList.begin(), ShrdVarUSRsList.end(),
         SortFunc(OldShrdVarUSRsList, OldShrdVarSizeList));
    sort(ShrdVarSizeList.begin(), ShrdVarSizeList.end(),
         SortFunc(OldShrdVarSizeList, OldShrdVarSizeList));
  }

  AContext.ShrdVarListMap     = ShrdVarListMap;
  AContext.ShrdVarUSRsListMap = ShrdVarUSRsListMap;
  AContext.ShrdVarSizeListMap = ShrdVarSizeListMap;

  // Generate new shared memory declarations for fused kernel
  string ShrdDeclStr;
  llvm::raw_string_ostream ShrdDeclStream{ShrdDeclStr};

  string VNameBase = "union_shared_";
  bool AllVisited;
  uint64_t MaxSize;

  for (long unsigned I = 0;; ++I) {
    AllVisited = true;
    MaxSize    = 0;

    for (auto &KName : AContext.Kernels) {
      auto &ShrdVarSizeList = ShrdVarSizeListMap.at(KName);

      if (I >= ShrdVarSizeList.size())
        continue;

      AllVisited = false;
      MaxSize    = ShrdVarSizeList[I] > MaxSize ? ShrdVarSizeList[I] : MaxSize;
    }

    if (AllVisited)
      break;

    // FIXME: need to print by more general methology
    // i.e. static float pad_temp_shared[2320] __attribute__((shared));
    ShrdDeclStream << "  static float union_shared_" << I << "_[" << MaxSize << "] __attribute__((shared));\n";
  }
  ShrdDeclStream << "\n";
  ShrdDeclStream.flush();

  AContext.NewShrdDeclString = ShrdDeclStr;
  return Err;
}
//---------------------------------------------------------------------------
int FusionTool::renameSharedVariables(const AnalysisContext &AContext)
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // Generate new names
  vector<string>         NewShrdVars;
  vector<string>         PrevShrdVars;
  vector<vector<string>> USRs;

  for (auto &KName : AContext.Kernels) {
    auto &PrevShrdVarList = AContext.ShrdVarListMap.at(KName);
    auto &USRsList        = AContext.ShrdVarUSRsListMap.at(KName);

    vector<string> NewShrdVarList;
    string NewNameBase = "union_shared_";
    for (long unsigned I = 0; I < PrevShrdVarList.size(); ++I) {
      NewShrdVarList.push_back(NewNameBase + to_string(I) + "_");
    }

    NewShrdVars.insert(NewShrdVars.end(),
                       NewShrdVarList.begin(), NewShrdVarList.end());
    PrevShrdVars.insert(PrevShrdVars.end(),
                        PrevShrdVarList.begin(), PrevShrdVarList.end());
    USRs.insert(USRs.end(),
                USRsList.begin(), USRsList.end());
  }

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
    Finder.addMatcher(Builder.getFuncBuildMatcher(KName),
                      &Builder);
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
    auto Matcher = Printer.getFuncDeclMatcher(KName);
    Finder.addMatcher(Matcher, &Printer);
  }
  return Tool.run(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
} // tools
} // bfuse
//---------------------------------------------------------------------------