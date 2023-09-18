
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <numeric>
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

#include "llvm/Support/raw_ostream.h"

#include "bfuse/Contexts.h"
#include "bfuse/Matchers.h"
#include "bfuse/Tools.h"
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
int FusionTool::analyzeParameters(AnalysisContext &Analysis)
{
  // Clang Tool
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDAFuncParamAnalyzer ParamAnalyzer;

  for (auto &KName : FContext.kernels) {
    Finder.addMatcher(ParamAnalyzer.getFuncParamMatcher(KName),
                      &ParamAnalyzer);
  }

  int Err = Tool.run(newFrontendActionFactory(&Finder).get());
  if (Err) {
    return Err;
  }

  Analysis.kernels      = FContext.kernels;
  Analysis.ParamListMap = ParamAnalyzer.ParamListMap;
  Analysis.USRsListMap  = ParamAnalyzer.USRsListMap;
  return Err;
}
//---------------------------------------------------------------------------
int FusionTool::analyzeThreadBoundaries(AnalysisContext &Analysis)
{
  auto PrintInfoToCondFunc = [](string V, auto &Info) {
    string Str;
    llvm::raw_string_ostream RawStream{Str};
    RawStream << "((" << V << " >= " << Info.first << ") && "
              << "(" << V << " < " << Info.second << "))";
    RawStream.flush();
    return Str;
  };

  int MaxBound = 0;
  for (auto &KName : FContext.kernels) {
    string CondStr;
    llvm::raw_string_ostream CondStream{CondStr};
    auto &KernelContext = FContext.kernelContextMap.at(KName);
    auto &BlockIdxInfo  = KernelContext.blockIdxInfo;
    auto &ThreadIdxInfo = KernelContext.threadIdxInfo;

    // threadIdx condition
    CondStream << "(";
    CondStream << PrintInfoToCondFunc("threadIdx.x", ThreadIdxInfo);
    CondStream << ") && ";

    // blockIdx condition
    CondStream << "(";
    CondStream << accumulate(BlockIdxInfo.begin() + 1,
                             BlockIdxInfo.end(),
                             PrintInfoToCondFunc("blockIdx.x", BlockIdxInfo[0]),
                             [&](string &Acc, auto &Info) {
                               return Acc + " || " + PrintInfoToCondFunc("blockIdx.x", Info);
                             }
                            );
    CondStream << ")";
    CondStream.flush();

    Analysis.BranchConditionMap[KName] = CondStr;
    Analysis.ThreadNumMap[KName]       = ThreadIdxInfo.second;

    MaxBound = MaxBound < ThreadIdxInfo.second ? ThreadIdxInfo.second : MaxBound;
  }

  Analysis.MaxThreadBound = MaxBound;
  return 0;
}
//---------------------------------------------------------------------------
int FusionTool::renameParameters(AnalysisContext &Analysis)
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
int FusionTool::rewriteCUDAInfos(AnalysisContext &Analysis)
{
  // Refactoring Tool
  RefactoringTool Tool(OptionsParser.getCompilations(),
                       OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDABlockInfoRewriter BlockInfoRewriter{Tool.getReplacements()};
  CUDASyncRewriter      SyncRewriter{Tool.getReplacements(), Analysis.ThreadNumMap};

  for (auto &KName : Analysis.kernels) {
    Finder.addMatcher(BlockInfoRewriter.getBlockInfoMatcher(KName),
                      &BlockInfoRewriter);
    Finder.addMatcher(SyncRewriter.getSyncMatcher(KName),
                      &SyncRewriter);
  }
  return Tool.runAndSave(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionTool::printFuncDeclExample() const
{
  // Clang Tool
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDAFuncDeclPrinter Printer;

  for (auto &KName : FContext.kernels) {
    auto Matcher = Printer.getFuncDeclMatcher(KName);
    Finder.addMatcher(Matcher, &Printer);
  }
  return Tool.run(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionTool::createFunction(AnalysisContext &Analysis, string &FuncStr)
{
  // Clang Tool
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDAFuncBuilder Builder{Analysis, FuncStr};

  for (auto &KName : Analysis.kernels) {
    auto Matcher = Builder.getFuncBuildMatcher(KName);
    Finder.addMatcher(Matcher, &Builder);
  }
  return Tool.run(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
} // tools
} // bfuse
//---------------------------------------------------------------------------