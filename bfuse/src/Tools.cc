
#include <cstdlib>
#include <cassert>
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
    RawStream << "(" << V << " >= " << Info.first << " && "
              << V << " < " << Info.second << ")";
    RawStream.flush();
    return Str;
  };

  // Create temp blockIdx, gridDim declarations
  string &TmpVarStr = Analysis.TmpBlockInfoString;
  llvm::raw_string_ostream TmpVarStream{TmpVarStr};

  TmpVarStream << "\n"
               << "  // FIXME: need to be deleted later\n"
               << "  int gridDim_x_  = 0; // temp declaration\n"
               << "  int blockIdx_x_ = 0; // temp declaration\n\n";
  TmpVarStream.flush();

  // Create new blockIdx, gridDim declarations
  // and define which kernel to be executed
  string &VarStr = Analysis.NewBlockInfoStringMap;
  llvm::raw_string_ostream VarStream{VarStr};

  // Comments
  auto &Kernels = FContext.kernels;

  VarStream << "  /*\n"
            << "   * KernelID_ means...\n";

  for (long unsigned I = 0; I < Kernels.size(); ++I) {
    string &KName = Kernels[I];
    VarStream << "   * " << I << ": " << KName << "\n";
  }
  VarStream << "   */\n";

  // Declarations
  VarStream << "  int gridDim_x_;\n"
            << "  int blockIdx_x_;\n"
            << "  int Others_;\n"
            << "  int KernelID_;\n"
            << "  \n";

  auto &KernelInfoMap    = FContext.kernelInfoMap;
  auto &KernelContextMap = FContext.kernelContextMap;
  bool IsAllVisited      = false;

  for (long unsigned VI = 0; !IsAllVisited; ++VI) {
    IsAllVisited = true;

    for (long unsigned KI = 0; KI < Kernels.size(); ++KI) {
      auto &KName         = Kernels[KI];
      auto &KernelInfo    = KernelInfoMap.at(KName);
      auto &KernelContext = KernelContextMap.at(KName);
      auto &BlockIdxInfo  = KernelContext.blockIdxInfo;
      auto &OtherBlocks   = KernelContext.otherBlocks;

      // Validation check
      assert(BlockIdxInfo.size() == OtherBlocks.size());
      if (VI >= BlockIdxInfo.size())
        continue;

      IsAllVisited = false;

      if (VI == 0 && KI == 0) { // first if case
        VarStream << "  ";
      } else {
        VarStream << "  else ";
      }

      VarStream << "if " << PrintInfoToCondFunc("blockIdx.x", BlockIdxInfo[VI]) << "\n"
                << "  {\n"
                << "    gridDim_x_ = " << KernelInfo.gridDim.size() << ";\n"
                << "    Others_    = " << OtherBlocks[VI] << ";\n"
                << "    KernelID_  = " << KI << ";\n"
                << "  }\n";
    }
  }
  VarStream << "  blockIdx_x_ = blockIdx.x - Others_;\n"
            << "  \n";
  VarStream.flush();

  // Create boundary conditions for each kernel
  // And calculate mx thread bound 
  int MaxBound = 0;
  for (long unsigned I = 0; I < Kernels.size(); ++I) {
    string &KName       = Kernels[I];
    auto &KernelContext = KernelContextMap.at(KName);
    auto &ThreadIdxInfo = KernelContext.threadIdxInfo;

    string CondStr;
    llvm::raw_string_ostream CondStream{CondStr};

    // "KernelID_" condition check
    CondStream << "(KernelID_ == " << I << ") && ";

    // threadIdx condition check
    CondStream << "(" << PrintInfoToCondFunc("threadIdx.x", ThreadIdxInfo) << ")";
    CondStream.flush();

    Analysis.BranchConditionMap[KName] = CondStr;
    Analysis.ThreadNumMap[KName]       = ThreadIdxInfo.second;

    // Calculate max boundary
    MaxBound = MaxBound < ThreadIdxInfo.second ? ThreadIdxInfo.second : MaxBound;
  }

  // Save max thread bound
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
  CUDABlockInfoRewriter BlockInfoRewriter{Tool.getReplacements(), Analysis.TmpBlockInfoString};
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
int FusionTool::saveFunction(AnalysisContext &Analysis, string &FuncStr)
{
  std::error_code EC;
  std::string FileName = Analysis.NewFuncName + ".cu";
  llvm::raw_fd_ostream FdStream{FileName, EC};

  FdStream << FuncStr;
  FdStream.close();

  return 0;
}
//---------------------------------------------------------------------------
} // tools
} // bfuse
//---------------------------------------------------------------------------