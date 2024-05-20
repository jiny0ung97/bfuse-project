
#include <utility>
#include <algorithm>
#include <numeric>
#include <string>
#include <map>
#include <tuple>
#include <functional>

#include "clang/Frontend/FrontendActions.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Refactoring/Rename/RenamingAction.h"

#include "llvm/Support/raw_ostream.h"

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
//--------------------------------------------------------------------------
static tuple<vector<string>, vector<string>, vector<vector<string>>>
renameVariables(vector<string> &Kernels,
                map<string, vector<string>> &VarListMap, map<string, vector<vector<string>>> &VarUSRsListMap,
                function<string(string, string)> &&RenameFunc)
{
  vector<string> NewVars;
  vector<string> PrevVars;
  vector<vector<string>> USRs;

  for (auto &KName : Kernels) {
    if (VarListMap.find(KName) == VarListMap.end()) {
      continue;
    }
    
    auto &PrevVarList = VarListMap.at(KName);
    auto &USRsList    = VarUSRsListMap.at(KName);

    vector<string> NewVarList{PrevVarList.size()};
    transform(PrevVarList.begin(), PrevVarList.end(),
              NewVarList.begin(),
              [&KName, &RenameFunc](const string &VName) {
                return RenameFunc(KName, VName);
              });

    NewVars.insert(NewVars.end(),
                   NewVarList.begin(), NewVarList.end());
    PrevVars.insert(PrevVars.end(),
                    PrevVarList.begin(), PrevVarList.end());
    USRs.insert(USRs.end(), USRsList.begin(), USRsList.end());
  }

  return make_tuple(NewVars, PrevVars, USRs);
}
//--------------------------------------------------------------------------
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
  auto [NewParams, PrevParams, USRs] = renameVariables(FContext_.Kernels_,
                                                       Analyzer.ParmListMap_, Analyzer.ParmUSRsListMap_,
                                                       [](const string &KName, const string &PName) {
                                                         return KName + "_" + PName + "_";
                                                       });

  // Run renaming frontend action
  RenamingAction Renaming{NewParams, PrevParams,
                          USRs, ReTool.getReplacements()};

  return ReTool.runAndSave(newFrontendActionFactory(&Renaming).get());
}
//---------------------------------------------------------------------------
int FusionTool::rewriteCUDAVariables()
{
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
int FusionTool::rewriteCUDASynchronize()
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
  CUDASyncRewriter SyncRewriter{Tool.getReplacements(), FContext_};

  for (auto &KName : FContext_.Kernels_) {
    Finder.addMatcher(ASTPatternMatcher::getSyncMatcher(KName),
                      &SyncRewriter);
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
int FusionTool::renameSharedVariables()
{
  // Clang Tool
  ClangTool Tool(OptionsParser_.getCompilations(),
                 OptionsParser_.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  CUDASharedVarAnalyzer Analyzer;

  for (auto &KName : FContext_.Kernels_) {
    Finder.addMatcher(ASTPatternMatcher::getSharedDeclMatcher(KName), &Analyzer);
  }

  auto Err = Tool.run(newFrontendActionFactory(&Finder).get());
  if (Err) {
    return Err;
  }

  // New Declaration of shared variables in the fused kernel
  llvm::raw_string_ostream UnionStream{UnionStr_};

  for (auto &KName : FContext_.Kernels_) {
    if (Analyzer.SharedDeclStrMap_.find(KName) == Analyzer.SharedDeclStrMap_.end()) {
      continue;
    }
    auto &ShrdDeclStr = Analyzer.SharedDeclStrMap_.at(KName);

    UnionStream << "  typedef struct " << KName << " {\n";
    for (auto &DeclStr : ShrdDeclStr) {
      UnionStream << "    " << DeclStr << ";\n";
    }
    UnionStream << "  } " << KName << "Ty_;\n";
  }

  UnionStream << "  typedef union ShrdUnion {\n";
  for (auto &KName : FContext_.Kernels_) {
    UnionStream << "    " << KName << "Ty_ " << KName <<";\n";
  }
  UnionStream << "  } ShrdUnionTy_;\n";
  UnionStream << "\n";
  UnionStream << "  __shared__ ShrdUnionTy_ SU_;\n";

  // Refactoring Tool
  RefactoringTool ReTool(OptionsParser_.getCompilations(),
                         OptionsParser_.getSourcePathList());

  // Collect parameters' information and return renamed parameter lists
  auto [NewShrdVars, PrevShrdVars, USRs] = renameVariables(FContext_.Kernels_,
                                                           Analyzer.ShrdVarListMap_, Analyzer.ShrdVarUSRsListMap_,
                                                           [](const string &KName, const string &SName) {
                                                             return "_SU_" + KName + "_" + SName;
                                                           });

  // Run renaming frontend action
  RenamingAction Renaming{NewShrdVars, PrevShrdVars,
                          USRs, ReTool.getReplacements()};

  return ReTool.runAndSave(newFrontendActionFactory(&Renaming).get());
}
//---------------------------------------------------------------------------
int FusionTool::createBFuseKernel()
{
  // Clang Tool
  ClangTool Tool(OptionsParser_.getCompilations(),
                 OptionsParser_.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  BFuseBuilder Builder{FContext_, UnionStr_, FuncStr_};

  for (auto &KName : FContext_.Kernels_) {
    Finder.addMatcher(ASTPatternMatcher::getFuncDeclMatcher(KName), &Builder);
  }
  return Tool.run(newFrontendActionFactory(&Finder).get());
}
//---------------------------------------------------------------------------
int FusionTool::createHFuseKernel()
{
  // Clang Tool
  ClangTool Tool(OptionsParser_.getCompilations(),
                 OptionsParser_.getSourcePathList());

  // Add AST matchers
  MatchFinder Finder;
  HFuseBuilder Builder{FContext_, FuncStr_};

  for (auto &KName : FContext_.Kernels_) {
    Finder.addMatcher(ASTPatternMatcher::getFuncDeclMatcher(KName), &Builder);
  }
  return Tool.run(newFrontendActionFactory(&Finder).get());
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