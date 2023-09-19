
#include <cstdlib>
#include <numeric>
#include <string>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Refactoring/Rename/USRFindingAction.h"

#include "llvm/Support/raw_ostream.h"

#include "bfuse/Contexts.h"
#include "bfuse/Utils.h"
#include "bfuse/Matchers.h"

using namespace std;

using namespace bfuse::contexts;

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;
//---------------------------------------------------------------------------
namespace bfuse {
namespace matchers {
//---------------------------------------------------------------------------
DeclarationMatcher CUDAFuncDeclPrinter::getFuncDeclMatcher(string &KName)
{
  return functionDecl(
           hasAttr(attr::CUDAGlobal),
           hasName(KName)
         ).bind(CUDAFuncDeclBindId);
}
//---------------------------------------------------------------------------
void CUDAFuncDeclPrinter::run(const MatchFinder::MatchResult &Result)
{
  // ASTContext    *Context = Result.Context;
  const FunctionDecl* FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);
  if (!FD) {
    ERROR_MESSAGE("cannot find function declaration pattern");
    return;
  }

  FD->dump();
}
//---------------------------------------------------------------------------
DeclarationMatcher CUDAFuncParamAnalyzer::getFuncParamMatcher(string &Kname)
{
  return parmVarDecl(
           hasAncestor(
             functionDecl(
               hasAttr(attr::CUDAGlobal),
               hasName(Kname)
             ).bind(CUDAFuncDeclBindId)
           )
         ).bind(CUDAFuncParamBindId);
}
//---------------------------------------------------------------------------
void CUDAFuncParamAnalyzer::run(const MatchFinder::MatchResult& Result)
{
  ASTContext *Context = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);
  const ParmVarDecl  *PD = Result.Nodes.getNodeAs<ParmVarDecl>(CUDAFuncParamBindId);
  if (!FD || !PD) {
    ERROR_MESSAGE("cannot find parameter pattern");
    return;
  }

  auto FName    = FD->getNameAsString();
  auto PName    = PD->getNameAsString();
  auto ParamUSR = getUSRsForDeclaration(PD->getUnderlyingDecl(), *Context);

  ParamListMap[FName].push_back(PName);
  USRsListMap[FName].push_back(ParamUSR);
}
//---------------------------------------------------------------------------
StatementMatcher CUDABlockInfoRewriter::getBlockInfoMatcher(string &KName)
{
  return memberExpr(
           hasObjectExpression(
             opaqueValueExpr(
               hasSourceExpression(
                 declRefExpr(
                   to(varDecl(
                     anyOf(
                       hasName("gridDim"),
                       hasName("blockIdx")
                     )).bind(CUDAIdxAndDimBindId)
           ))))),
           hasAncestor(
             functionDecl(
               hasAttr(attr::CUDAGlobal),
               hasName(KName)
             ).bind(CUDAFuncDeclBindId)
           )
         ).bind(CUDAIdxAndDimMemberBindId);
}
//---------------------------------------------------------------------------
void CUDABlockInfoRewriter::run(const MatchFinder::MatchResult &Result)
{
  ASTContext  *Context   = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);
  const MemberExpr *ME   = Result.Nodes.getNodeAs<MemberExpr>(CUDAIdxAndDimMemberBindId);
  const VarDecl    *VD   = Result.Nodes.getNodeAs<VarDecl>(CUDAIdxAndDimBindId);
  if (!ME || !VD) {
    ERROR_MESSAGE("cannot find block information pattern");
    return;
  }

  // Rewrite thread informations
  map<string, string> MemberReNamingMap = {
    {"__fetch_builtin_x", "x"},
    // {"__fetch_builtin_y", "y"},
    // {"__fetch_builtin_z", "z"}
  };
  auto ReNamingFunc = [](string &Var, string &Member) { return Var + "_" + Member + "_"; };

  string MName    = ME->getMemberNameInfo().getAsString();
  string VName    = VD->getNameAsString();
  string NewMName = MemberReNamingMap.at(MName);
  string NewVName = ReNamingFunc(VName, NewMName);

  auto CharSrcRange = CharSourceRange::getTokenRange(ME->getSourceRange());
  auto& SourceMgr   = Context->getSourceManager();

  Replacement RenameRepl{SourceMgr, CharSrcRange, NewVName};
  string RFilePath = RenameRepl.getFilePath().str();

  if (auto Err = Repls[RFilePath].add(RenameRepl)) {
    llvm::errs() << "CUDABlockInfoRewriter error occur\n";
    exit(0);
  }

  // Append new declaration of thread information
  string FName = FD->getNameAsString();
  if (VisitedFuncSet.find(FName) != VisitedFuncSet.end())
    return; // Already append. return

  auto SourceBeginLoc = FD->getBody()->getBeginLoc().getLocWithOffset(1);
  auto SourceEndLoc   = FD->getBody()->getEndLoc().getLocWithOffset(-1);

  // Append compound {}
  string CompBeginString = TmpBlockInfoString + "  {";
  string CompEndString   = "\n  }";

  Replacement CompBeginRepl{SourceMgr, SourceBeginLoc, 0, CompBeginString};
  string CBFile = CompBeginRepl.getFilePath().str();
  if (auto Err = Repls[CBFile].add(CompBeginRepl)) {
    llvm::errs() << "CUDABlockInfoRewriter error occur\n";
    exit(0);
  }

  Replacement CompEndRepl{SourceMgr, SourceEndLoc, 0, CompEndString};
  string CEFile = CompEndRepl.getFilePath().str();
  if (auto Err = Repls[CEFile].add(CompEndRepl)) {
    llvm::errs() << "CUDABlockInfoRewriter error occur\n";
    exit(0);
  }
  VisitedFuncSet.insert(FName);
}
//---------------------------------------------------------------------------
StatementMatcher CUDASyncRewriter::getSyncMatcher(string &KName)
{
  return callExpr(
           callee(
             functionDecl(
               hasName("__syncthreads")
           )),
           hasAncestor(
             functionDecl(
               hasAttr(attr::CUDAGlobal),
               hasName(KName)
             ).bind(CUDAFuncDeclBindId)
           )
         ).bind(CUDASyncBindId);
}
//---------------------------------------------------------------------------
void CUDASyncRewriter::run(const MatchFinder::MatchResult &Result)
{
  ASTContext    *Context = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);
  const CallExpr     *CE = Result.Nodes.getNodeAs<CallExpr>(CUDASyncBindId);
  if (!FD || !CE) {
    ERROR_MESSAGE("cannot find __syncthreads() pattern");
    return;
  }

  auto RefactoringFunc = [](int BlockDim) { return "asm(\"bar.sync 0, " + to_string(BlockDim) + ";\")"; };
  string FName  = FD->getNameAsString();
  int ThreadNum = ThreadNumMap.at(FName);

  string NewSync  = RefactoringFunc(ThreadNum);
  auto &SourceMgr = Context->getSourceManager();

  Replacement Repl{SourceMgr, CE, NewSync};
  string FilePath = Repl.getFilePath().str();

  if (auto Err = Repls[FilePath].add(Repl)) {
    llvm::errs() << "CUDASyncRewriter error occur\n";
    exit(0);
  }
}
//---------------------------------------------------------------------------
DeclarationMatcher CUDAFuncBuilder::getFuncBuildMatcher(string &KName)
{
  return parmVarDecl(
           hasAncestor(
             functionDecl(
               hasAttr(attr::CUDAGlobal),
               hasName(KName)
             ).bind(CUDAFuncDeclBindId)
           )
         ).bind(CUDAFuncParamBindId);
}
//---------------------------------------------------------------------------
void CUDAFuncBuilder::run(const MatchFinder::MatchResult &Result)
{
  ASTContext    *Context = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);
  const ParmVarDecl  *PD = Result.Nodes.getNodeAs<ParmVarDecl>(CUDAFuncParamBindId);
  if (!FD || !PD) {
    ERROR_MESSAGE("cannot find function declaration pattern");
    return;
  }

  // Print function body
  string FName         = FD->getNameAsString();
  auto &FuncBodyStrMap = Analysis.FuncBodyStringMap;
  auto &PrintPolicy    = Context->getPrintingPolicy();

  string BodyStr;
  llvm::raw_string_ostream BodyStream{BodyStr};

  if (FuncBodyStrMap.find(FName) == FuncBodyStrMap.end()) {

    // Assume that every function has blockIdx.x and gridDim.x
    for (auto *Child : FD->getBody()->children()) {
      if (auto *CS = dyn_cast<CompoundStmt>(Child)) {
        CS->printPretty(BodyStream, nullptr, PrintPolicy, /*Indentation=*/1U);
      }
    }
    BodyStream.flush();
    FuncBodyStrMap[FName] = BodyStr;
  }

  // Print function parameters
  auto &ParmStrList = Analysis.ParmStringList;

  string ParamStr;
  llvm::raw_string_ostream ParamStream{ParamStr};

  PD->print(ParamStream, Context->getPrintingPolicy());
  ParamStream.flush();
  ParmStrList.push_back(ParamStr);
}
//---------------------------------------------------------------------------
void CUDAFuncBuilder::onEndOfTranslationUnit()
{
  // Macro
  string TVMMacros =
R"(
#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
      (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
)";

  // Attributes
  string ExternAttr     = "extern \"C\"";
  string CUDAGlobalAttr = "__global__";
  string CUDALaunchAttr = "";
  CUDALaunchAttr += "__launch_bounds__(" + to_string(Analysis.MaxThreadBound) + ")";

  // Function declaration (name)
  string CUDAFuncName = "";
  for (auto &KName : Analysis.kernels) {
    CUDAFuncName += KName + "_";
  }
  CUDAFuncName += "fused_";

  // Function declaration (paramenter)
  // FIXME: need to fix __restrict -> __restrict__
  // Maybe bug?
  auto &ParmStrList = Analysis.ParmStringList;
  auto AccFunc = [](string a, string b) { return a + ", " + b; };

  string CUDAFuncParam = accumulate(ParmStrList.begin() + 1,
                                    ParmStrList.end(),
                                    ParmStrList[0],
                                    AccFunc);
  
  // Function body
  auto &BranchCondMap  = Analysis.BranchConditionMap;
  auto &FuncBodyStrMap = Analysis.FuncBodyStringMap;
  auto &Kernels        = Analysis.kernels;
  string CUDAFuncBody  = Analysis.NewBlockInfoStringMap;

  for (long unsigned I = 0; I < Kernels.size(); ++I) {
    auto &KName   = Kernels[I];
    auto &CondStr = BranchCondMap.at(KName);
    auto &BodyStr = FuncBodyStrMap.at(KName);

    // Comments
    CUDAFuncBody += "  // " + KName + "\n";

    if (I == 0) {
      CUDAFuncBody += "  ";
    } else {
      CUDAFuncBody += "  else ";
    }
    CUDAFuncBody += "if (" + CondStr + ")\n";
    CUDAFuncBody += BodyStr; // include {}
  }

  // Create fused function
  FuncStream << TVMMacros
             << ExternAttr << " " << CUDAGlobalAttr << " "
             << CUDALaunchAttr << " " << CUDAFuncName << "("
             << CUDAFuncParam << ")\n"
             << "{\n"
             <<    CUDAFuncBody
             << "}\n";
             
  FuncStream.flush();

  // Save new function name
  Analysis.NewFuncName = CUDAFuncName;
}
//---------------------------------------------------------------------------
} // namespace matchers
} // namespace bfuse
//---------------------------------------------------------------------------