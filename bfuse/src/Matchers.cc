
#include <cstdlib>
#include <string>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Refactoring/Rename/USRFindingAction.h"

#include "bfuse/Utils.h"
#include "bfuse/Matchers.h"

using namespace std;

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;
//---------------------------------------------------------------------------
namespace bfuse {
namespace matchers {
//---------------------------------------------------------------------------
DeclarationMatcher CUDAFuncDeclPrinter::getFuncDeclMatcher(std::string &KName)
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
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);
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
void CUDAFuncParamAnalyzer::run(const MatchFinder::MatchResult &Result)
{
  ASTContext    *Context = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);
  const ParmVarDecl *PD  = Result.Nodes.getNodeAs<ParmVarDecl>(CUDAFuncParamBindId);
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
           ))
         ).bind(CUDAIdxAndDimMemberBindId);
}
//---------------------------------------------------------------------------
void CUDABlockInfoRewriter::run(const MatchFinder::MatchResult &Result)
{
  ASTContext  *Context = Result.Context;
  const MemberExpr *ME = Result.Nodes.getNodeAs<MemberExpr>(CUDAIdxAndDimMemberBindId);
  const VarDecl    *VD = Result.Nodes.getNodeAs<VarDecl>(CUDAIdxAndDimBindId);
  if (!ME || !VD) {
    ERROR_MESSAGE("cannot find block information pattern");
    return;
  }

  map<string, string> MemberReNamingMap = {
    {"__fetch_builtin_x", "x"},
    {"__fetch_builtin_y", "y"},
    {"__fetch_builtin_z", "z"}
  };
  auto ReNamingFunc = [](string &Var, string &Member) { return Var + "_" + Member + "_"; };

  string MName    = ME->getMemberNameInfo().getAsString();
  string VName    = VD->getNameAsString();
  string NewMName = MemberReNamingMap.at(MName);
  string NewVName = ReNamingFunc(VName, NewMName);

  auto CharSrcRange = CharSourceRange::getTokenRange(ME->getSourceRange());
  auto &SourceMgr   = Context->getSourceManager();

  Replacement Repl{SourceMgr, CharSrcRange, NewVName};
  string FilePath = Repl.getFilePath().str();

  if (auto Err = Repls[FilePath].add(Repl)) {
    llvm::errs() << "CUDABlockInfoRewriter error occur\n";
    exit(0);
  }
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

  string FName        = FD->getNameAsString();
  auto &KContext      = KernelContextMap.at(FName);
  auto &ThreadIdxInfo = KContext.threadIdxInfo;

  auto RefactoringFunc = [](int BlockDim) { return "asm(\"bar.sync 0, " + to_string(BlockDim) + ";\");"; };
  string NewSync       = RefactoringFunc(ThreadIdxInfo.second);
  auto &SourceMgr      = Context->getSourceManager();

  Replacement Repl{SourceMgr, CE, NewSync};
  string FilePath = Repl.getFilePath().str();

  if (auto Err = Repls[FilePath].add(Repl)) {
    llvm::errs() << "CUDASyncRewriter error occur\n";
    exit(0);
  }
}
//---------------------------------------------------------------------------
} // namespace matchers
} // namespace bfuse
//---------------------------------------------------------------------------