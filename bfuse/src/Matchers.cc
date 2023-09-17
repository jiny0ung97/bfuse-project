
#include <string>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include "bfuse/Utils.h"
#include "bfuse/Matchers.h"

using namespace std;

using namespace clang;
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
         )
         .bind(CUDAFuncDeclBindId);
}
//---------------------------------------------------------------------------
void CUDAFuncDeclPrinter::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);

  if (!FD) {
    ERROR_MESSAGE("cannot find function declaration");
    return;
  }

  FD->dump();
}
//---------------------------------------------------------------------------
DeclarationMatcher CUDAFuncParamRewriter::getFuncParamMatcher(string &Kname)
{
  return parmVarDecl(
           hasAncestor(
             functionDecl(
               hasAttr(attr::CUDAGlobal),
               hasName(Kname)
           ))
         )
         .bind(CUDAFuncParamBindId);
}
//---------------------------------------------------------------------------
void CUDAFuncParamRewriter::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context= Result.Context;
  const ParmVarDecl *PD = Result.Nodes.getNodeAs<ParmVarDecl>(CUDAFuncParamBindId);

  if (!PD) {
    ERROR_MESSAGE("cannot find parameter declarations");
    return;
  }

  PD->dump();
}
//---------------------------------------------------------------------------
StatementMatcher CUDABlockIdxRewriter::getBlockIdxMatcher(string &KName)
{
  return memberExpr(
           hasObjectExpression(
             opaqueValueExpr(
               hasSourceExpression(
                 declRefExpr(
                   to(varDecl(
                     hasName("blockIdx")
           )))))),
           hasAncestor(
             functionDecl(
               hasAttr(attr::CUDAGlobal),
               hasName(KName)
           ))
         )
         .bind(CUDABlockIdxBindId);
}
//---------------------------------------------------------------------------
void CUDABlockIdxRewriter::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context= Result.Context;
  const MemberExpr *ME = Result.Nodes.getNodeAs<MemberExpr>(CUDABlockIdxBindId);

  if (!ME) {
    ERROR_MESSAGE("cannot find blockIdx");
    return;
  }

  ME->dump();
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
           ))
         )
         .bind(CUDASyncBindId);
}
//---------------------------------------------------------------------------
void CUDASyncRewriter::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context= Result.Context;
  const CallExpr *CE = Result.Nodes.getNodeAs<CallExpr>(CUDASyncBindId);

  if (!CE) {
    ERROR_MESSAGE("cannot find __syncthreads()");
    return;
  }

  CE->dump();
}
//---------------------------------------------------------------------------
} // namespace matchers
} // namespace bfuse
//---------------------------------------------------------------------------