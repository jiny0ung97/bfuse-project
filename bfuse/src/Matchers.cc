
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include "bfuse/Matchers.h"

using namespace clang;
using namespace clang::ast_matchers;
//---------------------------------------------------------------------------
namespace bfuse {
namespace matchers {
//---------------------------------------------------------------------------
void CUDAFunctionDeclPrinter::run(const MatchFinder::MatchResult &Result) {
  if (const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFunctionDeclBindId)) {
    FD->dump();
  }
}
//---------------------------------------------------------------------------
} // namespace matchers
} // namespace bfuse
//---------------------------------------------------------------------------