
#pragma once

#include <string>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

//---------------------------------------------------------------------------
namespace bfuse {
namespace matchers {
//---------------------------------------------------------------------------
const std::string CUDAFunctionDeclBindId = "cudaFunctionDecl";
//---------------------------------------------------------------------------
class CUDAFunctionDeclPrinter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
};
//---------------------------------------------------------------------------
} // namespace matchers
} // namespace bfuse
//---------------------------------------------------------------------------