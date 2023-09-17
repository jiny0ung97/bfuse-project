
#pragma once

#include <string>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

//---------------------------------------------------------------------------
namespace bfuse {
namespace matchers {
//---------------------------------------------------------------------------
class CUDAFunctionDeclPrinter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The cuda function declaration bind id
  const std::string CUDAFunctionDeclBindId = "cudaFunctionDecl";

public:
  /// Get matcher value
  clang::ast_matchers::DeclarationMatcher getDeclarationMatcher() const;
  /// Run AST matcher
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
};
//---------------------------------------------------------------------------
} // namespace matchers
} // namespace bfuse
//---------------------------------------------------------------------------