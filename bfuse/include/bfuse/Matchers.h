
#pragma once

#include <string>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

//---------------------------------------------------------------------------
namespace bfuse {
namespace matchers {
//---------------------------------------------------------------------------
class CUDAFuncDeclPrinter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The cuda function declaration bind id
  const std::string CUDAFuncDeclBindId = "cudaFuncDecl";

public:
  /// Get function declaration matcher
  clang::ast_matchers::DeclarationMatcher getFuncDeclMatcher(std::string &KName);
  /// Run AST matcher
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
};
//---------------------------------------------------------------------------
class CUDAFuncParamRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The cuda function parameters bind id
  const std::string CUDAFuncParamBindId = "cudaFuncParam";

public:
  /// Get function parameters matcher
  clang::ast_matchers::DeclarationMatcher getFuncParamMatcher(std::string &Kname);
  /// Run AST matcher
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
};
//---------------------------------------------------------------------------
class CUDABlockIdxRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The cuda blockIdx bind id
  const std::string CUDABlockIdxBindId = "cudaBlockIdx";

public:
  /// Get blockIdx declaration reference matcher
  clang::ast_matchers::StatementMatcher getBlockIdxMatcher(std::string &KName);
  /// Run AST matcher
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
};
//---------------------------------------------------------------------------
class CUDASyncRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The cuda synchronization bind id
  const std::string CUDASyncBindId = "cudaSync";

public:
  /// Get synchronization matcher
  clang::ast_matchers::StatementMatcher getSyncMatcher(std::string &KName);
  /// Run AST matcher
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
};
//---------------------------------------------------------------------------
} // namespace matchers
} // namespace bfuse
//---------------------------------------------------------------------------