
#pragma once

#include <string>
#include <map>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Core/Replacement.h"

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
class CUDAFuncParamAnalyzer
      : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  using ParamList = std::vector<std::string>;
  using USRsList  = std::vector<std::vector<std::string>>;

private:
  /// The cuda function declaration bind id
  const std::string CUDAFuncDeclBindId = "cudaFuncDecl";
  /// The cuda function parameters bind id
  const std::string CUDAFuncParamBindId = "cudaFuncParam";

public:
  /// The map of function parameters' list
  std::map<std::string, ParamList> ParamListMap;
  /// The map of USRs lists for renaming parameters
  std::map<std::string, USRsList> USRsListMap;

  /// Get function parameters matcher
  clang::ast_matchers::DeclarationMatcher getFuncParamMatcher(std::string &Kname);
  /// Run AST matcher
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
};
//---------------------------------------------------------------------------
class CUDABlockIdxRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  using FileReplacementsMap = std::map<std::string, clang::tooling::Replacements>;

private:
  /// The cuda blockIdx bind id
  const std::string CUDABlockIdxBindId = "cudaBlockIdx";

  /// The container of refactoring replacements
  FileReplacementsMap &Repls;

public:
  /// Get blockIdx declaration reference matcher
  clang::ast_matchers::StatementMatcher getBlockIdxMatcher(std::string &KName);
  /// Run AST matcher
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  explicit CUDABlockIdxRewriter(FileReplacementsMap &OtherReps) : Repls{OtherReps} {}
};
//---------------------------------------------------------------------------
class CUDASyncRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  using FileReplacementsMap = std::map<std::string, clang::tooling::Replacements>;

private:
  /// The cuda synchronization bind id
  const std::string CUDASyncBindId = "cudaSync";

  /// The container of refactoring replacements
  FileReplacementsMap &Repls;

public:
  /// Get synchronization matcher
  clang::ast_matchers::StatementMatcher getSyncMatcher(std::string &KName);
  /// Run AST matcher
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  explicit CUDASyncRewriter(FileReplacementsMap &OtherRepls) : Repls{OtherRepls} {}
};
//---------------------------------------------------------------------------
} // namespace matchers
} // namespace bfuse
//---------------------------------------------------------------------------