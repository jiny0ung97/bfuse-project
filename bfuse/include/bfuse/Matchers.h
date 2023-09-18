
#pragma once

#include <string>
#include <map>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Core/Replacement.h"

#include "llvm/Support/raw_ostream.h"

#include "bfuse/Contexts.h"
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
  /// Run AST match finder
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
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
};
//---------------------------------------------------------------------------
class CUDABlockInfoRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  using FileReplacementsMap = std::map<std::string, clang::tooling::Replacements>;

private:
  /// The cuda block information member (x, y, z) bind id
  const std::string CUDAIdxAndDimMemberBindId = "cudaIdxAndDimMember";
  /// The cuda block information bind id
  const std::string CUDAIdxAndDimBindId = "cudaIdxAndDim";

  /// The container of refactoring replacements
  FileReplacementsMap &Repls;

public:
  /// Get block information declarations' reference matcher
  clang::ast_matchers::StatementMatcher getBlockInfoMatcher(std::string &KName);
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  explicit CUDABlockInfoRewriter(FileReplacementsMap &OtherRepls) : Repls{OtherRepls} {}
};
//---------------------------------------------------------------------------
class CUDASyncRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  using FileReplacementsMap  = std::map<std::string, clang::tooling::Replacements>;
  using NameKernelContextMap = std::map<std::string, contexts::KernelContext>;

private:
  /// The cuda function declaration bind id
  const std::string CUDAFuncDeclBindId = "cudaFuncDecl";
  /// The cuda synchronization bind id
  const std::string CUDASyncBindId = "cudaSync";

  /// The container of refactoring replacements
  FileReplacementsMap &Repls;
  /// The vector to contain kernel contexts
  NameKernelContextMap &KernelContextMap;

public:
  /// Get synchronization matcher
  clang::ast_matchers::StatementMatcher getSyncMatcher(std::string &KName);
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  explicit CUDASyncRewriter(FileReplacementsMap &OtherRepls,
                            NameKernelContextMap &OtherKernelContextMap)
                           : Repls{OtherRepls}, KernelContextMap{OtherKernelContextMap} {}
};
//---------------------------------------------------------------------------
class CUDAFuncBuilder
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The cuda function declaration bind id
  const std::string CUDAFuncDeclBindId = "cudaFuncDecl";
  /// The cuda function parameters bind id
  const std::string CUDAFuncParamBindId = "cudaFuncParam";

  /// The analysis of functions to be fused
  contexts::AnalysisContext &Analysis;

  /// The string stream of fused function
  llvm::raw_string_ostream FuncStream;
  /// The list of functions to be fused
  std::map<std::string, std::string> FuncBodyStringMap;
  /// The string list of parameters
  std::vector<std::string> ParmStringList;

public:
  /// Get function declaration matcher
  clang::ast_matchers::DeclarationMatcher getFuncBuildMatcher(std::string &KName);
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
  /// Run finder at the end of the translation unit
  virtual void onEndOfTranslationUnit() override;

  /// The constructor
  CUDAFuncBuilder(contexts::AnalysisContext &OtherAnalysis, std::string &FuncStr)
                 : Analysis{OtherAnalysis}, FuncStream{FuncStr} {}
};
//---------------------------------------------------------------------------
} // namespace matchers
} // namespace bfuse
//---------------------------------------------------------------------------