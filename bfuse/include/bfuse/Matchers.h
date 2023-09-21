
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
class CUDADeclExtractor
      : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  using FileReplacementsMap = std::map<std::string, clang::tooling::Replacements>;
  
private:
  /// The cuda function declaration bind id
  const std::string CUDAFuncDeclBindId = "cudaFuncDecl";

  /// The container of refactoring replacements
  FileReplacementsMap &Repls;

public:
  /// Get function declaration matcher
  clang::ast_matchers::DeclarationMatcher getFuncDeclMatcher(std::string &KName);
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  explicit CUDADeclExtractor(FileReplacementsMap &OtherRepls)
                              : Repls{OtherRepls} {}
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
  /// The cuda function declaration bind id
  const std::string CUDAFuncDeclBindId = "cudaFuncDecl";
  /// The cuda block information member (x, y, z) bind id
  const std::string CUDAIdxAndDimMemberBindId = "cudaIdxAndDimMember";
  /// The cuda block information bind id
  const std::string CUDAIdxAndDimBindId = "cudaIdxAndDim";

  /// The container of refactoring replacements
  FileReplacementsMap &Repls;
  /// The map of new blockIdx, gridDim declarations
  std::string &TmpBlockInfoString;

public:
  /// Get block information declarations' reference matcher
  clang::ast_matchers::StatementMatcher getBlockInfoMatcher(std::string &KName);
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  CUDABlockInfoRewriter(FileReplacementsMap &OtherRepls, std::string &OtherInfoString)
                       : Repls{OtherRepls}, TmpBlockInfoString{OtherInfoString} {}
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
  /// The map of threads' number
  std::map<std::string, int> &ThreadNumMap;

public:
  /// Get synchronization matcher
  clang::ast_matchers::StatementMatcher getSyncMatcher(std::string &KName);
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  CUDASyncRewriter(FileReplacementsMap &OtherRepls, std::map<std::string, int> &OtherThreadNumMap)
                  : Repls{OtherRepls}, ThreadNumMap{OtherThreadNumMap} {}
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