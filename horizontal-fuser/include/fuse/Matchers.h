
#pragma once

#include <string>
#include <map>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Core/Replacement.h"

#include "fuse/Contexts.h"
//---------------------------------------------------------------------------
namespace fuse {
namespace matchers {
//---------------------------------------------------------------------------
class ASTPatternMatcher {
public:
  /// Get function declaration matcher
  static clang::ast_matchers::DeclarationMatcher getFuncDeclMatcher(const std::string &KName);
  /// Get function parameters matcher
  static clang::ast_matchers::DeclarationMatcher getFuncParmMatcher(const std::string &KName);
  /// Get block information declarations' reference matcher
  static clang::ast_matchers::StatementMatcher getBlockIdxMatcher(const std::string &KName);
  /// Get synchronization matcher
  static clang::ast_matchers::StatementMatcher getSyncMatcher(const std::string &KName);
  /// Get shared memory declaration matcher
  static clang::ast_matchers::StatementMatcher getSharedDeclMatcher(const std::string &KName);
  /// Get function declaration matcher
  static clang::ast_matchers::DeclarationMatcher getFuncBuildMatcher(const std::string &KName);

  /// The cuda function declaration bind id
  static const std::string CUDAFuncDecl;
  /// The cuda function parameters bind id
  static const std::string CUDAFuncParm;
  /// The cuda block information member (x, y, z) bind id
  static const std::string CUDABlockIdxVarMember;
  /// The cuda block information bind id
  static const std::string CUDABlockIdxVar;
  /// The cuda synchronization bind id
  static const std::string CUDASync;
  /// The shared memory variable declaration bind id
  static const std::string CUDASharedDecl;
  /// The shared memory variable bind id
  static const std::string CUDASharedVar;
};
//---------------------------------------------------------------------------
class CUDAKernelRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The container of refactoring replacements
  std::map<std::string, clang::tooling::Replacements> &Repls_;
  /// The map of visited functions
  std::map<std::string, bool> IsVisitedMap_;

public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  explicit CUDAKernelRewriter(std::map<std::string, clang::tooling::Replacements> &Repls) : Repls_{Repls} {}
};
//---------------------------------------------------------------------------
class CUDACompStmtRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The container of refactoring replacements
  std::map<std::string, clang::tooling::Replacements> &Repls_;
  /// The map of visited functions
  std::map<std::string, bool> IsVisitedMap_;

public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  explicit CUDACompStmtRewriter(std::map<std::string, clang::tooling::Replacements> &Repls) : Repls_{Repls} {}
};
//---------------------------------------------------------------------------
class CUDAFuncParmAnalyzer
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The map of visited functions
  std::map<std::string, bool> IsVisitedMap_;

public:
  /// The map of function parameters' list
  std::map<std::string, std::vector<std::string>> ParmListMap_;
  /// The map of USRs lists for renaming parameters
  std::map<std::string, std::vector<std::vector<std::string>>> ParmUSRsListMap_;

  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
};
//---------------------------------------------------------------------------
class CUDABlockInfoRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The container of refactoring replacements
  std::map<std::string, clang::tooling::Replacements> &Repls_;

public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  CUDABlockInfoRewriter(std::map<std::string, clang::tooling::Replacements> &Repls) : Repls_{Repls} {}
};
//---------------------------------------------------------------------------
class CUDASyncRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The container of refactoring replacements
  std::map<std::string, clang::tooling::Replacements> &Repls_;
  /// The map of kernel's information
  contexts::FusionContext FContext_;

public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  CUDASyncRewriter(std::map<std::string, clang::tooling::Replacements> &Repls, const contexts::FusionContext FContext)
                  : Repls_{Repls}, FContext_{FContext} {}
};
//---------------------------------------------------------------------------
class CUDASharedDeclExtractor
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The container of refactoring replacements
  std::map<std::string, clang::tooling::Replacements> &Repls_;

public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  CUDASharedDeclExtractor(std::map<std::string, clang::tooling::Replacements> &Repls) : Repls_{Repls} {}
};
//---------------------------------------------------------------------------
class CUDASharedDeclRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The container of refactoring replacements
  std::map<std::string, clang::tooling::Replacements> &Repls_;
  /// The kernels to be fused
  std::vector<std::string> Kernels_;
  /// The map of ASTContexts
  std::map<std::string, clang::ASTContext *> ASTContextMap_;
  /// The map of source locations
  std::map<std::string, clang::SourceLocation> SourceLocMap_;
  /// The map of shared memory declarations
  std::map<std::string, std::vector<std::string>> SharedDeclStrMap_;

public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
  /// Run finder at the end of the translation unit
  virtual void onEndOfTranslationUnit() override;

  /// The constructor
  CUDASharedDeclRewriter(std::map<std::string, clang::tooling::Replacements> &Repls) : Repls_{Repls} {}
};
//---------------------------------------------------------------------------
class CUDASharedVarAnalyzer
      : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  /// The map of shared memory variables' list
  std::map<std::string, std::vector<std::string>> ShrdVarListMap_;
  /// The map of USRs lists for renaming shared memory variables
  std::map<std::string, std::vector<std::vector<std::string>>> ShrdVarUSRsListMap_;
  /// The map of shared memory declarations
  std::map<std::string, std::vector<std::string>> SharedDeclStrMap_;

public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
};
//---------------------------------------------------------------------------
class BFuseBuilder
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// Fusion information
  contexts::FusionContext FContext_;
  /// New shared variable declarations
  std::string UnionStr_;
  /// The string stream of fused function
  llvm::raw_string_ostream FuncStream_;
  /// The list of functions to be fused
  std::map<std::string, std::string> FuncBodyStringMap_;
  /// Check whether at least one of the functions have template parameters
  bool IsFuncTemplate_ = false;
  /// The string list of template parameters
  std::vector<std::string> TemplStringList_;
  /// The string list of parameters
  std::vector<std::string> ParmStringList_;

public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
  /// Run finder at the end of the translation unit
  virtual void onEndOfTranslationUnit() override;

  /// The constructor
  BFuseBuilder(const contexts::FusionContext &FContext, const std::string &UnionStr, std::string &FuncStr) : FContext_{FContext}, UnionStr_{UnionStr}, FuncStream_{FuncStr} {}
};
//---------------------------------------------------------------------------
class HFuseBuilder
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// Fusion information
  contexts::FusionContext FContext_;
  /// The string stream of fused function
  llvm::raw_string_ostream FuncStream_;
  /// The list of functions to be fused
  std::map<std::string, std::string> FuncBodyStringMap_;
  /// Check whether at least one of the functions have template parameters
  bool IsFuncTemplate_ = false;
  /// The string list of template parameters
  std::vector<std::string> TemplStringList_;
  /// The string list of parameters
  std::vector<std::string> ParmStringList_;

public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
  /// Run finder at the end of the translation unit
  virtual void onEndOfTranslationUnit() override;

  /// The constructor
  HFuseBuilder(const contexts::FusionContext &FContext, std::string &FuncStr) : FContext_{FContext}, FuncStream_{FuncStr} {}
};
//---------------------------------------------------------------------------
class CUDAFuncDeclPrinter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
};
//---------------------------------------------------------------------------
} // namespace matchers
} // namespace fuse
//---------------------------------------------------------------------------