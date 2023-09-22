
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
using FileReplacementsMapTy = std::map<std::string, clang::tooling::Replacements>;
using VarListTy      = contexts::VarListTy;
using USRsListTy     = contexts::USRsListTy;
using SizeListTy     = contexts::SizeListTy;
using VarListMapTy   = contexts::VarListMapTy;
using USRsListMapTy  = contexts::USRsListMapTy;
using SizeListMapTy  = contexts::SizeListMapTy;
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
  FileReplacementsMapTy &Repls;

public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  explicit CUDAKernelRewriter(FileReplacementsMapTy &OtherRepls) : Repls{OtherRepls} {}
};
//---------------------------------------------------------------------------
class CUDACompStmtRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The container of refactoring replacements
  FileReplacementsMapTy &Repls;
  /// The temp blockIdx, gridDim declarations
  const std::string &TmpBlockInfoString;

public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  explicit CUDACompStmtRewriter(FileReplacementsMapTy &OtherRepls,
                                const std::string &OtherTmpBlockInfoString)
                               : Repls{OtherRepls}, TmpBlockInfoString{OtherTmpBlockInfoString} {}
};
//---------------------------------------------------------------------------
class CUDAFuncParmAnalyzer
      : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  /// The map of function parameters' list
  VarListMapTy ParmListMap;
  /// The map of USRs lists for renaming parameters
  USRsListMapTy ParmUSRsListMap;

  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
};
//---------------------------------------------------------------------------
class CUDABlockInfoRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The container of refactoring replacements
  FileReplacementsMapTy &Repls;

public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  CUDABlockInfoRewriter(FileReplacementsMapTy &OtherRepls) : Repls{OtherRepls} {}
};
//---------------------------------------------------------------------------
class CUDASyncRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The container of refactoring replacements
  FileReplacementsMapTy &Repls;
  /// The map of threads' number
  const std::map<std::string, int> &ThreadNumMap;

public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  CUDASyncRewriter(FileReplacementsMapTy &OtherRepls, const std::map<std::string, int> &OtherThreadNumMap)
                  : Repls{OtherRepls}, ThreadNumMap{OtherThreadNumMap} {}
};
//---------------------------------------------------------------------------
class CUDASharedDeclExtractor
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The container of refactoring replacements
  FileReplacementsMapTy &Repls;

public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  CUDASharedDeclExtractor(FileReplacementsMapTy &OtherRepls) : Repls{OtherRepls} {}
};
//---------------------------------------------------------------------------
class CUDASharedDeclRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The container of refactoring replacements
  FileReplacementsMapTy &Repls;
    /// The kernels to be fused
  const std::vector<std::string> &Kernels;
  /// The map of ASTContexts
  std::map<std::string, clang::ASTContext *> ASTContextMap;
  /// The map of source locations
  std::map<std::string, clang::SourceLocation> SourceLocMap;
  /// The map of shared memory declarations
  std::map<std::string, std::string> SharedDeclStringMap;

public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
  /// Run finder at the end of the translation unit
  virtual void onEndOfTranslationUnit() override;

  /// The constructor
  CUDASharedDeclRewriter(FileReplacementsMapTy &OtherRepls,
                         const std::vector<std::string> &OtherKernels)
                        : Repls{OtherRepls}, Kernels{OtherKernels} {}
};
//---------------------------------------------------------------------------
class CUDASharedVarAnalyzer
      : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  /// The map of shared memory variables' list
  VarListMapTy ShrdVarListMap;
  /// The map of USRs lists for renaming shared memory variables
  USRsListMapTy ShrdVarUSRsListMap;
  /// The map of shared memory variables' size
  SizeListMapTy ShrdVarSizeListMap;

  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
};
//---------------------------------------------------------------------------
class CUDAFuncBuilder
      : public clang::ast_matchers::MatchFinder::MatchCallback {
private:
  /// The analysis of functions to be fused
  const contexts::AnalysisContext &Analysis;
  /// The string stream of fused function
  llvm::raw_string_ostream FuncStream;
  /// The list of functions to be fused
  std::map<std::string, std::string> FuncBodyStringMap;
  /// The string list of parameters
  std::vector<std::string> ParmStringList;

public:
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
  /// Run finder at the end of the translation unit
  virtual void onEndOfTranslationUnit() override;

  /// The constructor
  CUDAFuncBuilder(const contexts::AnalysisContext &OtherAnalysis, std::string &FuncStr)
                 : Analysis{OtherAnalysis}, FuncStream{FuncStr} {}
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
} // namespace bfuse
//---------------------------------------------------------------------------