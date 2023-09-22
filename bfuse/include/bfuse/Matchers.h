
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
  clang::ast_matchers::DeclarationMatcher getFuncDeclMatcher(const std::string &KName);
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
};
//---------------------------------------------------------------------------
class CUDADeclRewriter
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
  clang::ast_matchers::DeclarationMatcher getFuncDeclMatcher(const std::string &KName);
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  explicit CUDADeclRewriter(FileReplacementsMap &OtherRepls) : Repls{OtherRepls} {}
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
  clang::ast_matchers::DeclarationMatcher getFuncDeclMatcher(const std::string &KName);
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  explicit CUDADeclExtractor(FileReplacementsMap &OtherRepls) : Repls{OtherRepls} {}
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
  std::map<std::string, ParamList> ParmListMap;
  /// The map of USRs lists for renaming parameters
  std::map<std::string, USRsList> ParmUSRsListMap;

  /// Get function parameters matcher
  clang::ast_matchers::DeclarationMatcher getFuncParamMatcher(const std::string &Kname);
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
  /// The map of new blockIdx, gridDim declarations
  std::string TmpBlockInfoString;

public:
  /// Get block information declarations' reference matcher
  clang::ast_matchers::StatementMatcher getBlockInfoMatcher(const std::string &KName);
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  CUDABlockInfoRewriter(FileReplacementsMap &OtherRepls, const std::string &OtherInfoString)
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
  std::map<std::string, int> ThreadNumMap;

public:
  /// Get synchronization matcher
  clang::ast_matchers::StatementMatcher getSyncMatcher(const std::string &KName);
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  CUDASyncRewriter(FileReplacementsMap &OtherRepls, const std::map<std::string, int> &OtherThreadNumMap)
                  : Repls{OtherRepls}, ThreadNumMap{OtherThreadNumMap} {}
};
//---------------------------------------------------------------------------
class CUDASharedDeclExtractor
      : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  using FileReplacementsMap = std::map<std::string, clang::tooling::Replacements>;

private:
  /// The cuda function declaration bind id
  const std::string CUDAFuncDeclBindId = "cudaFuncDecl";
  /// The shared memory variable declaration bind id
  const std::string CUDASharedDeclBindId = "cudaSharedDecl";

  /// The container of refactoring replacements
  FileReplacementsMap &Repls;

public:
  /// The string of shared memory declarations
  std::map<std::string, std::string> SharedDeclStringMap;

public:
  /// Get shared memory declaration matcher
  clang::ast_matchers::StatementMatcher getSharedDeclMatcher(const std::string &KName);
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;

  /// The constructor
  CUDASharedDeclExtractor(FileReplacementsMap &OtherRepls) : Repls{OtherRepls} {}
};
//---------------------------------------------------------------------------
class CUDASharedDeclRewriter
      : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  using FileReplacementsMap = std::map<std::string, clang::tooling::Replacements>;

private:
  /// The cuda function declaration bind id
  const std::string CUDAFuncDeclBindId = "cudaFuncDecl";
  /// The shared memory variable declaration bind id
  const std::string CUDASharedDeclBindId = "cudaSharedDecl";

  /// The container of refactoring replacements
  FileReplacementsMap &Repls;
  /// The string of shared memory declarations
  std::map<std::string, std::string> SharedDeclStringMap;

  /// [Temp]
  // std::string SharedDeclString = "\n";
  std::vector<std::string> Kernels;
  std::map<std::string, clang::ASTContext *> ASTContextMap;
  std::map<std::string, clang::SourceLocation> SourceLocMap;

public:
  // /// Get function declaration matcher
  // clang::ast_matchers::DeclarationMatcher getFuncDeclMatcher(const std::string &KName);
  /// Get shared memory declaration matcher
  clang::ast_matchers::StatementMatcher getSharedDeclMatcher(const std::string &KName);
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
  /// Run finder at the end of the translation unit
  virtual void onEndOfTranslationUnit() override;

  /// The constructor
  CUDASharedDeclRewriter(FileReplacementsMap &OtherRepls,
                         const std::map<std::string, std::string> &OtherSharedDeclStringMap,
                         const std::vector<std::string> &OtherKernels)
                        : Repls{OtherRepls}, SharedDeclStringMap{OtherSharedDeclStringMap}, Kernels{OtherKernels} {}
};
//---------------------------------------------------------------------------
class CUDASharedVarAnalyzer
      : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  using ShrdVarList      = std::vector<std::string>;
  using ShrdVarUSRsList  = std::vector<std::vector<std::string>>;
  using ShrdVarSizeList  = std::vector<uint64_t>;

private:
  /// The cuda function declaration bind id
  const std::string CUDAFuncDeclBindId = "cudaFuncDecl";
  /// The shared memory variable declaration bind id
  const std::string CUDASharedDeclBindId = "cudaSharedDecl";
  /// The shared memory variable  bind id
  const std::string CUDASharedVarBindId = "cudaSharedVar";

public:
  /// The map of shared memory variables' list
  std::map<std::string, ShrdVarList> ShrdVarListMap;
  /// The map of USRs lists for renaming shared memory variables
  std::map<std::string, ShrdVarUSRsList> ShrdVarUSRsListMap;
  /// The map of shared memory variables' size
  std::map<std::string, ShrdVarSizeList> ShrdVarSizeListMap;

  /// Get shared memory declaration matcher
  clang::ast_matchers::StatementMatcher getSharedDeclMatcher(const std::string &KName);
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
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
  const contexts::AnalysisContext &Analysis;
  /// The string stream of fused function
  llvm::raw_string_ostream FuncStream;
  /// The list of functions to be fused
  std::map<std::string, std::string> FuncBodyStringMap;
  /// The string list of parameters
  std::vector<std::string> ParmStringList;

public:
  /// Get function declaration matcher
  clang::ast_matchers::DeclarationMatcher getFuncBuildMatcher(const std::string &KName);
  /// Run AST match finder
  virtual void run(const clang::ast_matchers::MatchFinder::MatchResult &Result) override;
  /// Run finder at the end of the translation unit
  virtual void onEndOfTranslationUnit() override;

  /// The constructor
  CUDAFuncBuilder(const contexts::AnalysisContext &OtherAnalysis, std::string &FuncStr)
                 : Analysis{OtherAnalysis}, FuncStream{FuncStr} {}
};
//---------------------------------------------------------------------------
} // namespace matchers
} // namespace bfuse
//---------------------------------------------------------------------------