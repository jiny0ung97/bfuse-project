
#pragma once

#include <string>

#include "clang/Tooling/CommonOptionsParser.h"

#include "bfuse/Contexts.h"
//---------------------------------------------------------------------------
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
class FusionTool {
private:
  /// The clang refactoring tool
  clang::tooling::CommonOptionsParser &OptionsParser;

  /// The string result of the fused kernels
  std::string FuncStr;

public:
  /// Extract specific delcarations out of compound statement
  int initiallyRewriteKernels(contexts::AnalysisContext &AContext);
  /// Analyze function parameters
  int analyzeParameters(contexts::AnalysisContext &AContext);
  /// Rename function parameters
  int renameParameters(contexts::AnalysisContext &AContext);
  /// Rewrite the source code
  int rewriteCUDAVariables(contexts::AnalysisContext &AContext);
  /// Create fused function
  int createFusedKernel(contexts::AnalysisContext &AContext);
  /// Save fused function into disk
  int saveFusedKernel(contexts::AnalysisContext &AContext, const std::string &ResultPath);
  /// Test function for print function declations
  int printFuncDecl(contexts::AnalysisContext &AContext);

  /// The constructor
  explicit FusionTool(clang::tooling::CommonOptionsParser &OtherOptionsParser)
                     : OptionsParser{OtherOptionsParser} {}

  /// Delete default constructor
  FusionTool() = delete;
  /// Delete copy constructor
  FusionTool(const FusionTool &Other) = delete;
  /// Delete move constructor
  FusionTool(FusionTool &&Other) = delete;
  /// Delete copy assignment operator
  FusionTool& operator=(const FusionTool &Other) = delete;
  /// Delete move assignment operator
  FusionTool& operator=(FusionTool &&Other) = delete;
};
//---------------------------------------------------------------------------
} // tools
} // bfuse
//---------------------------------------------------------------------------