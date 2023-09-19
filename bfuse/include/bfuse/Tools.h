
#pragma once

#include <string>

#include "clang/Tooling/CommonOptionsParser.h"

#include "bfuse/Bfuse.h"
#include "bfuse/Contexts.h"
//---------------------------------------------------------------------------
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
class FusionTool {
private:
  /// The clang refactoring tool
  clang::tooling::CommonOptionsParser &OptionsParser;
  /// The Fusion Context
  contexts::FusionContext &FContext;

public:
  /// Analyze function parameters
  int analyzeParameters(contexts::AnalysisContext &Analysis);
  /// Analyze threadIdx, blockIdx boundry and create branch condition
  int analyzeThreadBoundaries(contexts::AnalysisContext &Analysis);

  /// Rename function parameters
  int renameParameters(contexts::AnalysisContext &Analysis);
  /// Rewrite the source code
  int rewriteCUDAInfos(contexts::AnalysisContext &Analysis);
  /// Create fused function
  int createFunction(contexts::AnalysisContext &Analysis, std::string &FuncStr);
  /// Save fused function into disk
  int saveFunction(contexts::AnalysisContext &Analysis, std::string &FuncStr);
  /// Test function for print function declations
  int printFuncDeclExample() const;

  /// The constructor
  explicit FusionTool(clang::tooling::CommonOptionsParser &OtherOptionsParser,
                      contexts::FusionContext &OtherFContext)
                     : OptionsParser{OtherOptionsParser}, FContext{OtherFContext} {}

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