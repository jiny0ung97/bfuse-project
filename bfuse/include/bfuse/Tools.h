
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
  /// The fusion context
  contexts::FusionContext &Context;

public:
  /// Analyze the source code
  int analyze(contexts::AnalysisContext &Analysis);
  /// Renaming variables
  int rename(contexts::AnalysisContext &Analysis);
  /// Rewrite the source code
  int rewrite(contexts::AnalysisContext &Analysis);
  /// Create fused function
  int createFunction(contexts::AnalysisContext &Analysis, std::string &FuncStr);
  /// Test function for print function declations
  int printFunctionDeclExample() const;

  /// The constructor
  FusionTool(clang::tooling::CommonOptionsParser &OtherOptionsParser,
             contexts::FusionContext &OtherContext)
            : OptionsParser{OtherOptionsParser}, Context{OtherContext} {}

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