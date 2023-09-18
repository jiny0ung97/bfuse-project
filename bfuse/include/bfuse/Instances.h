
#pragma once

#include <string>
#include <map>

#include "clang/Tooling/CommonOptionsParser.h"

#include "bfuse/Bfuse.h"
#include "bfuse/Contexts.h"
//---------------------------------------------------------------------------
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
class FusionRewriteTool {
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
  /// Test function for print function declations
  int printFunctionDeclExample() const;

  /// The constructor
  FusionRewriteTool(clang::tooling::CommonOptionsParser &OtherOptionsParser,
                    contexts::FusionContext &OtherContext)
                   : OptionsParser{OtherOptionsParser}, Context{OtherContext} {}

  /// Delete default constructor
  FusionRewriteTool() = delete;
  /// Delete copy constructor
  FusionRewriteTool(const FusionRewriteTool &Other) = delete;
  /// Delete move constructor
  FusionRewriteTool(FusionRewriteTool &&Other) = delete;
  /// Delete copy assignment operator
  FusionRewriteTool& operator=(const FusionRewriteTool &Other) = delete;
  /// Delete move assignment operator
  FusionRewriteTool& operator=(FusionRewriteTool &&Other) = delete;
};
//---------------------------------------------------------------------------
class FusionBuildTool {
private:
  /// 

public:
  /// Create new fused function from raw string
  int createFunctionFromCode();
  /// Write the fused function to file
  int write(std::string &FilePath);
};
//---------------------------------------------------------------------------
} // tools
} // bfuse
//---------------------------------------------------------------------------