
#pragma once

#include <string>

#include "llvm/Support/raw_ostream.h"

#include "bfuse/Bfuse.h"
#include "bfuse/Contexts.h"
//---------------------------------------------------------------------------
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
class FusionRewriteTool {
private:
  /// The arguments to build compilation database
  OptionsParserArguments &Args;
  /// The fusion context
  contexts::FusionContext &Context;

public:
  /// Analyze the source code
  int analyze(contexts::AnalyzeContext &Analysis);
  /// Rewrite the source code to raw ostream
  int rewrite(contexts::AnalyzeContext &Analysis, llvm::raw_ostream &RawOstream);
  /// Test function for print function declations
  int printFunctionDeclExample() const;

  /// The constructor
  FusionRewriteTool(OptionsParserArguments &OtherArgs, contexts::FusionContext &OtherContext)
                : Args{OtherArgs}, Context{OtherContext} {}

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
  int createFunctionFromCode(llvm::raw_string_ostream &RawString);
  /// Write the fused function to file
  int write(std::string &FilePath);
};
//---------------------------------------------------------------------------
} // tools
} // bfuse
//---------------------------------------------------------------------------