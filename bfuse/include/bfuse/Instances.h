
#pragma once

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include "llvm/Support/raw_ostream.h"

#include "bfuse/Bfuse.h"
#include "bfuse/Contexts.h"
//---------------------------------------------------------------------------
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
class FusionInstance {
private:
  /// The arguments to build compilation database
  OptionsParserArguments &Args;
  /// The fusion context
  contexts::FusionContext &Context;

public:
  /// Analyze the source code
  int analyze();
  /// Rewrite the source code to raw ostream
  int rewrite(llvm::raw_ostream &RawOstream);
  /// Test function for print function declations
  int printFunctionDeclExample() const;

  /// The constructor
  FusionInstance(OptionsParserArguments &OtherArgs, contexts::FusionContext &OtherContext)
                : Args{OtherArgs}, Context{OtherContext} {}

  /// Delete default constructor
  FusionInstance() = delete;
  /// Delete copy constructor
  FusionInstance(const FusionInstance &Other) = delete;
  /// Delete move constructor
  FusionInstance(FusionInstance &&Other) = delete;
  /// Delete copy assignment operator
  FusionInstance& operator=(const FusionInstance &Other) = delete;
  /// Delete move assignment operator
  FusionInstance& operator=(FusionInstance &&Other) = delete;
};
//---------------------------------------------------------------------------
} // tools
} // bfuse
//---------------------------------------------------------------------------