
#pragma once

#include <string>

#include "clang/Tooling/CommonOptionsParser.h"

#include "fuse/Contexts.h"
//---------------------------------------------------------------------------
namespace fuse {
namespace tools {
//---------------------------------------------------------------------------
class FusionTool {
private:
  /// The clang refactoring tool
  clang::tooling::CommonOptionsParser &OptionsParser_;

  /// The conatiner of fusion information
  contexts::FusionContext FContext_;

  /// The string result of the fused kernels
  std::string FuncStr_;

public:
  /// Rewrite Kernels at first
  int initiallyRewriteKernels();
  /// Extract specific delcarations out of compound statement
  int rewriteCompStmt();
  /// Rename function parameters
  int renameParameters();
  /// Rewrite the CUDA variables
  int rewriteCUDAVariables();
  /// Hoist shared memory variable declarations
  int hoistSharedDecls();
  // /// Rename shared memory variables
  // int renameSharedVariables();
  // /// Create fused function
  // int createFusedKernel();

  // /// Save fused function into disk
  // int saveFusedKernel(const contexts::AnalysisContext &AContext, const std::string &ResultPath);
  
  /// Test function for print function declations
  int printFuncDecl();

  /// The constructor
  explicit FusionTool(clang::tooling::CommonOptionsParser &OptionsParser, contexts::FusionContext &FContext)
                     : OptionsParser_{OptionsParser}, FContext_{FContext} {}

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
} // fuse
//---------------------------------------------------------------------------