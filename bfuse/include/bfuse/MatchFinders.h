
#pragma once

#include <string>
#include <vector>
#include <map>
#include <unordered_map>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Core/Replacement.h"

#include "bfuse/Tools.h"
//---------------------------------------------------------------------------
namespace bfuse {
namespace match_finders {
//---------------------------------------------------------------------------
const std::string macroExpandMatcher = "macro-expand";
//---------------------------------------------------------------------------
class MacroExpander : public clang::ast_matchers::MatchFinder::MatchCallback {
public:
  using ClangReplacementsMap = std::map<std::string, clang::tooling::Replacements>;
  using KernelContextMap     = std::unordered_map<std::string, tools::KernelContext>;

private:
  /// The map to contain clang replacements
  ClangReplacementsMap& replacements;
  /// The kernel contexts to be expaneded
  KernelContextMap& kernelContexts;

public:
  /// The main function of this match finder
  void run(const clang::ast_matchers::MatchFinder::MatchResult& Result) override;

  /// The constructor
  explicit MacroExpander(ClangReplacementsMap& Replacements, KernelContextMap& KernelContexts)
                        : replacements{Replacements}, kernelContexts{KernelContexts} {}
};
//---------------------------------------------------------------------------
} // namesapce match_finders
} // namesapce bfuse
//---------------------------------------------------------------------------