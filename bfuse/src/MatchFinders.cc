
#include <cstdlib>
#include <utility>
#include <string>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Core/Replacement.h"
#include "llvm/Support/raw_ostream.h"

#include "bfuse/Tools.h"
#include "bfuse/Utils.h"
#include "bfuse/MatchFinders.h"

using namespace std;
using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;
//---------------------------------------------------------------------------
namespace bfuse {
namespace match_finders {
//---------------------------------------------------------------------------
void MacroExpander::run(const MatchFinder::MatchResult& Result)
{
  ASTContext *Context    = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>("macro-expand");

  if (!FD || FD->isTemplateInstantiation()) {
    CHECK_ERROR("cannot find \"macro-expand\" match finder.");
  }

  string Impl;
  llvm::raw_string_ostream ImplStream(Impl);

  Stmt *Body = FD->getBody();
  Body->printPretty(ImplStream, nullptr, Context->getPrintingPolicy());
  ImplStream.flush();

  auto& SourceManager = Context->getSourceManager();

  string Path = FD->getLocation().printToString(SourceManager);
  Path        = Path.substr(0, Path.find_first_of(":"));
  auto Repl   = Replacement(SourceManager, Body, Impl);

  if (auto err = replacements[Path].add(Repl)) {
    llvm::errs() << "[bfuse ERROR]: failed to store replacements";
    exit(0);
  }
}
//---------------------------------------------------------------------------  
} // namespace match_finders
} // namespace bfuse
//---------------------------------------------------------------------------  