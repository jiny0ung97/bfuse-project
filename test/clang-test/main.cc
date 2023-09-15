

#include <cstdlib>
#include <iostream>
#include <memory>

#include "clang/AST/ASTImporter.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"

using namespace std;
using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;
//---------------------------------------------------------------------------
template <typename Node, typename Matcher>
Node* getFirstDecl(Matcher M, const unique_ptr<ASTUnit>& Unit)
{
  auto MB = M.bind("bindStr");
  auto MatchRes = match(MB, Unit->getASTContext());
  assert(MatchRes.size() >= 1);

  Node *Result = const_cast<Node *>(MatchRes[0].template getNodeAs<Node>("bindStr"));
  assert(Result);

  return Result;
}
//---------------------------------------------------------------------------
int codeASTImporterExample()
{
  unique_ptr<ASTUnit> ToUnit = buildASTFromCode("", "to.cc");
  unique_ptr<ASTUnit> FromUnit = buildASTFromCode(
    R"(
      int function(int a)
      {
        return a + 10;
      }
    )",
    "from.cc");

  auto Matcher = functionDecl(hasName("function"));
  auto *From   = getFirstDecl<FunctionDecl>(Matcher, FromUnit);

  ASTImporter Importer(ToUnit->getASTContext(), ToUnit->getFileManager(),
                       FromUnit->getASTContext(), FromUnit->getFileManager(),
                       /*MinimalImport=*/true);

  llvm::Expected<Decl *> ImportedOrErr = Importer.Import(From);
  if (!ImportedOrErr) {
    llvm::Error Err = ImportedOrErr.takeError();
    llvm::errs() << "ERROR: " << Err << "\n";
    llvm::consumeError(move(Err));
  }

  Decl *Imported = *ImportedOrErr;
  Imported->getTranslationUnitDecl()->dump();

  // if (llvm::Error Err = Importer.ImportDefinition(From)) {
  //   llvm::errs() << "ERROR: " << Err << "\n";
  //   llvm::consumeError(move(Err));
  //   exit(0);
  // }

  // llvm::errs() <<"Imported definition.\n";
  // Imported->getTranslationUnitDecl()->dump();

}
//---------------------------------------------------------------------------
void printTranslationUnitDecl(const TranslationUnitDecl& TU, const StringRef FName)
{
  std::error_code EC;
  llvm::raw_fd_ostream OS{FName, EC};
  TU.print(OS);
  OS.close();
}
//---------------------------------------------------------------------------
template <typename Node>
void rewriteAndSaveASTUnit(const ASTUnit& Unit, const StringRef FName, const Node* SomeNode)
{
  auto& SM = Unit.getSourceManager();
  auto& LO = Unit.getLangOpts();
  Rewriter R{SM, LO};

  R.ReplaceText(SomeNode->getSourceRange(), "fn_" + "();");

  std::error_code EC;
  llvm::raw_fd_ostream OS{FName, EC};
  R.getEditBuffer(SM.getMainFileID()).write(OS);
  OS.close();
}
//---------------------------------------------------------------------------
int main()
{
  string FuncName1 = "matmul";
  string FuncName2 = "conv2d";
  string NewFuncName = FuncName1 + "_" + FuncName2 + "_fused";
  string SourceCode = "void " + NewFuncName + "();";
  unique_ptr<ASTUnit> Unit = buildASTFromCode(SourceCode, "tmp.cc");

  auto &Context = Unit->getASTContext();
  auto *TU      = Context.getTranslationUnitDecl();

  auto Matcher = functionDecl(hasName(NewFuncName));
  auto *D = getFirstDecl<FunctionDecl>(Matcher, Unit);

  auto& SM = Unit->getSourceManager();
  auto& LO = Unit->getLangOpts();
  Rewriter R{SM, LO};

  R.ReplaceText(D->getSourceRange(), "fn_" + NewFuncName + "();");

  std::error_code EC;
  llvm::raw_fd_ostream OS{"test.cc", EC};
  R.getEditBuffer(SM.getMainFileID()).write(OS);
  OS.close();

  // Use previous code from HFuse-project...
  // 1. Bring Matcher Classes
  // 2. Apply changes...
  // 3. And make new ASTUnit from Code && Get TranslationUnitDecl
  // 4. Add new function using addDecl()
  // 5. Add that function into the new file

  return 0;
}
//---------------------------------------------------------------------------