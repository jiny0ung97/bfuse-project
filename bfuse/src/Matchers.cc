
#include <cstdlib>
#include <numeric>
#include <string>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Refactoring/Rename/USRFindingAction.h"

#include "llvm/Support/raw_ostream.h"

#include "bfuse/Contexts.h"
#include "bfuse/Utils.h"
#include "bfuse/Matchers.h"

using namespace std;

using namespace bfuse::contexts;

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;
//---------------------------------------------------------------------------
namespace bfuse {
namespace matchers {
//---------------------------------------------------------------------------
DeclarationMatcher CUDAFuncDeclPrinter::getFuncDeclMatcher(const string &KName)
{
  return functionDecl(
           hasAttr(attr::CUDAGlobal),
           hasName(KName)
         ).bind(CUDAFuncDeclBindId);
}
//---------------------------------------------------------------------------
void CUDAFuncDeclPrinter::run(const MatchFinder::MatchResult &Result)
{
  // ASTContext    *Context = Result.Context;
  const FunctionDecl* FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);
  if (!FD) {
    return;
  }

  FD->dump();
}
//---------------------------------------------------------------------------
DeclarationMatcher CUDADeclRewriter::getFuncDeclMatcher(const string &KName)
{
  return functionDecl(
           hasAttr(attr::CUDAGlobal),
           hasName(KName)
         ).bind(CUDAFuncDeclBindId);
}
//---------------------------------------------------------------------------
void CUDADeclRewriter::run(const MatchFinder::MatchResult& Result)
{
  ASTContext *Context = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);  
  if (!FD) {
    return;
  }

  auto &SourceMgr = Context->getSourceManager();
  auto *FuncBody  = FD->getBody();

  string BodyStr;
  llvm::raw_string_ostream BodyStream{BodyStr};

  FuncBody->printPretty(BodyStream, nullptr, Context->getPrintingPolicy());
  BodyStream.flush();

  Replacement BodyRepl{SourceMgr, FuncBody, BodyStr};
  // string Path = FD->getLocation().printToString(SourceMgr); // Should take a detour
  // Path = Path.substr(0, Path.find_first_of(":"));
  // if (auto Err = Repls[Path].add(BodyRepl)) {
  //   llvm::errs() << "CUDADeclRewriter error occur\n";
  //   exit(0);
  // }
  string FilePath = BodyRepl.getFilePath().str();
  if (auto Err = Repls[FilePath].add(BodyRepl)) {
    llvm::errs() << "CUDADeclRewriter error occur\n";
    exit(0);
  }

  // TODO: need to validate that kernels are all existed
}
//---------------------------------------------------------------------------
DeclarationMatcher CUDADeclExtractor::getFuncDeclMatcher(const string &KName)
{
  return functionDecl(
           hasAttr(attr::CUDAGlobal),
           hasName(KName)
         ).bind(CUDAFuncDeclBindId);
}
//---------------------------------------------------------------------------
void CUDADeclExtractor::run(const MatchFinder::MatchResult& Result)
{
  ASTContext *Context = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);  
  if (!FD) {
    return;
  }

  // Append compound {}
  auto &SourceMgr     = Context->getSourceManager();
  auto SourceBeginLoc = FD->getBody()->getBeginLoc().getLocWithOffset(1);
  auto SourceEndLoc   = FD->getBody()->getEndLoc().getLocWithOffset(-1);

  string TmpBlockInfoString = "\n  int blockIdx_x_;\n  int gridDim_x_;\n";
  string CompBeginString    = "\n  {";
  string CompEndString      = "\n  }";

  Replacement CompBeginRepl{SourceMgr, SourceBeginLoc, 0, TmpBlockInfoString + CompBeginString};
  string CBFile = CompBeginRepl.getFilePath().str();
  if (auto Err = Repls[CBFile].add(CompBeginRepl)) {
    llvm::errs() << "CUDADeclExtractor-Begin error occur\n";
    exit(0);
  }

  Replacement CompEndRepl{SourceMgr, SourceEndLoc, 0, CompEndString};
  string CEFile = CompEndRepl.getFilePath().str();
  if (auto Err = Repls[CEFile].add(CompEndRepl)) {
    llvm::errs() << "CUDADeclExtractor-End error occur\n";
    exit(0);
  }
}
//---------------------------------------------------------------------------
DeclarationMatcher CUDAFuncParamAnalyzer::getFuncParamMatcher(const string &Kname)
{
  return parmVarDecl(
           hasAncestor(
             functionDecl(
               hasAttr(attr::CUDAGlobal),
               hasName(Kname)
             ).bind(CUDAFuncDeclBindId)
           )
         ).bind(CUDAFuncParamBindId);
}
//---------------------------------------------------------------------------
void CUDAFuncParamAnalyzer::run(const MatchFinder::MatchResult& Result)
{
  ASTContext *Context = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);
  const ParmVarDecl  *PD = Result.Nodes.getNodeAs<ParmVarDecl>(CUDAFuncParamBindId);
  if (!PD) {
    return;
  }

  auto FName    = FD->getNameAsString();
  auto PName    = PD->getNameAsString();
  auto ParamUSR = getUSRsForDeclaration(PD->getUnderlyingDecl(), *Context);

  ParmListMap[FName].push_back(PName);
  ParmUSRsListMap[FName].push_back(ParamUSR);
}
//---------------------------------------------------------------------------
StatementMatcher CUDABlockInfoRewriter::getBlockInfoMatcher(const string &KName)
{
  return memberExpr(
           hasObjectExpression(
             opaqueValueExpr(
               hasSourceExpression(
                 declRefExpr(
                   to(varDecl(
                     anyOf(
                       hasName("gridDim"),
                       hasName("blockIdx")
                     )).bind(CUDAIdxAndDimBindId)
           ))))),
           hasAncestor(
             functionDecl(
               hasAttr(attr::CUDAGlobal),
               hasName(KName)
             )
           )
         ).bind(CUDAIdxAndDimMemberBindId);
}
//---------------------------------------------------------------------------
void CUDABlockInfoRewriter::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context = Result.Context;
  const MemberExpr *ME = Result.Nodes.getNodeAs<MemberExpr>(CUDAIdxAndDimMemberBindId);
  const VarDecl    *VD = Result.Nodes.getNodeAs<VarDecl>(CUDAIdxAndDimBindId);
  if (!ME) {
    return;
  }

  // Rewrite thread informations
  map<string, string> MemberReNamingMap = {
    {"__fetch_builtin_x", "x"},
    // {"__fetch_builtin_y", "y"},
    // {"__fetch_builtin_z", "z"}
  };
  auto ReNamingFunc = [](string &Var, string &Member) { return Var + "_" + Member + "_"; };

  string MName    = ME->getMemberNameInfo().getAsString();
  string VName    = VD->getNameAsString();
  string NewMName = MemberReNamingMap.at(MName);
  string NewVName = ReNamingFunc(VName, NewMName);

  auto CharSrcRange = CharSourceRange::getTokenRange(ME->getSourceRange());
  auto& SourceMgr   = Context->getSourceManager();

  Replacement RenameRepl{SourceMgr, CharSrcRange, NewVName};
  string RFilePath = RenameRepl.getFilePath().str();
  if (auto Err = Repls[RFilePath].add(RenameRepl)) {
    llvm::errs() << "CUDABlockInfoRewriter error occur\n";
    exit(0);
  }
}
//---------------------------------------------------------------------------
StatementMatcher CUDASyncRewriter::getSyncMatcher(const string &KName)
{
  return callExpr(
           callee(
             functionDecl(
               hasName("__syncthreads")
           )),
           hasAncestor(
             functionDecl(
               hasAttr(attr::CUDAGlobal),
               hasName(KName)
             ).bind(CUDAFuncDeclBindId)
           )
         ).bind(CUDASyncBindId);
}
//---------------------------------------------------------------------------
void CUDASyncRewriter::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);
  const CallExpr     *CE = Result.Nodes.getNodeAs<CallExpr>(CUDASyncBindId);
  if (!CE) {
    return;
  }

  auto RefactoringFunc = [](int BlockDim) { return "asm(\"bar.sync 0, " + to_string(BlockDim) + ";\")"; };
  string FName  = FD->getNameAsString();
  int ThreadNum = ThreadNumMap.at(FName);

  string NewSync  = RefactoringFunc(ThreadNum);
  auto &SourceMgr = Context->getSourceManager();

  Replacement Repl{SourceMgr, CE, NewSync};
  string FilePath = Repl.getFilePath().str();
  if (auto Err = Repls[FilePath].add(Repl)) {
    llvm::errs() << "CUDASyncRewriter error occur\n";
    exit(0);
  }
}
//---------------------------------------------------------------------------
StatementMatcher CUDASharedDeclExtractor::getSharedDeclMatcher(const std::string &KName)
{
  return declStmt(
           hasSingleDecl(
             varDecl(
               hasAttr(attr::CUDAShared)
             )
           ),
           hasAncestor(
             functionDecl(
               hasAttr(attr::CUDAGlobal),
               hasName(KName)
             ).bind(CUDAFuncDeclBindId)
           )
         ).bind(CUDASharedDeclBindId);
}
//---------------------------------------------------------------------------
void CUDASharedDeclExtractor::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context = Result.Context;
  const DeclStmt *DS = Result.Nodes.getNodeAs<DeclStmt>(CUDASharedDeclBindId);
  if (!DS) {
    return;
  }

  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);
  auto KName = FD->getNameAsString();

  // Add declaration at front
  string DeclStr;
  llvm::raw_string_ostream DeclStream{DeclStr};

  DS->printPretty(DeclStream, nullptr, Context->getPrintingPolicy(), /*Indentation=*/1U);
  DeclStream.flush();

  if (SharedDeclStringMap.find(KName) == SharedDeclStringMap.end()) {
    SharedDeclStringMap[KName] = "\n";
  }
  auto &SharedDeclString = SharedDeclStringMap.at(KName);
  SharedDeclString += DeclStr;

  // Remove existing declaration
  auto &SourceMgr     = Context->getSourceManager();
  auto SourceBeginLoc = DS->getBeginLoc();

  Replacement RemoveRepl{SourceMgr, SourceBeginLoc, 0, "// "};
  string RFPath = RemoveRepl.getFilePath().str();
  if (auto Err = Repls[RFPath].add(RemoveRepl)) {
    llvm::errs() << "CUDASharedDeclExtractor error occur\n";
    exit(0);
  }
}
//---------------------------------------------------------------------------
// DeclarationMatcher CUDASharedDeclRewriter::getFuncDeclMatcher(const string &KName)
// {
//   return functionDecl(
//            hasAttr(attr::CUDAGlobal),
//            hasName(KName)
//          ).bind(CUDAFuncDeclBindId);
// }
//---------------------------------------------------------------------------
StatementMatcher CUDASharedDeclRewriter::getSharedDeclMatcher(const std::string &KName)
{
  return declStmt(
           hasSingleDecl(
             varDecl(
               hasAttr(attr::CUDAShared)
             )
           ),
           hasAncestor(
             functionDecl(
               hasAttr(attr::CUDAGlobal),
               hasName(KName)
             ).bind(CUDAFuncDeclBindId)
           )
         ).bind(CUDASharedDeclBindId);
}
//---------------------------------------------------------------------------
// void CUDASharedDeclRewriter::run(const MatchFinder::MatchResult &Result)
// {
//   ASTContext *Context = Result.Context;
//   const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);
//   if (!FD) {
//     return;
//   }

//   auto &SourceMgr = Context->getSourceManager();
//   auto SourceLoc  = FD->getBody()->getBeginLoc().getLocWithOffset(1);
//   auto FName = FD->getNameAsString();
//   auto &SharedDeclString = SharedDeclStringMap.at(FName);

//   Replacement Repl{SourceMgr, SourceLoc, 0, SharedDeclString};
//   string FilePath = Repl.getFilePath().str();
//   if (auto Err = Repls[FilePath].add(Repl)) {
//     llvm::errs() << "CUDASharedDeclRewriter-Remove error occur\n";
//     exit(0);
//   }
// }
//---------------------------------------------------------------------------
void CUDASharedDeclRewriter::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context = Result.Context;
  const DeclStmt *DS = Result.Nodes.getNodeAs<DeclStmt>(CUDASharedDeclBindId);
  if (!DS) {
    return;
  }

  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);
  auto FName = FD->getNameAsString();

  // Add declaration at front
  string DeclStr;
  llvm::raw_string_ostream DeclStream{DeclStr};

  DS->printPretty(DeclStream, nullptr, Context->getPrintingPolicy(), /*Indentation=*/1U);
  DeclStream.flush();


  if (SharedDeclStringMap.find(FName) == SharedDeclStringMap.end()) {
    SharedDeclStringMap[FName] = "\n";
  }
  auto &SharedDeclString = SharedDeclStringMap.at(FName);
  SharedDeclString += DeclStr;

  if (ASTContextMap.find(FName) == ASTContextMap.end()) {
    ASTContextMap[FName] = Context;
  }

  auto SourceLoc = FD->getBody()->getBeginLoc().getLocWithOffset(1);
  if (SourceLocMap.find(FName) == SourceLocMap.end()) {
    SourceLocMap[FName] = SourceLoc;
  }

  // auto &SourceMgr = Context->getSourceManager();

  // Replacement Repl{SourceMgr, SourceLoc, 0, DeclStr};
  // string FilePath = Repl.getFilePath().str();
  // if (auto Err = Repls[FilePath].add(Repl)) {
  //   llvm::errs() << "CUDASharedDeclRewriter error occur\n";
  //   exit(0);
  // }
}
//---------------------------------------------------------------------------
void CUDASharedDeclRewriter::onEndOfTranslationUnit()
{
  for (auto &KName : Kernels) {
    auto *Context   = ASTContextMap.at(KName);
    auto &SourceLoc = SourceLocMap.at(KName);
    auto &ShrdDeclStr = SharedDeclStringMap.at(KName);

    Replacement Repl{Context->getSourceManager(), SourceLoc, 0, ShrdDeclStr};
    string FilePath = Repl.getFilePath().str();
    if (auto Err = Repls[FilePath].add(Repl)) {
      llvm::errs() << "CUDASharedDeclRewriter error occur\n";
      exit(0);
    }
  }
}
//---------------------------------------------------------------------------
StatementMatcher CUDASharedVarAnalyzer::getSharedDeclMatcher(const std::string &KName)
{
  return declStmt(
           hasSingleDecl(
             varDecl(
               hasAttr(attr::CUDAShared)
             ).bind(CUDASharedVarBindId)
           ),
           hasAncestor(
             functionDecl(
               hasAttr(attr::CUDAGlobal),
               hasName(KName)
             ).bind(CUDAFuncDeclBindId)
           )
         ).bind(CUDASharedDeclBindId);
}
//---------------------------------------------------------------------------
void CUDASharedVarAnalyzer::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context = Result.Context;
  const DeclStmt *DS  = Result.Nodes.getNodeAs<DeclStmt>(CUDASharedDeclBindId);
  if (!DS) {
    return;
  }

  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);
  const VarDecl      *VD = Result.Nodes.getNodeAs<VarDecl>(CUDASharedVarBindId);

  auto FName    = FD->getNameAsString();
  auto VName    = VD->getNameAsString();
  auto ParamUSR = getUSRsForDeclaration(VD->getUnderlyingDecl(), *Context);

  ShrdVarListMap[FName].push_back(VName);
  ShrdVarUSRsListMap[FName].push_back(ParamUSR);

  auto TypeInfo = Context->getTypeInfo(VD->getType());
  ShrdVarSizeListMap[FName].push_back(TypeInfo.Width / TypeInfo.Align);
}
//---------------------------------------------------------------------------
DeclarationMatcher CUDAFuncBuilder::getFuncBuildMatcher(const string &KName)
{
  return parmVarDecl(
           hasAncestor(
             functionDecl(
               hasAttr(attr::CUDAGlobal),
               hasName(KName)
             ).bind(CUDAFuncDeclBindId)
           )
         ).bind(CUDAFuncParamBindId);
}
//---------------------------------------------------------------------------
void CUDAFuncBuilder::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(CUDAFuncDeclBindId);
  const ParmVarDecl  *PD = Result.Nodes.getNodeAs<ParmVarDecl>(CUDAFuncParamBindId);
  if (!PD) {
    return;
  }

  // Print function body
  string FName      = FD->getNameAsString();
  auto &PrintPolicy = Context->getPrintingPolicy();

  string BodyStr;
  llvm::raw_string_ostream BodyStream{BodyStr};

  if (FuncBodyStringMap.find(FName) == FuncBodyStringMap.end()) {
    for (auto *Child : FD->getBody()->children()) {
      if (auto *CS = dyn_cast<CompoundStmt>(Child)) {
        CS->printPretty(BodyStream, nullptr, PrintPolicy, /*Indentation=*/1U);
      }
    }
    BodyStream.flush();
    FuncBodyStringMap[FName] = BodyStr;
  }

  // Print function parameters
  string ParamStr;
  llvm::raw_string_ostream ParamStream{ParamStr};

  PD->print(ParamStream, Context->getPrintingPolicy());
  ParamStream.flush();
  ParmStringList.push_back(ParamStr);
}
//---------------------------------------------------------------------------
void CUDAFuncBuilder::onEndOfTranslationUnit()
{
  // Macro
  string TVMMacros =
R"(
#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
      (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
)";

  // Attributes
  string ExternAttr     = "extern \"C\"";
  string CUDAGlobalAttr = "__global__";
  string CUDALaunchAttr = "";
  CUDALaunchAttr += "__launch_bounds__(" + to_string(Analysis.MaxThreadBound) + ")";

  // Function declaration (name)
  string CUDAFuncName = Analysis.NewFuncName;

  // Function declaration (paramenter)
  // FIXME: need to fix __restrict -> __restrict__
  // Maybe bug?
  auto AccFunc = [](string a, string b) { return a + ", " + b; };

  string CUDAFuncParam = accumulate(ParmStringList.begin() + 1,
                                    ParmStringList.end(),
                                    ParmStringList[0],
                                    AccFunc);
  
  // Function body
  auto &BranchCondMap  = Analysis.BranchConditionMap;
  auto &Kernels        = Analysis.Kernels;
  string CUDAFuncBody  = "";

  CUDAFuncBody += Analysis.NewShrdDeclString + Analysis.NewBlockInfoString;

  for (long unsigned I = 0; I < Kernels.size(); ++I) {
    auto &KName   = Kernels[I];
    auto &CondStr = BranchCondMap.at(KName);
    auto &BodyStr = FuncBodyStringMap.at(KName);

    // Comments
    CUDAFuncBody += "  // " + KName + "\n";

    if (I == 0) {
      CUDAFuncBody += "  ";
    } else {
      CUDAFuncBody += "  else ";
    }
    CUDAFuncBody += "if (" + CondStr + ")\n";
    CUDAFuncBody += BodyStr; // include {}
  }

  // Create fused function
  FuncStream << TVMMacros
             << ExternAttr << " " << CUDAGlobalAttr << " "
             << CUDALaunchAttr << " " << CUDAFuncName << "("
             << CUDAFuncParam << ")\n"
             << "{\n"
             <<    CUDAFuncBody
             << "}\n";
             
  FuncStream.flush();
}
//---------------------------------------------------------------------------
} // namespace matchers
} // namespace bfuse
//---------------------------------------------------------------------------