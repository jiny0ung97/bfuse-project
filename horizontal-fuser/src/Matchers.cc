
#include <cstdlib>
#include <string>
#include <numeric>
#include <algorithm>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Tooling/Core/Replacement.h"
#include "clang/Tooling/Refactoring/Rename/USRFindingAction.h"

#include "llvm/Support/raw_ostream.h"

#include "fuse/Matchers.h"
#include "fuse/Contexts.h"

using namespace std;

using namespace fuse::contexts;

using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;
//---------------------------------------------------------------------------
namespace fuse {
namespace matchers {
//--------------------------------------------------------------------------
const string ASTPatternMatcher::CUDAFuncDecl          = "cudaFuncDecl";
const string ASTPatternMatcher::CUDAFuncParm          = "cudaFuncParm";
const string ASTPatternMatcher::CUDABlockIdxVarMember = "cudaBlockIdxMember";
const string ASTPatternMatcher::CUDABlockIdxVar       = "cudaBlockIdxVar";
const string ASTPatternMatcher::CUDASync              = "cudaSync";
const string ASTPatternMatcher::CUDASharedDecl        = "cudaSharedDecl";
const string ASTPatternMatcher::CUDASharedVar         = "cudaSharedVar";
//--------------------------------------------------------------------------
DeclarationMatcher ASTPatternMatcher::getFuncDeclMatcher(const string &KName)
{
  return functionDecl(
           hasAttr(attr::CUDAGlobal),
           hasName(KName)
         ).bind(CUDAFuncDecl);
}
//---------------------------------------------------------------------------
DeclarationMatcher ASTPatternMatcher::getFuncParmMatcher(const string &KName)
{
  return parmVarDecl(
           hasAncestor(
             functionDecl(
               hasAttr(attr::CUDAGlobal),
               hasName(KName)
             ).bind(CUDAFuncDecl)
           )
         ).bind(CUDAFuncParm);
}
//---------------------------------------------------------------------------
StatementMatcher ASTPatternMatcher::getBlockIdxMatcher(const string &KName)
{
  return memberExpr(
           hasObjectExpression(
             opaqueValueExpr(
               hasSourceExpression(
                 declRefExpr(
                   to(varDecl(
                     anyOf(
                       hasName("gridDim"),
                       hasName("blockIdx"),
                       hasName("blockDim"),
                       hasName("threadIdx")
                     )).bind(CUDABlockIdxVar)
           ))))),
           hasAncestor(
             functionDecl(
               hasAttr(attr::CUDAGlobal),
               hasName(KName)
             ).bind(CUDAFuncDecl)
           )
         ).bind(CUDABlockIdxVarMember);
}
//---------------------------------------------------------------------------
StatementMatcher ASTPatternMatcher::getSyncMatcher(const string &KName)
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
             ).bind(CUDAFuncDecl)
           )
         ).bind(CUDASync);
}
//---------------------------------------------------------------------------
StatementMatcher ASTPatternMatcher::getSharedDeclMatcher(const std::string &KName)
{
  return declStmt(
           hasSingleDecl(
             varDecl(
               hasAttr(attr::CUDAShared)
             ).bind(CUDASharedVar)
           ),
           hasAncestor(
             functionDecl(
               hasAttr(attr::CUDAGlobal),
               hasName(KName)
             ).bind(CUDAFuncDecl)
           )
         ).bind(CUDASharedDecl);
}
//---------------------------------------------------------------------------
void CUDAFuncDeclPrinter::run(const MatchFinder::MatchResult &Result)
{
  const FunctionDecl* FD = Result.Nodes.getNodeAs<FunctionDecl>(ASTPatternMatcher::CUDAFuncDecl);
  if (!FD) {
    return;
  }

  // Print function
  FD->dump();
}
//---------------------------------------------------------------------------
void CUDAKernelRewriter::run(const MatchFinder::MatchResult& Result)
{
  ASTContext *Context    = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(ASTPatternMatcher::CUDAFuncDecl);  
  if (!FD) {
    return;
  }

  string FName = FD->getNameAsString();
  if (IsVisitedMap_.find(FName) != IsVisitedMap_.end()) {
    return;
  }
  IsVisitedMap_[FName] = true;

  if (FD->isTemplateInstantiation())
    return;

  auto &SourceMgr = Context->getSourceManager();
  auto *FuncBody  = FD->getBody();

  string BodyStr;
  llvm::raw_string_ostream BodyStream{BodyStr};

  FuncBody->printPretty(BodyStream, nullptr, Context->getPrintingPolicy());
  BodyStream.flush();

  Replacement BodyRepl{SourceMgr, FuncBody, BodyStr};
  string FilePath = FD->getLocation().printToString(SourceMgr);
  FilePath = FilePath.substr(0, FilePath.find_first_of(":"));

  if (auto Err = Repls_[FilePath].add(BodyRepl)) {
    llvm::errs() << "CUDAKernelRewriter error occur\n";
    exit(0);
  }
}
//---------------------------------------------------------------------------
void CUDACompStmtRewriter::run(const MatchFinder::MatchResult& Result)
{
  ASTContext *Context    = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(ASTPatternMatcher::CUDAFuncDecl);  
  if (!FD) {
    return;
  }

  string FName = FD->getNameAsString();
  if (IsVisitedMap_.find(FName) != IsVisitedMap_.end()) {
    return;
  }
  IsVisitedMap_[FName] = true;

  // Append compound {}
  auto &SourceMgr     = Context->getSourceManager();
  auto SourceBeginLoc = FD->getBody()->getBeginLoc().getLocWithOffset(1);
  auto SourceEndLoc   = FD->getBody()->getEndLoc().getLocWithOffset(-1);

  string TmpBlockInfoString;
  llvm::raw_string_ostream TmpVarStream{TmpBlockInfoString};
  TmpVarStream << "\n"
               << "  // Temp declaration to avoid semantic errors\n"
               << "  int gridDim_x_  = 0;\n"
               << "  int gridDim_y_  = 0;\n"
               << "  int gridDim_z_  = 0;\n"
               << "  int blockIdx_x_ = 0;\n"
               << "  int blockIdx_y_ = 0;\n"
               << "  int blockIdx_z_ = 0;\n"
               << "  int blockDim_x_  = 0;\n"
               << "  int blockDim_y_  = 0;\n"
               << "  int blockDim_z_  = 0;\n"
               << "  int threadIdx_x_ = 0;\n"
               << "  int threadIdx_y_ = 0;\n"
               << "  int threadIdx_z_ = 0;\n";
  TmpVarStream.flush();

  string CompBeginString = "\n  {";
  string CompEndString   = "\n  }";

  Replacement CompBeginRepl{SourceMgr, SourceBeginLoc, 0, TmpBlockInfoString + CompBeginString};
  string CBFile = CompBeginRepl.getFilePath().str();
  if (auto Err = Repls_[CBFile].add(CompBeginRepl)) {
    llvm::errs() << "CUDADeclExtractor \'{\' error occur\n";
    exit(0);
  }

  Replacement CompEndRepl{SourceMgr, SourceEndLoc, 0, CompEndString};
  string CEFile = CompEndRepl.getFilePath().str();
  if (auto Err = Repls_[CEFile].add(CompEndRepl)) {
    llvm::errs() << "CUDADeclExtractor \'}\' error occur\n";
    exit(0);
  }
}
//---------------------------------------------------------------------------
void CUDAFuncParmAnalyzer::run(const MatchFinder::MatchResult& Result)
{
  ASTContext *Context    = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(ASTPatternMatcher::CUDAFuncDecl);
  if (!FD) {
    return;
  }

  string FName = FD->getNameAsString();
  if (IsVisitedMap_.find(FName) != IsVisitedMap_.end()) {
    return;
  }
  IsVisitedMap_[FName] = true;

  // Analyze function template arguments
  if (auto *TD = FD->getDescribedFunctionTemplate()) {
    auto *PDList = TD->getTemplateParameters();
    for (auto *PD : *PDList) {
      string PName = PD->getNameAsString();
      if (PName.empty())
        continue;

      auto ParamUSR = getUSRsForDeclaration(PD->getUnderlyingDecl(), *Context);
      ParmListMap_[FName].push_back(PName);
      ParmUSRsListMap_[FName].push_back(ParamUSR);
    }
  }

  // Analyze function parameters
  for (auto *PD : FD->parameters()) {
    string PName = PD->getNameAsString();
    if (PName.empty())
      continue;

    auto ParamUSR = getUSRsForDeclaration(PD->getUnderlyingDecl(), *Context);
    ParmListMap_[FName].push_back(PName);
    ParmUSRsListMap_[FName].push_back(ParamUSR);
  }
}
//---------------------------------------------------------------------------
void CUDABlockInfoRewriter::run(const MatchFinder::MatchResult &Result)
{
  ASTContext  *Context = Result.Context;
  const MemberExpr *ME = Result.Nodes.getNodeAs<MemberExpr>(ASTPatternMatcher::CUDABlockIdxVarMember);
  const VarDecl    *VD = Result.Nodes.getNodeAs<VarDecl>(ASTPatternMatcher::CUDABlockIdxVar);
  if (!ME) {
    return;
  }

  // Rewrite Block informations
  map<string, string> MemberReNamingMap = {
    {"__fetch_builtin_x", "x"},
    {"__fetch_builtin_y", "y"},
    {"__fetch_builtin_z", "z"}
  };
  auto ReNamingFunc = [](string &Var, string &Member) { return Var + "_" + Member + "_"; };

  string MName     = ME->getMemberNameInfo().getAsString();
  string VName     = VD->getNameAsString();
  string &NewMName = MemberReNamingMap.at(MName);
  string NewVName  = ReNamingFunc(VName, NewMName);

  auto CharSrcRange = CharSourceRange::getTokenRange(ME->getSourceRange());
  auto& SourceMgr   = Context->getSourceManager();

  Replacement RenameRepl{SourceMgr, CharSrcRange, NewVName};
  string RFilePath = RenameRepl.getFilePath().str();
  if (auto Err = Repls_[RFilePath].add(RenameRepl)) {
    llvm::errs() << "CUDABlockInfoRewriter error occur\n";
    exit(0);
  }
}
//---------------------------------------------------------------------------
void CUDASyncRewriter::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context = Result.Context;
  const CallExpr *CE  = Result.Nodes.getNodeAs<CallExpr>(ASTPatternMatcher::CUDASync);
  if (!CE) {
    return;
  }

  // Rewrite synchronization
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(ASTPatternMatcher::CUDAFuncDecl);
  string FName  = FD->getNameAsString();
  int ThreadNum = ThreadNumMap_.at(FName);

  auto RefactoringFunc = [](int BlockDim) { return "asm(\"bar.sync 0, " + to_string(BlockDim) + ";\")"; };

  string NewSync  = RefactoringFunc(ThreadNum);
  auto &SourceMgr = Context->getSourceManager();

  Replacement Repl{SourceMgr, CE, NewSync};
  string FilePath = Repl.getFilePath().str();
  if (auto Err = Repls_[FilePath].add(Repl)) {
    llvm::errs() << "CUDASyncRewriter error occur\n";
    exit(0);
  }
}
//---------------------------------------------------------------------------
void CUDASharedDeclExtractor::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context = Result.Context;
  const DeclStmt *DS = Result.Nodes.getNodeAs<DeclStmt>(ASTPatternMatcher::CUDASharedDecl);
  if (!DS) {
    return;
  }

  // Remove existing declaration
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(ASTPatternMatcher::CUDAFuncDecl);
  // const VarDecl      *VD = Result.Nodes.getNodeAs<VarDecl>(ASTPatternMatcher::CUDASharedVar);

  string FName = FD->getNameAsString();

  auto &SourceMgr     = Context->getSourceManager();
  auto SourceBeginLoc = DS->getBeginLoc();

  Replacement RemoveRepl{SourceMgr, SourceBeginLoc, 0, "// "};
  string RFPath = RemoveRepl.getFilePath().str();
  if (auto Err = Repls_[RFPath].add(RemoveRepl)) {
    llvm::errs() << "CUDASharedDeclExtractor error occur\n";
    exit(0);
  }
}
//---------------------------------------------------------------------------
void CUDASharedDeclRewriter::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context = Result.Context;
  const DeclStmt *DS  = Result.Nodes.getNodeAs<DeclStmt>(ASTPatternMatcher::CUDASharedDecl);
  if (!DS) {
    return;
  }

  // Store SharedDecl, ASTContext and SourceLocation
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(ASTPatternMatcher::CUDAFuncDecl);
  // const VarDecl      *VD = Result.Nodes.getNodeAs<VarDecl>(ASTPatternMatcher::CUDASharedVar);

  string FName = FD->getNameAsString();

  string DeclStr;
  llvm::raw_string_ostream DeclStream{DeclStr};
  DS->printPretty(DeclStream, nullptr, Context->getPrintingPolicy(), /*Indentation=*/1U);
  DeclStream.flush();

  if (SharedDeclStringMap_.find(FName) == SharedDeclStringMap_.end()) {
    SharedDeclStringMap_[FName] = "\n";
    Kernels_.push_back(FName);
  }
  auto &SharedDeclString = SharedDeclStringMap_.at(FName);
  SharedDeclString += DeclStr;

  if (ASTContextMap_.find(FName) == ASTContextMap_.end()) {
    ASTContextMap_[FName] = Context;
  }
  auto SourceLoc = FD->getBody()->getBeginLoc().getLocWithOffset(1);
  if (SourceLocMap_.find(FName) == SourceLocMap_.end()) {
    SourceLocMap_[FName] = SourceLoc;
  }
}
//---------------------------------------------------------------------------
void CUDASharedDeclRewriter::onEndOfTranslationUnit()
{
  // Rewrite (Hoist) shared memory declarations
  for (auto &KName : Kernels_) {
    auto *Context     = ASTContextMap_.at(KName);
    auto &SourceLoc   = SourceLocMap_.at(KName);
    auto &ShrdDeclStr = SharedDeclStringMap_.at(KName);

    Replacement Repl{Context->getSourceManager(), SourceLoc, 0, ShrdDeclStr};
    string FilePath = Repl.getFilePath().str();
    if (auto Err = Repls_[FilePath].add(Repl)) {
      llvm::errs() << "CUDASharedDeclRewriter error occur\n";
      exit(0);
    }
  }
}
//---------------------------------------------------------------------------
} // namespace matchers
} // namespace fuse
//---------------------------------------------------------------------------