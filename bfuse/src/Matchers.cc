
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
                       hasName("blockIdx")
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
  // ASTContext    *Context = Result.Context;
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
  ASTContext *Context = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(ASTPatternMatcher::CUDAFuncDecl);  
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
  string FilePath = BodyRepl.getFilePath().str();
  if (auto Err = Repls[FilePath].add(BodyRepl)) {
    llvm::errs() << "CUDADeclRewriter error occur\n";
    exit(0);
  }

  // TODO: need to validate that kernels are all existed
}
//---------------------------------------------------------------------------
void CUDACompStmtRewriter::run(const MatchFinder::MatchResult& Result)
{
  ASTContext *Context = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(ASTPatternMatcher::CUDAFuncDecl);  
  if (!FD) {
    return;
  }

  // Append compound {}
  auto &SourceMgr     = Context->getSourceManager();
  auto SourceBeginLoc = FD->getBody()->getBeginLoc().getLocWithOffset(1);
  auto SourceEndLoc   = FD->getBody()->getEndLoc().getLocWithOffset(-1);

  string CompBeginString = "\n  {";
  string CompEndString   = "\n  }";

  Replacement CompBeginRepl{SourceMgr, SourceBeginLoc, 0, TmpBlockInfoString + CompBeginString};
  string CBFile = CompBeginRepl.getFilePath().str();
  if (auto Err = Repls[CBFile].add(CompBeginRepl)) {
    llvm::errs() << "CUDADeclExtractor \'{\' error occur\n";
    exit(0);
  }

  Replacement CompEndRepl{SourceMgr, SourceEndLoc, 0, CompEndString};
  string CEFile = CompEndRepl.getFilePath().str();
  if (auto Err = Repls[CEFile].add(CompEndRepl)) {
    llvm::errs() << "CUDADeclExtractor \'}\' error occur\n";
    exit(0);
  }
}
//---------------------------------------------------------------------------
void CUDAFuncParmAnalyzer::run(const MatchFinder::MatchResult& Result)
{
  ASTContext *Context = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(ASTPatternMatcher::CUDAFuncDecl);
  if (!FD) {
    return;
  }

  string FName = FD->getNameAsString();

  // Analyze function template arguments
  if (FD->isTemplateInstantiation()) {
    auto *TD     = FD->getDescribedFunctionTemplate();
    auto *PDList = TD->getTemplateParameters();

    for (auto *PD : *PDList) {
      string PName = PD->getNameAsString();
      if (PName.empty())
        continue;

      auto ParamUSR = getUSRsForDeclaration(PD->getUnderlyingDecl(), *Context);
      ParmListMap[FName].push_back(PName);
      ParmUSRsListMap[FName].push_back(ParamUSR);      
    }
  }

  // Analyze function parameters
  for (auto *PD : FD->parameters()) {
    string PName = PD->getNameAsString();
    if (PName.empty())
      continue;

    auto ParamUSR = getUSRsForDeclaration(PD->getUnderlyingDecl(), *Context);
    ParmListMap[FName].push_back(PName);
    ParmUSRsListMap[FName].push_back(ParamUSR);
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
  if (auto Err = Repls[RFilePath].add(RenameRepl)) {
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
  int ThreadNum = ThreadNumMap.at(FName);

  auto RefactoringFunc = [](int BlockDim) { return "asm(\"bar.sync 0, " + to_string(BlockDim) + ";\")"; };

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
void CUDASharedDeclExtractor::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context = Result.Context;
  const DeclStmt *DS = Result.Nodes.getNodeAs<DeclStmt>(ASTPatternMatcher::CUDASharedDecl);
  if (!DS) {
    return;
  }

  // Remove existing declaration
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(ASTPatternMatcher::CUDAFuncDecl);
  string KName = FD->getNameAsString();

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
void CUDASharedDeclRewriter::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context = Result.Context;
  const DeclStmt *DS  = Result.Nodes.getNodeAs<DeclStmt>(ASTPatternMatcher::CUDASharedDecl);
  if (!DS) {
    return;
  }

  // Store SharedDecl, ASTContext and SourceLocation
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(ASTPatternMatcher::CUDAFuncDecl);
  string FName = FD->getNameAsString();

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
}
//---------------------------------------------------------------------------
void CUDASharedDeclRewriter::onEndOfTranslationUnit()
{
  // Rewrite (Hoist) shared memory declarations
  for (auto &KName : Kernels) {
    auto *Context     = ASTContextMap.at(KName);
    auto &SourceLoc   = SourceLocMap.at(KName);
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
void CUDASharedVarAnalyzer::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context = Result.Context;
  const DeclStmt *DS  = Result.Nodes.getNodeAs<DeclStmt>(ASTPatternMatcher::CUDASharedDecl);
  if (!DS) {
    return;
  }

  // Analyze shared memory variables
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(ASTPatternMatcher::CUDAFuncDecl);
  const VarDecl      *VD = Result.Nodes.getNodeAs<VarDecl>(ASTPatternMatcher::CUDASharedVar);

  string FName  = FD->getNameAsString();
  string VName  = VD->getNameAsString();
  auto ParamUSR = getUSRsForDeclaration(VD->getUnderlyingDecl(), *Context);

  ShrdVarListMap[FName].push_back(VName);
  ShrdVarUSRsListMap[FName].push_back(ParamUSR);

  auto TypeInfo = Context->getTypeInfo(VD->getType());
  ShrdVarSizeListMap[FName].push_back(TypeInfo.Width / TypeInfo.Align);
}
//---------------------------------------------------------------------------
void CUDAFuncBuilder::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(ASTPatternMatcher::CUDAFuncDecl);
  if (!FD) {
    return;
  }

  // Print function template parameters
  string TemplStr;
  llvm::raw_string_ostream TemplStream{TemplStr};

  if (FD->isTemplateInstantiation()) {
    IsFuncTemplate = true;

    auto *TD     = FD->getDescribedFunctionTemplate();
    auto *PDList = TD->getTemplateParameters();

    for (auto *PD : *PDList) {
      PD->print(TemplStream, Context->getPrintingPolicy());
      TemplStream.flush();
      TemplStringList.push_back(TemplStr);
    }
  }

  // Print function parameters
  string ParamStr;
  llvm::raw_string_ostream ParamStream{ParamStr};

  for (auto *PD : FD->parameters()) {
    PD->print(ParamStream, Context->getPrintingPolicy());
    ParamStream.flush();
    ParmStringList.push_back(ParamStr);
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
  string CUDAFuncRtrTy = "void";
  // string CUDAFuncName  = Analysis.NewFuncName + "fused_kernel_bfuse";
  string CUDAFuncName  = Analysis.NewFuncName + "fused_kernel_hfuse_idx_0";

  // Function declaration (template parameter)
  auto AccFunc = [](string a, string b) { return a + ", " + b; };

  string CUDAFuncTempl = "";
  if (IsFuncTemplate) {
    CUDAFuncTempl += "template <";
    if (TemplStringList.size() > 1) {
      CUDAFuncTempl += accumulate(TemplStringList.begin() + 1,
                            TemplStringList.end(),
                            TemplStringList[0],
                            AccFunc);
    }
    else if (TemplStringList.size() == 1) {
      CUDAFuncTempl += TemplStringList[0];
    }
    CUDAFuncTempl += ">";
  }

  // Function declaration (paramenter)
  // FIXME: need to fix __restrict -> __restrict__
  // Maybe bug?
  string CUDAFuncParam = "";
  if (ParmStringList.size() > 1) {
    CUDAFuncParam += accumulate(ParmStringList.begin() + 1,
                                ParmStringList.end(),
                                ParmStringList[0],
                                AccFunc);
  }
  else if (ParmStringList.size() == 1) {
    CUDAFuncParam += ParmStringList[0];
  }
  
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
  FuncStream << ""
  // FuncStream << TVMMacros
             << CUDAFuncTempl << "\n"
             << ExternAttr << " " << CUDAGlobalAttr << " "
             << CUDALaunchAttr << " " << CUDAFuncRtrTy << " " << CUDAFuncName << "("
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