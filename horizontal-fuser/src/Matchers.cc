
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
    exit(1);
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
    exit(1);
  }

  Replacement CompEndRepl{SourceMgr, SourceEndLoc, 0, CompEndString};
  string CEFile = CompEndRepl.getFilePath().str();
  if (auto Err = Repls_[CEFile].add(CompEndRepl)) {
    llvm::errs() << "CUDADeclExtractor \'}\' error occur\n";
    exit(1);
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
    exit(1);
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
  string FName = FD->getNameAsString();
  auto &KInfo  = FContext_.KernelInfoMap_.at(FName);

  int Threads;
  if (KInfo.BlockDim_.size() % 32 == 0) {
    Threads = KInfo.BlockDim_.size();
  } else {
    Threads = (int(KInfo.BlockDim_.size() / 32) + 1) * 32;
  }
  long unsigned KI = find(FContext_.Kernels_.begin(), FContext_.Kernels_.end(), FName) - FContext_.Kernels_.begin();

  auto RefactoringFunc = [](long unsigned Idx, int BlockDim) { return "asm(\"bar.sync " + to_string(Idx) + ", " + to_string(BlockDim) + ";\")"; };

  string NewSync  = RefactoringFunc(KI + 1, Threads);
  auto &SourceMgr = Context->getSourceManager();

  Replacement Repl{SourceMgr, CE, NewSync};
  string FilePath = Repl.getFilePath().str();
  if (auto Err = Repls_[FilePath].add(Repl)) {
    llvm::errs() << "CUDASyncRewriter error occur\n";
    exit(1);
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
  const VarDecl      *VD = Result.Nodes.getNodeAs<VarDecl>(ASTPatternMatcher::CUDASharedVar);

  string FName = FD->getNameAsString();
  if (VD->hasExternalStorage()) {
    return;
  }

  auto &SourceMgr     = Context->getSourceManager();
  auto SourceBeginLoc = DS->getBeginLoc();

  Replacement RemoveRepl{SourceMgr, SourceBeginLoc, 0, "// "};
  string RFPath = RemoveRepl.getFilePath().str();
  if (auto Err = Repls_[RFPath].add(RemoveRepl)) {
    llvm::errs() << "CUDASharedDeclExtractor error occur\n";
    exit(1);
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
  const VarDecl      *VD = Result.Nodes.getNodeAs<VarDecl>(ASTPatternMatcher::CUDASharedVar);

  string FName = FD->getNameAsString();
  if (VD->hasExternalStorage()) {
    return;
  }

  string DeclStr;
  llvm::raw_string_ostream DeclStream{DeclStr};
  DS->printPretty(DeclStream, nullptr, Context->getPrintingPolicy(), /*Indentation=*/0U);
  DeclStream.flush();

  if (SharedDeclStrMap_.find(FName) == SharedDeclStrMap_.end()) {
    SharedDeclStrMap_[FName] = vector<string>();
    Kernels_.push_back(FName);
  }

  auto &SharedDeclStr = SharedDeclStrMap_.at(FName);
  SharedDeclStr.push_back(DeclStr);

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
    auto &ShrdDeclStr = SharedDeclStrMap_.at(KName);

    string TmpShrdDeclStr;
    llvm::raw_string_ostream TmpStream{TmpShrdDeclStr};

    for (auto &DeclStr : ShrdDeclStr) {
      TmpStream << "  " << DeclStr << ";\n";
    }
    TmpStream.flush();

    Replacement Repl{Context->getSourceManager(), SourceLoc, 0, TmpShrdDeclStr};
    string FilePath = Repl.getFilePath().str();
    if (auto Err = Repls_[FilePath].add(Repl)) {
      llvm::errs() << "CUDASharedDeclRewriter error occur\n";
      exit(1);
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

  if (VD->hasExternalStorage()) {
    return;
  }

  string FName  = FD->getNameAsString();
  string VName  = VD->getNameAsString();
  auto ShrdUSR = getUSRsForDeclaration(VD->getUnderlyingDecl(), *Context);

  // Skip temp variable
  if(VName.compare("SU_") == 0) {
    return;
  }

  string DeclStr;
  llvm::raw_string_ostream DeclStream{DeclStr};
  DS->printPretty(DeclStream, nullptr, Context->getPrintingPolicy(), /*Indentation=*/0U);
  DeclStream.flush();

  ShrdVarListMap_[FName].push_back(VName);
  ShrdVarUSRsListMap_[FName].push_back(ShrdUSR);

  if (SharedDeclStrMap_.find(FName) == SharedDeclStrMap_.end()) {
    SharedDeclStrMap_[FName] = vector<string>();
  }
  auto &SharedDeclStr = SharedDeclStrMap_.at(FName);

  string ShrdAttr   = " __attribute__((shared))";
  string StaticAttr = "static";

  auto ShrdIter   = DeclStr.find(ShrdAttr);
  auto StaticIter = DeclStr.find(StaticAttr);

  DeclStr.erase(ShrdIter, ShrdIter+ShrdAttr.size());
  DeclStr.erase(StaticIter, StaticIter+StaticAttr.size());

  SharedDeclStr.push_back(DeclStr);
}
//---------------------------------------------------------------------------
void BFuseBuilder::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context    = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(ASTPatternMatcher::CUDAFuncDecl);
  if (!FD) {
    return;
  }

  // Print function template parameters
  if (auto *TD = FD->getDescribedFunctionTemplate()) {
    IsFuncTemplate_ = true;

    auto *PDList = TD->getTemplateParameters();
    for (auto *PD : *PDList) {
      string TemplStr;
      llvm::raw_string_ostream TemplStream{TemplStr};

      PD->print(TemplStream, Context->getPrintingPolicy());
      TemplStream.flush();
      TemplStringList_.push_back(TemplStr);
    }
  }

  // Print function parameters
  for (auto *PD : FD->parameters()) {
    string ParamStr;
    llvm::raw_string_ostream ParamStream{ParamStr};

    PD->print(ParamStream, Context->getPrintingPolicy());
    ParamStream.flush();
    ParmStringList_.push_back(ParamStr);
  }

  // Print function body
  string FName      = FD->getNameAsString();
  auto &PrintPolicy = Context->getPrintingPolicy();

  string BodyStr;
  llvm::raw_string_ostream BodyStream{BodyStr};

  if (FuncBodyStringMap_.find(FName) == FuncBodyStringMap_.end()) {
    for (auto *Child : FD->getBody()->children()) {
      if (auto *CS = dyn_cast<CompoundStmt>(Child)) {
        CS->printPretty(BodyStream, nullptr, PrintPolicy, /*Indentation=*/1U);
      }
    }
    BodyStream.flush();

    string TmpName = "";
    size_t TmpIter;

    TmpName += "_SU_" + FName + "_";
    while ((TmpIter = BodyStr.find(TmpName)) != string::npos) {
      BodyStr.replace(TmpIter, TmpName.size(), "SU_." + FName + ".");
    }
    FuncBodyStringMap_[FName] = BodyStr;
  }
}
//---------------------------------------------------------------------------
void BFuseBuilder::onEndOfTranslationUnit()
{
  // Attributes
  string ExternAttr     = "extern \"C\"";
  string CUDAGlobalAttr = "__global__";
  string CUDALaunchAttr = "";
  CUDALaunchAttr += "__launch_bounds__(" + to_string(FContext_.FusedBlockDim_.size()) + ")";

  // Function declaration (name)
  string CUDAFuncRtrTy = "void";
  string CUDAFuncName  = FContext_.FusedKernelName_;

  // Function declaration (template parameter)
  auto AccFunc = [](string a, string b) { return a + ", " + b; };

  string CUDAFuncTempl = "";
  if (IsFuncTemplate_) {
    string FuncTemplDecl;
    if (TemplStringList_.size() > 1) {
      FuncTemplDecl = accumulate(TemplStringList_.begin() + 1,
                                 TemplStringList_.end(),
                                 TemplStringList_[0],
                                 AccFunc);
    }
    else if (TemplStringList_.size() == 1) {
      FuncTemplDecl = TemplStringList_[0];
    }

    CUDAFuncTempl += "template <" + FuncTemplDecl + ">\n";
  }

  // Function declaration (paramenter)
  string CUDAFuncParam = "";
  if (ParmStringList_.size() > 1) {
    CUDAFuncParam += accumulate(ParmStringList_.begin() + 1,
                                ParmStringList_.end(),
                                ParmStringList_[0],
                                AccFunc);
  }
  else if (ParmStringList_.size() == 1) {
    CUDAFuncParam += ParmStringList_[0];
  }
  
  // Function body
  auto &NewBlockInfoString = FContext_.FusedBlockDeclStr_;
  auto &NewCondStrMap      = FContext_.FusedCondStrMap_;

  string CUDAFuncBody = "";
  CUDAFuncBody += NewBlockInfoString + "\n" + UnionStr_ + "\n";

  // Check CUDA block scheduler
  // CUDAFuncBody += "  uint streamingMultiprocessorId;\n";
	// CUDAFuncBody += "  asm(\"mov.u32 %0, %smid;\" : \"=r\"(streamingMultiprocessorId));\n";
  // CUDAFuncBody += "  uint warpId;\n";
	// CUDAFuncBody += "  asm volatile (\"mov.u32 %0, %warpid;\" : \"=r\"(warpId));\n";
  // CUDAFuncBody += "  uint laneId;\n";
	// CUDAFuncBody += "  asm volatile (\"mov.u32 %0, %laneid;\" : \"=r\"(laneId));\n";
  // CUDAFuncBody += "  printf(\"Block: %d | SM: %d - Here!\\n\", blockIdx.x, streamingMultiprocessorId);\n";
  // CUDAFuncBody += "  \n";
  
  for (auto &KName : FContext_.Kernels_) {
    auto &CondStr = NewCondStrMap.at(KName);
    auto &BodyStr = FuncBodyStringMap_.at(KName);

    // Comments
    CUDAFuncBody += "  // " + KName + "\n";
    CUDAFuncBody += CondStr + BodyStr;
  }

  // Create fused function
  FuncStream_ << CUDAFuncTempl // include "\n"
              << ExternAttr << " " << CUDAGlobalAttr << " "
              << CUDALaunchAttr << " " << CUDAFuncRtrTy << " " << CUDAFuncName << "("
              << CUDAFuncParam << ")\n"
              << "{\n"
              <<    CUDAFuncBody
              << "}\n";
             
  FuncStream_.flush();
}
//---------------------------------------------------------------------------
void HFuseBuilder::run(const MatchFinder::MatchResult &Result)
{
  ASTContext *Context    = Result.Context;
  const FunctionDecl *FD = Result.Nodes.getNodeAs<FunctionDecl>(ASTPatternMatcher::CUDAFuncDecl);
  if (!FD) {
    return;
  }

  // Print function template parameters
  if (auto *TD = FD->getDescribedFunctionTemplate()) {
    IsFuncTemplate_ = true;

    auto *PDList = TD->getTemplateParameters();
    for (auto *PD : *PDList) {
      string TemplStr;
      llvm::raw_string_ostream TemplStream{TemplStr};

      PD->print(TemplStream, Context->getPrintingPolicy());
      TemplStream.flush();
      TemplStringList_.push_back(TemplStr);
    }
  }

  // Print function parameters
  for (auto *PD : FD->parameters()) {
    string ParamStr;
    llvm::raw_string_ostream ParamStream{ParamStr};

    PD->print(ParamStream, Context->getPrintingPolicy());
    ParamStream.flush();
    ParmStringList_.push_back(ParamStr);
  }

  // Print function body
  string FName      = FD->getNameAsString();
  auto &PrintPolicy = Context->getPrintingPolicy();

  string BodyStr;
  llvm::raw_string_ostream BodyStream{BodyStr};

  if (FuncBodyStringMap_.find(FName) == FuncBodyStringMap_.end()) {
    for (auto *Child : FD->getBody()->children()) {
      if (auto *CS = dyn_cast<CompoundStmt>(Child)) {
        CS->printPretty(BodyStream, nullptr, PrintPolicy, /*Indentation=*/1U);
      }
    }
    BodyStream.flush();
    FuncBodyStringMap_[FName] = BodyStr;
  }
}
//---------------------------------------------------------------------------
void HFuseBuilder::onEndOfTranslationUnit()
{
  // Attributes
  string ExternAttr     = "extern \"C\"";
  string CUDAGlobalAttr = "__global__";
  string CUDALaunchAttr = "";
  CUDALaunchAttr += "__launch_bounds__(" + to_string(FContext_.FusedBlockDim_.size()) + ")";

  // Function declaration (name)
  string CUDAFuncRtrTy = "void";
  string CUDAFuncName  = FContext_.FusedKernelName_;

  // Function declaration (template parameter)
  auto AccFunc = [](string a, string b) { return a + ", " + b; };

  string CUDAFuncTempl = "";
  if (IsFuncTemplate_) {
    string FuncTemplDecl;
    if (TemplStringList_.size() > 1) {
      FuncTemplDecl = accumulate(TemplStringList_.begin() + 1,
                                 TemplStringList_.end(),
                                 TemplStringList_[0],
                                 AccFunc);
    }
    else if (TemplStringList_.size() == 1) {
      FuncTemplDecl = TemplStringList_[0];
    }

    CUDAFuncTempl += "template <" + FuncTemplDecl + ">\n";
  }

  // Function declaration (paramenter)
  string CUDAFuncParam = "";
  if (ParmStringList_.size() > 1) {
    CUDAFuncParam += accumulate(ParmStringList_.begin() + 1,
                                ParmStringList_.end(),
                                ParmStringList_[0],
                                AccFunc);
  }
  else if (ParmStringList_.size() == 1) {
    CUDAFuncParam += ParmStringList_[0];
  }
  
  // Function body
  auto &NewBlockInfoStrMap = FContext_.FusedBlockDeclStrMap_;
  auto &NewCondStrMap      = FContext_.FusedCondStrMap_;

  string CUDAFuncBody = "";

  for (auto &KName : FContext_.Kernels_) {
    auto &BlockDeclStr = NewBlockInfoStrMap.at(KName);
    auto &CondStr      = NewCondStrMap.at(KName);
    auto &BodyStr      = FuncBodyStringMap_.at(KName);

    // Insert declarations
    BodyStr.insert(4, BlockDeclStr + "\n");

    // Comments
    CUDAFuncBody += "  // " + KName + "\n";
    CUDAFuncBody += CondStr + BodyStr;
  }

  // Create fused function
  FuncStream_ << CUDAFuncTempl // include "\n"
              << ExternAttr << " " << CUDAGlobalAttr << " "
              << CUDALaunchAttr << " " << CUDAFuncRtrTy << " " << CUDAFuncName << "("
              << CUDAFuncParam << ")\n"
              << "{\n"
              <<    CUDAFuncBody
              << "}\n";
             
  FuncStream_.flush();
}
//---------------------------------------------------------------------------
} // namespace matchers
} // namespace fuse
//---------------------------------------------------------------------------