
#include <utility>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Refactoring.h"
#include "clang/Tooling/Tooling.h"

#include "bfuse/Bfuse.h"
#include "bfuse/Utils.h"
#include "bfuse/Tools.h"
#include "bfuse/MatchFinders.h"

using namespace std;
using namespace clang::tooling;
using namespace clang::attr;
using namespace clang::ast_matchers;
//---------------------------------------------------------------------------
static DeclarationMatcher MacroExpandMatcher = functionDecl(hasAttr(CUDAGlobal))
                                                           .bind(bfuse::match_finders::macroExpandMatcher);
//---------------------------------------------------------------------------
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
Arguments::Arguments(const char *ProgName, string& Path)
{
  filePath = Path;

  argv    = (const char**)malloc(sizeof(char *) * 2);
  argv[0] = ProgName;
  argv[1] = filePath.c_str();
}
//---------------------------------------------------------------------------
Arguments::~Arguments() { free(argv); }
//---------------------------------------------------------------------------
FusionTools FusionTools::create(FusionInfo& FInfo, map<string, KernelInfo>& KInfo)
{
  vector<KernelInfo> KInfoVector;
  for (auto& KName : FInfo.kernels) {
    KInfoVector.push_back(KInfo[KName]);
  }
  return FusionTools{KInfoVector};
}
//---------------------------------------------------------------------------
FusionTools::FusionTools(vector<KernelInfo>& Infos)
{
  unordered_map<string, int> CurBounds;
  unordered_map<string, int> EndBounds;

  const int TotalSM  = 84;
  int Idx            = 0;
  int TotalBounds    = 0;
  bool LastLoop      = false;

  for (auto& info : Infos) {
    auto& KName = info.kernelName;
    CurBounds[KName] = 0;
    EndBounds[KName] = info.gridDim.size();

    auto Pair = make_pair(0, info.blockDim.size());
    kernelContexts[KName] = KernelContext(info, move(Pair));
  }

  while(true) {
    auto& KName   = Infos[Idx].kernelName;
    auto& Context = kernelContexts.find(KName)->second;
    auto& BlockIdxInfo  = Context.blockIdxInfo;
    auto& OtherBlocks   = Context.otherBlocks;

    int Stride  = EndBounds[KName] - CurBounds[KName];

    if (!LastLoop && Stride > TotalSM)
      Stride = TotalSM;
    
    BlockIdxInfo.emplace_back(TotalBounds, TotalBounds + Stride);
    OtherBlocks.push_back(TotalBounds - CurBounds[KName]);

    if (LastLoop)
      break;

    CurBounds[KName] += Stride;
    TotalBounds      += Stride;

    if (CurBounds[KName] == EndBounds[KName])
      LastLoop = true;

    Idx = (Idx + 1) % Infos.size();
  }
}
//---------------------------------------------------------------------------
void FusionTools::expandMacros(CommonOptionsParser& OptionParser)
{
  RefactoringTool Tool(OptionParser.getCompilations(),
                       OptionParser.getSourcePathList());

  match_finders::MacroExpander Expander(Tool.getReplacements(), kernelContexts);
  MatchFinder Finder;
  Finder.addMatcher(MacroExpandMatcher, &Expander);

  if (!Tool.run(newFrontendActionFactory(&Finder).get())) {
    CHECK_ERROR("cannot run MacroExpander tools.");
    exit(0);
  }
}
//---------------------------------------------------------------------------
void FusionTools::renameParameters(CommonOptionsParser& OptionParser)
{

}
//---------------------------------------------------------------------------
void FusionTools::rewriteThreadInfo(CommonOptionsParser& OptionParser)
{

}
//---------------------------------------------------------------------------
void FusionTools::barrierRewriter(CommonOptionsParser& OptionParser)
{

}
//---------------------------------------------------------------------------
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------