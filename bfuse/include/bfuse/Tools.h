
#pragma once

#include <utility>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <tuple>

#include "clang/Tooling/CommonOptionsParser.h"

#include "bfuse/Bfuse.h"
//---------------------------------------------------------------------------
namespace bfuse {
namespace tools {
//---------------------------------------------------------------------------
class Arguments {
private:
  /// The argc parameter
  int argc = 2;
  /// The argv parameter
  const char **argv;
  /// The file path of fused kernels
  std::string filePath;

public:
  /// Get argc, argv parameter
  std::tuple<int, const char **>getArguments() const { return std::make_tuple(argc, argv); }

  /// The constructor
  Arguments(const char *ProgName, std::string& Path);
  /// The destructor
  ~Arguments();

  /// Delete default constructor
  Arguments() = delete;
  /// Delete copy constructor
  Arguments(const Arguments& other) = delete;
  /// Delete move constructor
  Arguments(Arguments&& other) = delete;
  /// Delete copy assignment operator
  Arguments& operator=(const Arguments& other) = delete;
  /// Delete move assignment operator
  Arguments& operator=(Arguments&& other) = delete;
};
//---------------------------------------------------------------------------
struct KernelContext {
  using IdxBoundPair = std::pair<int, int>;

  /// The kernel's information
  KernelInfo info;
  /// The kernel's threadIdx boudnary
  IdxBoundPair threadIdxInfo;
  /// The kernel's blockIdx boundary
  std::vector<IdxBoundPair> blockIdxInfo;
  /// # of blocks from other fused kernels
  /// To rewrite blockIdx variables
  std::vector<int> otherBlocks;

  /// The constructor
  KernelContext(const KernelInfo& Info, IdxBoundPair&& ThreadIdxInfo)
               : info{Info}, threadIdxInfo{std::move(ThreadIdxInfo)}, blockIdxInfo{}, otherBlocks{} {}

  /// The default constructor
  KernelContext() = default;
  /// Delete copy constructor
  KernelContext(const KernelContext& other) = default;
  /// Delete move constructor
  KernelContext(KernelContext&& other) = default;
  /// Delete copy assignment operator
  KernelContext& operator=(const KernelContext& other) = default;
  /// Delete move assignment operator
  KernelContext& operator=(KernelContext&& other) = default;
};
//---------------------------------------------------------------------------
class FusionTools {
public:
  using IdxBoundPair = KernelContext::IdxBoundPair;

  /// The vector to contain kernel contexts
  std::unordered_map<std::string, KernelContext> kernelContexts;

  /// 
  constexpr static bool baseLine = false;
  ///
  constexpr static bool isBarSyncEnabled = false;
  ///
  constexpr static bool launchBound = false;
  ///
  constexpr static bool imBalancedThread = false;

  /// Create class FusionTools' object
  static FusionTools create(FusionInfo& FInfo, std::map<std::string, KernelInfo>& KInfo);

  /// LibTooling : expand macros
  void expandMacros(clang::tooling::CommonOptionsParser& OptionParser);
  /// LibTooling : rename parameters
  void renameParameters(clang::tooling::CommonOptionsParser& OptionParser);
  /// LibTooling : rewrite thread info
  void rewriteThreadInfo(clang::tooling::CommonOptionsParser& OptionParser);
  /// LibTooling : barrier rewriter
  void barrierRewriter(clang::tooling::CommonOptionsParser& OptionParser);

  /// The constructor
  explicit FusionTools(std::vector<KernelInfo>& Infos);
};
//---------------------------------------------------------------------------
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------