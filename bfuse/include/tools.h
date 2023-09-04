
#pragma once

#include <utility>
#include <string>
#include <vector>
#include <map>
#include <tuple>

#include "bfuse.h"
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
  inline std::tuple<int, const char **>getArguments() const { return std::make_tuple(argc, argv); }

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
  KernelContext(KernelInfo& Info, IdxBoundPair& ThreadIdxInfo,
                std::vector<IdxBoundPair>& BlockIdxInfo, std::vector<int>& OtherBlocks)
                : info{Info}, threadIdxInfo{ThreadIdxInfo},
                  blockIdxInfo{BlockIdxInfo}, otherBlocks{OtherBlocks} {}
};
//---------------------------------------------------------------------------
class FusionTools {
public:
  using IdxBoundPair = KernelContext::IdxBoundPair;

private:
  /// The vector to contain kernel contexts
  std::vector<KernelContext> kernelContexts;

public:
  /// 
  constexpr static bool baseLine = false;
  ///
  constexpr static bool isBarSyncEnabled = false;
  ///
  constexpr static bool launchBound = false;
  ///
  constexpr static bool imBalancedThread = false;

  /// Create FusionTools Object
  static FusionTools create(FusionInfo& FInfo, std::map<std::string, KernelInfo>& KInfo);
  /// Get kernel contexts
  std::vector<KernelContext> getKernelContexts() { return kernelContexts; }

  /// The constructor
  explicit FusionTools(std::vector<KernelInfo>& Infos);
};
//---------------------------------------------------------------------------
} // namespace tools
} // namespace bfuse
//---------------------------------------------------------------------------