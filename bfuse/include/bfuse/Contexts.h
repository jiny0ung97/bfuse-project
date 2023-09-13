
#pragma once

#include <string>
#include <vector>
#include <map>

#include "bfuse/Bfuse.h"
//---------------------------------------------------------------------------
namespace bfuse {
namespace contexts {
//---------------------------------------------------------------------------
class KernelContext {
public:
  using IdxBoundPair = std::pair<int, int>;

private:
  /// The kernel's threadIdx boudnary
  IdxBoundPair threadIdxInfo;
  /// The kernel's blockIdx boundary
  std::vector<IdxBoundPair> blockIdxInfo;
  /// # of blocks from other fused kernels
  /// To rewrite blockIdx variables
  std::vector<int> otherBlocks;

public:
  /// The constructor
  explicit KernelContext(IdxBoundPair&& ThreadIdxInfo)
                        : threadIdxInfo{std::move(ThreadIdxInfo)}, blockIdxInfo{}, otherBlocks{} {}

  /// The default constructor
  KernelContext() = default;
  /// The default copy constructor
  KernelContext(const KernelContext& other) = default;
  /// The default move constructor
  KernelContext(KernelContext&& other) = default;
  /// The default copy assignment operator
  KernelContext& operator=(const KernelContext& other) = default;
  /// The default move assignment operator
  KernelContext& operator=(KernelContext&& other) = default;
};
//---------------------------------------------------------------------------
class FusionContext {
private:
  /// The kernels to be fused
  std::vector<std::string> kernels;
  /// The kernel's information
  std::map<std::string, KernelInfo> kernelInfoMap;
  /// The vector to contain kernel contexts
  std::map<std::string, KernelContext> kernelContextMap;

public:
  /// The constructor
  FusionContext(FusionInfo& FInfo, std::map<std::string, KernelInfo>& KInfoMap);
};
//---------------------------------------------------------------------------
} // namespace contexts
} // namespace bfuse
//---------------------------------------------------------------------------