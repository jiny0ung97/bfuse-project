
#pragma once

#include <utility>
#include <string>
#include <vector>
#include <map>
//---------------------------------------------------------------------------
namespace bfuse {
namespace contexts {
//---------------------------------------------------------------------------
struct GridDim {
  /// Grid's x, y, z dimension
  int x, y, z;
  /// The size of grid dimension
  int size() const { return x * y * z; }

};
//---------------------------------------------------------------------------
struct BlockDim {
  /// Block's x, y, z dimension
  int x, y, z;
  /// The size of block dimension
  int size() const { return x * y * z; }
};
//---------------------------------------------------------------------------
class KernelInfo {
public:
  /// The kernels' file path
  std::string filePath;
  /// Whether the kernel code has synchronization barriers
  bool hasBarriers;
  /// The kernel's grid dimension
  GridDim gridDim;
  /// The kernel's block dimension
  BlockDim blockDim;

  /// Print KernelInfo
  void print(const std::string& KName) const;
};
//---------------------------------------------------------------------------
class FusionInfo {
public:
  /// The kernels to be fused
  std::vector<std::string> kernels;

  /// Print FusionInfo
  void print() const;
};
//---------------------------------------------------------------------------
class KernelContext {
public:
  using IdxBoundPair = std::pair<int, int>;

  /// The kernel's threadIdx boudnary
  IdxBoundPair threadIdxInfo;
  /// The kernel's blockIdx boundary
  std::vector<IdxBoundPair> blockIdxInfo;
  /// # of blocks from other fused kernels
  /// To rewrite blockIdx variables
  std::vector<int> otherBlocks;

  /// The constructor
  explicit KernelContext(IdxBoundPair &&ThreadIdxInfo)
                        : threadIdxInfo{std::move(ThreadIdxInfo)}, blockIdxInfo{}, otherBlocks{} {}
  /// Print KernelContext
  void print() const;

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
public:
  /// The kernels to be fused
  std::vector<std::string> kernels;
  /// The kernel's information
  std::map<std::string, KernelInfo> kernelInfoMap;
  /// The vector to contain kernel contexts
  std::map<std::string, KernelContext> kernelContextMap;

  /// The constructor
  FusionContext(FusionInfo &FInfo, std::map<std::string, KernelInfo> &KInfoMap);
  /// Print FusionContext
  void print() const;

  /// The default constructor
  FusionContext() = default;
  /// The default copy constructor
  FusionContext(const FusionContext &other) = default;
  /// The default move constructor
  FusionContext(FusionContext &&other) = default;
  /// The default copy assignment operator
  FusionContext& operator=(const FusionContext &other) = default;
  /// The default move assignment operator
  FusionContext& operator=(FusionContext &&other) = default;
};
//---------------------------------------------------------------------------
class AnalysisContext {
public:
  using ParamList = std::vector<std::string>;
  using USRsList  = std::vector<std::vector<std::string>>;

  /// The kernels to be fused
  std::vector<std::string> kernels;
  /// The map of function parameters' list
  std::map<std::string, ParamList> ParamListMap;
  /// The map of USRs lists for renaming parameters
  std::map<std::string, USRsList> USRsListMap;

  /// The fused kernel's block boundary
  int BlockBoundary;
  /// The fused kernel's thread boundary
  int ThreadBoundary;

  /// Print AnalysisContext
  void print() const;
};
//---------------------------------------------------------------------------
} // namespace contexts
} // namespace bfuse
//---------------------------------------------------------------------------