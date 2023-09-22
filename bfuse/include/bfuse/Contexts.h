
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
  /// Num of blocks from other fused kernels
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
  // FusionContext(FusionInfo &FInfo, std::map<std::string, KernelInfo> &KInfoMap);
  /// The constructor
  FusionContext(std::vector<std::string> &&OtherKernels, std::map<std::string, KernelInfo> &&OtherKernelInfoMap,
                std::map<std::string, KernelContext> &&OtherKernelContextMap)
                : kernels{std::move(OtherKernels)}, kernelInfoMap{std::move(OtherKernelInfoMap)},
                  kernelContextMap{std::move(OtherKernelContextMap)} {}
  /// Print FusionContext
  void print() const;

  /// Create the KernelContext
  static FusionContext create(FusionInfo &FInfo, std::map<std::string, KernelInfo> &KInfoMap);
};
//---------------------------------------------------------------------------
class AnalysisContext {
public:
  using VarList   = std::vector<std::string>;
  using USRsList  = std::vector<std::vector<std::string>>;
  using SizeList  = std::vector<uint64_t>;

  /// The kernels to be fused
  std::vector<std::string> Kernels;
  /// The map of threads' number
  std::map<std::string, int> ThreadNumMap;
  /// The Branch Condition
  std::map<std::string, std::string> BranchConditionMap;
  /// The map of temp blockIdx, gridDim declarations
  std::string TmpBlockInfoString;
  /// The map of new blockIdx, gridDim declarations
  std::string NewBlockInfoString;
  /// The name of fused kernel
  std::string NewFuncName;
  /// The max threads bound
  int MaxThreadBound;

  /// Renaming Parameters ///
  /// The map of function parameters' list
  std::map<std::string, VarList> ParmListMap;
  /// The map of USRs lists for renaming parameters
  std::map<std::string, USRsList> ParmUSRsListMap;

  /// Renaming Shared memory Variables ///
  /// The string of shared memory declarations
  std::map<std::string, std::string> SharedDeclStringMap;
  /// The map of shared memory variables' list
  std::map<std::string, VarList> ShrdVarListMap;
  /// The map of USRs lists for renaming shared memory variables
  std::map<std::string, USRsList> ShrdVarUSRsListMap;
  /// The map of shared memory variables' size
  std::map<std::string, SizeList> ShrdVarSizeListMap;
  /// 
  std::string NewShrdDeclString;

  /// The constructor
  AnalysisContext(std::vector<std::string> &&OtherKernels, std::map<std::string, int> &&OtherThreadNumMap,
                  std::map<std::string, std::string> &&OtherBranchConditionMap,
                  std::string &&OtherTmpBlockInfoString, std::string &&OtherNewBlockInfoString,
                  std::string &&OtherNewFuncName, int OtherMaxThreadBound)
                  : Kernels{OtherKernels}, ThreadNumMap{OtherThreadNumMap},
                    BranchConditionMap{OtherBranchConditionMap},
                    TmpBlockInfoString{OtherTmpBlockInfoString},
                    NewBlockInfoString{OtherNewBlockInfoString},
                    NewFuncName{OtherNewFuncName}, MaxThreadBound{OtherMaxThreadBound} {}
  /// Print AnalysisContext
  void print() const;

  /// Create the AnalysisContext
  static AnalysisContext create(FusionContext &FContext);
};
//---------------------------------------------------------------------------
} // namespace contexts
} // namespace bfuse
//---------------------------------------------------------------------------