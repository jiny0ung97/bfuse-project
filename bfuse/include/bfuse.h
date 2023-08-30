
#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
//---------------------------------------------------------------------------
namespace bfuse {
/*
 * struct GridDim - 
 */
struct GridDim {
  int x, y, z;
  int size() const { return x * y * z; }

};
//---------------------------------------------------------------------------
/*
 * struct BlockDim - 
 */
struct BlockDim {
  int x, y, z;
  int size() const { return x * y * z; }
};
//---------------------------------------------------------------------------
/*
 * struct KernelInfo - 
 */
struct KernelInfo {
  /*
   * GetInfo2String() - 
   */
  std::string getStringInfo();

  std::string kernelName;
  bool        hasBarriers;
  GridDim     gridDim;
  BlockDim    blockDim;
  
  int    reg;
  double execTime;
};
//---------------------------------------------------------------------------
/*
 * struct FusionInfo - 
 */
struct FusionInfo {
  /*
   * GetInfo2String() - 
   */
  std::string getStringInfo();

  std::string filePath;
  std::vector<std::string> kernels;
};
//---------------------------------------------------------------------------
/*
 * bfuse() - 
 */
void bfuse(const char *ProgName, std::string FusionInfoPath,
           std::string KernelInfoPath, std::string BasePath);
} // namespace bfuse
//---------------------------------------------------------------------------