
#pragma once

#include <cstdlib>
#include <string>

#include "llvm/Support/YAMLTraits.h"
#include "bfuse.h"
//---------------------------------------------------------------------------
namespace utils {
/*
 * ReadYAMLInfo() - 
 */
template <typename Info>
Info readYAMLInfo(const std::string& Path)
{
  using FileOrError = llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>;

  FileOrError Buffer = llvm::MemoryBuffer::getFile(Path.c_str());
  if (!Buffer) {
    llvm::errs() << "failed to read configs.\n";
    std::abort();
  }

  Info Infos;
  llvm::yaml::Input Yaml{Buffer.get()->getBuffer()};
  Yaml >> Infos;

  if (Yaml.error()) {
    llvm::errs() << "failed to get configs.\n";
    std::abort();
  }

  return Infos;
}
} // namespace utils
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<bfuse::KernelInfo> {
  static void mapping(llvm::yaml::IO &io, bfuse::KernelInfo &info)
  {
    io.mapRequired("KernelName",  info.kernelName);
    io.mapRequired("HasBarriers", info.hasBarriers);
    io.mapRequired("GridDim",     info.gridDim);
    io.mapRequired("BlockDim",    info.blockDim);
    io.mapRequired("Reg",         info.reg);

    io.mapOptional("ExecTime", info.execTime, 0.0);
  }
};
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<bfuse::FusionInfo> {
  static void mapping(llvm::yaml::IO &io, bfuse::FusionInfo &info)
  {
    io.mapRequired("File",    info.filePath);
    io.mapRequired("Kernels", info.kernels);
  }
};
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<bfuse::GridDim> {
  static void mapping(llvm::yaml::IO &io, bfuse::GridDim &dim)
  {
    io.mapRequired("X", dim.x);
    io.mapRequired("Y", dim.y);
    io.mapRequired("Z", dim.z);
  }
};
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<bfuse::BlockDim> {
  static void mapping(llvm::yaml::IO &io, bfuse::BlockDim &dim)
  {
    io.mapRequired("X", dim.x);
    io.mapRequired("Y", dim.y);
    io.mapRequired("Z", dim.z);
  }
};
//---------------------------------------------------------------------------
LLVM_YAML_IS_SEQUENCE_VECTOR(bfuse::FusionInfo)
LLVM_YAML_IS_STRING_MAP(bfuse::KernelInfo)
//---------------------------------------------------------------------------