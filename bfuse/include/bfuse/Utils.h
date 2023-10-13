
#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <map>

#include "llvm/Support/YAMLTraits.h"

#include "bfuse/Contexts.h"

// TODO: need to be changed into cxx style.
#define ERROR_MESSAGE(m)                   \
  do                                       \
  {                                        \
    std::cerr << __FILE__ << ":"           \
              << __LINE__ << ": "          \
              << "bfuse error: "           \
              << m << "\n";                \
  } while (0)
//---------------------------------------------------------------------------
namespace bfuse {
namespace utils {
//---------------------------------------------------------------------------
std::string extractFilePath(contexts::FusionInfo& FInfo);
//---------------------------------------------------------------------------
void backUpFiles(const std::string &FileName);
//---------------------------------------------------------------------------
void recoverFiles(const std::string &FileName);
//---------------------------------------------------------------------------
template <typename Info>
Info readYAMLInfo(const std::string &Path)
{
  auto Buffer = llvm::MemoryBuffer::getFile(Path.c_str());
  if (!Buffer) {
    llvm::errs() << "[bfuse ERROR]: failed to read configs.\n";
    std::exit(0);
  }

  Info Infos;
  llvm::yaml::Input Yaml{Buffer.get()->getBuffer()};
  Yaml >> Infos;

  if (Yaml.error()) {
    llvm::errs() << "[bfuse ERROR]: failed to get configs.\n";
    std::exit(0);
  }

  return Infos;
}
//---------------------------------------------------------------------------
} // namespace utils
} // namespace bfuse
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<bfuse::contexts::KernelInfo> {
  static void mapping(llvm::yaml::IO &Io, bfuse::contexts::KernelInfo &Info)
  {
    Io.mapRequired("KernelName",  Info.kernelName);
    Io.mapRequired("HasBarriers", Info.hasBarriers);
    Io.mapRequired("GridDim",     Info.gridDim);
    Io.mapRequired("BlockDim",    Info.blockDim);
    Io.mapRequired("Reg",         Info.reg);
    Io.mapOptional("ExecTime",    Info.execTime, -1);
  }
};
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<bfuse::contexts::FusionInfo> {
  static void mapping(llvm::yaml::IO &Io, bfuse::contexts::FusionInfo &Info)
  {
    Io.mapRequired("File",    Info.filePath);
    Io.mapRequired("Kernels", Info.kernels);
  }
};
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<bfuse::contexts::GridDim> {
  static void mapping(llvm::yaml::IO &Io, bfuse::contexts::GridDim &Dim)
  {
    Io.mapRequired("X", Dim.x);
    Io.mapRequired("Y", Dim.y);
    Io.mapRequired("Z", Dim.z);
  }
};
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<bfuse::contexts::BlockDim> {
  static void mapping(llvm::yaml::IO &Io, bfuse::contexts::BlockDim &Dim)
  {
    Io.mapRequired("X", Dim.x);
    Io.mapRequired("Y", Dim.y);
    Io.mapRequired("Z", Dim.z);
  }
};
//---------------------------------------------------------------------------
LLVM_YAML_IS_SEQUENCE_VECTOR(bfuse::contexts::FusionInfo)
LLVM_YAML_IS_STRING_MAP(bfuse::contexts::KernelInfo)
//---------------------------------------------------------------------------