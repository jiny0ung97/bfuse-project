
#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <map>

#include "llvm/Support/YAMLTraits.h"

#include "fuse/Contexts.h"

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
namespace fuse {
namespace utils {
//---------------------------------------------------------------------------
std::string extractFilePath(contexts::FusionInfo& FInfo);
//---------------------------------------------------------------------------
void backUpFile(const std::string &FileName);
//---------------------------------------------------------------------------
void recoverFile(const std::string &FileName);
//---------------------------------------------------------------------------
void writeFile(const std::string &Path, const std::string &FileName, const std::string &Str);
//---------------------------------------------------------------------------
template <typename Info>
Info readYAMLInfo(const std::string &Path)
{
  auto Buffer = llvm::MemoryBuffer::getFile(Path.c_str());
  if (!Buffer) {
    llvm::errs() << "readYAMLInfo: failed to read configs.\n";
    std::exit(0);
  }

  Info Infos;
  llvm::yaml::Input Yaml{Buffer.get()->getBuffer()};

  Yaml >> Infos;
  if (Yaml.error()) {
    llvm::errs() << "readYAMLInfo: failed to get configs.\n";
    std::exit(0);
  }

  return Infos;
}
//---------------------------------------------------------------------------
template <typename Info>
void writeYAMLInfo(const std::string &Path, const std::string &FileName, Info &Infos)
{
  std::error_code FileErr;
  llvm::raw_fd_ostream Os(Path + "/" + FileName, FileErr, llvm::sys::fs::OF_None);

  llvm::yaml::Output Yaml{Os};

  Yaml << Infos;
}
//---------------------------------------------------------------------------
} // namespace utils
} // namespace fuse
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<fuse::contexts::KernelInfo> {
  static void mapping(llvm::yaml::IO &Io, fuse::contexts::KernelInfo &Info)
  {
    Io.mapRequired("KernelName",  Info.KernelName_);
    Io.mapRequired("HasBarriers", Info.HasBarriers_);
    Io.mapRequired("GridDim",     Info.GridDim_);
    Io.mapRequired("BlockDim",    Info.BlockDim_);
    Io.mapRequired("Reg",         Info.Reg_);
    Io.mapOptional("ExecTime",    Info.ExecTime_, -1);
  }
};
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<fuse::contexts::FusionInfo> {
  static void mapping(llvm::yaml::IO &Io, fuse::contexts::FusionInfo &Info)
  {
    Io.mapRequired("File",    Info.File_);
    Io.mapRequired("Kernels", Info.Kernels_);
  }
};
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<fuse::contexts::GridDim> {
  static void mapping(llvm::yaml::IO &Io, fuse::contexts::GridDim &Dim)
  {
    Io.mapRequired("X", Dim.X);
    Io.mapRequired("Y", Dim.Y);
    Io.mapRequired("Z", Dim.Z);
  }
};
//---------------------------------------------------------------------------
template <>
struct llvm::yaml::MappingTraits<fuse::contexts::BlockDim> {
  static void mapping(llvm::yaml::IO &Io, fuse::contexts::BlockDim &Dim)
  {
    Io.mapRequired("X", Dim.X);
    Io.mapRequired("Y", Dim.Y);
    Io.mapRequired("Z", Dim.Z);
  }
};
//---------------------------------------------------------------------------
LLVM_YAML_IS_SEQUENCE_VECTOR(fuse::contexts::FusionInfo)
LLVM_YAML_IS_STRING_MAP(fuse::contexts::KernelInfo)
//---------------------------------------------------------------------------