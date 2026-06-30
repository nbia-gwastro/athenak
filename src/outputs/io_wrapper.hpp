#ifndef OUTPUTS_IO_WRAPPER_HPP_
#define OUTPUTS_IO_WRAPPER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file io_wrapper.hpp
//  \brief defines a set of small wrapper functions for MPI versus serial outputs.

#include <string>
#include <cstdio>
#include "athena.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
using  IOWrapperFile = MPI_File;
#else
using  IOWrapperFile = FILE*;
#endif

using IOWrapperSizeT = std::uint64_t;

class IOWrapper {
 public:
#if MPI_PARALLEL_ENABLED
  IOWrapper() : fh_(nullptr), comm_(MPI_COMM_WORLD) {}
  void SetCommunicator(MPI_Comm scomm) { comm_=scomm;}
#else
  IOWrapper() {fh_=nullptr;}
#endif
  ~IOWrapper() {}
  // nested type definition of strongly typed/scoped enum in class definition
  enum class FileMode {read, write, append};

  // wrapper functions for basic I/O tasks
  int Open(const char* fname, FileMode rw, bool single_file_per_rank = false);
  std::size_t Read_bytes(void *buf, IOWrapperSizeT size, IOWrapperSizeT count,
                         bool single_file_per_rank = false);
  std::size_t Read_bytes_at(void *buf, IOWrapperSizeT size, IOWrapperSizeT count,
                            IOWrapperSizeT offset, bool single_file_per_rank = false);
  std::size_t Read_bytes_at_all(void *buf, IOWrapperSizeT size, IOWrapperSizeT count,
                                IOWrapperSizeT offset, bool single_file_per_rank = false);
  std::size_t Write_any_type(const void *buf, IOWrapperSizeT count, std::string type,
                             bool single_file_per_rank = false);
  std::size_t Write_any_type_at(const void *buf, IOWrapperSizeT cnt,IOWrapperSizeT offset,
                                std::string datatype, bool single_file_per_rank = false);
  std::size_t Write_any_type_at_all(const void *buf, IOWrapperSizeT cnt,
                                    IOWrapperSizeT offset, std::string datatype,
                                    bool single_file_per_rank = false);
  // Collective write of `count` elements (of `datatype`) from a CONTIGUOUS
  // source buffer into scattered destinations in the file. Destinations are
  // `ndisps` runs of `blocklength` elements, located at offsets
  //   displacement + disps[i] * sizeof(datatype)
  // in the file (i = 0..ndisps-1). Required: count == ndisps * blocklength on
  // each rank.
  //
  // On MPI builds, internally builds an MPI_Type_create_indexed_block filetype,
  // calls MPI_File_set_view + MPI_File_write_at_all, then resets the view to
  // default. ALL ranks in the file's communicator must call this together;
  // ranks with no data pass count=0, ndisps=0 (they still participate in the
  // collective). Falls back to per-segment fseek+fwrite for non-MPI or
  // single_file_per_rank builds (not collective there).
  std::size_t Write_indexed_at_all(const void *buf, IOWrapperSizeT count,
                                    std::string datatype,
                                    IOWrapperSizeT displacement,
                                    int blocklength,
                                    const int *disps, int ndisps,
                                    bool single_file_per_rank = false);
  std::size_t Read_Reals(void *buf, IOWrapperSizeT count,
                         bool single_file_per_rank = false);
  std::size_t Read_Reals_at(void *buf, IOWrapperSizeT count, IOWrapperSizeT offset,
                            bool single_file_per_rank = false);
  std::size_t Read_Reals_at_all(void *buf, IOWrapperSizeT count, IOWrapperSizeT offset,
                                bool single_file_per_rank = false);
  int Close(bool single_file_per_rank = false);
  int Seek(IOWrapperSizeT offset, bool single_file_per_rank = false);
  IOWrapperSizeT GetPosition(bool single_file_per_rank = false);

 private:
  IOWrapperFile fh_;
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm_;
#endif
};
#endif // OUTPUTS_IO_WRAPPER_HPP_
