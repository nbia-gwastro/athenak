//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file boundary_face.cpp
//! \brief BoundaryFaceOutput: dumps the first ghost-cell layer on one physical
//! face per output cadence in BCTABLE-shaped per-snapshot files. A standalone
//! concatenator stitches snapshots into the time-series BCTABLE file consumed by
//! a BCTableReader-style problem generator. Fixed 7-variable set
//! (dens, mom1, mom2, mom3, bfc1, bfc2, bfc3); root-level boundary blocks only.

#include <sys/stat.h>     // mkdir
#include <algorithm>      // std::sort
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>        // std::pair
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "mhd/mhd.hpp"
#include "outputs.hpp"

namespace {
constexpr char     BCFACE_MAGIC[8]   = {'B','C','T','A','B','L','E','\0'};
constexpr uint32_t BCFACE_VERSION    = 1u;
constexpr int      BCFACE_NVAR       = 7;
// Header layout (packed, native LE; matches `BCTableHeader` in the pgen reader
// under `#pragma pack(push,1)`):
//   char magic[8] + 7×uint32 fields + uint32 reserved[15] = 8 + 28 + 60 = 96 bytes.
// The comment "128-byte packed header" in construct_tables_mhd_paral.py is
// stale; struct.calcsize('<8sIIIIIII15I') == 96.
constexpr int      BCFACE_HDR_BYTES  = 96;
// Files are written under bin/<file_id>/<chunk>/... where
// chunk = file_number / BCFACE_CHUNK_SIZE. Bounds the per-directory inode count
// so Lustre / glob / ls performance stays sane at 100k+ snapshot runs.
constexpr int      BCFACE_CHUNK_SIZE = 10000;
}  // anonymous namespace

//----------------------------------------------------------------------------------------
// BoundaryFaceOutput constructor — parses athinput parameters, validates,
// caches per-rank geometry, and pre-allocates the row packing buffer.

BoundaryFaceOutput::BoundaryFaceOutput(ParameterInput *pin, Mesh *pm,
                                       OutputParameters op) :
    BaseTypeOutput(pin, pm, op) {
  // Top-level output directory; per-face subdir is created once file_id is
  // known (below). Chunk subdirs are created lazily in WriteOutputFile.
  mkdir("bin", 0775);

  // ---- 1. parse 'face' (required) ----
  std::string face = pin->GetString(op.block_name, "face");
  if      (face == "ix1") { face_id_ = 0; }
  else if (face == "ox1") { face_id_ = 1; }
  else if (face == "ix2") { face_id_ = 2; }
  else if (face == "ox2") { face_id_ = 3; }
  else if (face == "ix3") { face_id_ = 4; }
  else if (face == "ox3") { face_id_ = 5; }
  else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Output block '" << op.block_name << "' has face='" << face
              << "'. Must be one of: ix1, ox1, ix2, ox2, ix3, ox3." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // default file_id (since this output type bypasses the standard 'variable'
  // mechanism, out_params.file_id is otherwise empty/undefined here)
  if (out_params.file_id.empty()) {
    out_params.file_id = std::string("bcface_") + face;
  }

  // Per-face output directory (one level below `bin/`). Chunked subdirs that
  // bound the per-directory file count are created lazily in WriteOutputFile.
  mkdir((std::string("bin/") + out_params.file_id).c_str(), 0775);

  // ---- 2. parse 'dtype' (optional, default float64) ----
  std::string dtype = pin->GetOrAddString(op.block_name, "dtype", "float64");
  if (dtype == "float64") {
    dtype_code_ = 1; elem_size_ = static_cast<int>(sizeof(double));
  } else if (dtype == "float32") {
    dtype_code_ = 2; elem_size_ = static_cast<int>(sizeof(float));
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Output block '" << op.block_name << "' has dtype='" << dtype
              << "'. Must be 'float64' or 'float32'." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // ---- 3. require MHD (writer assumes pmhd->u0 and pmhd->b0 exist) ----
  if (pm->pmb_pack->pmhd == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Output block '" << op.block_name
              << "' (file_type=bcface) requires MHD, but no <mhd> block is configured."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // ---- 4. dimensionality compatibility ----
  if (pm->one_d && face_id_ >= 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Output block '" << op.block_name << "': face=" << face
              << " is not compatible with a 1D mesh." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pm->two_d && face_id_ >= 4) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Output block '" << op.block_name << "': face=" << face
              << " is not compatible with a 2D mesh." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // ---- 5. global raster (N1, N2) + per-block window (window_a, window_b) ----
  // Convention matches the legacy python concatenator construct_tables_mhd_paral.py:
  //   * For an x1-face the perp dims are (x2, x3) with x2 the faster index.
  //   * For an x2-face the perp dims are (x1, x3) with x1 the faster index.
  //   * For an x3-face the perp dims are (x1, x2) with x1 the faster index.
  // The window in each perp dim includes ghost cells (nx + 2*ng) so corner ghosts
  // are sampled too — same as the python which uses the full meshblock extent.
  const auto &indcs = pm->mb_indcs;
  const int ng  = indcs.ng;
  const int nc1 = indcs.nx1 + 2 * ng;
  const int nc2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  const int nc3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;

  if (face_id_ < 2) {              // ix1 / ox1
    window_a_ = static_cast<uint32_t>(nc2);
    window_b_ = static_cast<uint32_t>(nc3);
    N1_ = static_cast<uint32_t>(pm->nmb_rootx2) * window_a_;
    N2_ = static_cast<uint32_t>(pm->nmb_rootx3) * window_b_;
  } else if (face_id_ < 4) {       // ix2 / ox2
    window_a_ = static_cast<uint32_t>(nc1);
    window_b_ = static_cast<uint32_t>(nc3);
    N1_ = static_cast<uint32_t>(pm->nmb_rootx1) * window_a_;
    N2_ = static_cast<uint32_t>(pm->nmb_rootx3) * window_b_;
  } else {                          // ix3 / ox3
    window_a_ = static_cast<uint32_t>(nc1);
    window_b_ = static_cast<uint32_t>(nc2);
    N1_ = static_cast<uint32_t>(pm->nmb_rootx1) * window_a_;
    N2_ = static_cast<uint32_t>(pm->nmb_rootx2) * window_b_;
  }

  // ---- 6. build face_blocks_ and (re)allocate device packing buffers ----
  // rebuild_face_blocks also sizes slab_d_ / m_local_d_ / m_local_h_ to match
  // the current number of boundary blocks on this rank.
  rebuild_face_blocks(pm);
  cached_nmb_thispack_ = pm->pmb_pack->nmb_thispack;

  // disps_ and permuted_buf_ are resized lazily in WriteOutputFile based on
  // the current nseg/total_elems; no pre-allocation needed here.

  if (global_variable::my_rank == 0) {
    std::cout << "BoundaryFaceOutput[" << op.block_name << "]: face=" << face
              << " face_id=" << face_id_ << " dtype=" << dtype
              << " nvar=" << BCFACE_NVAR << " N1=" << N1_ << " N2=" << N2_
              << " window_a=" << window_a_ << " window_b=" << window_b_ << std::endl;
  }
}

//----------------------------------------------------------------------------------------
// Rebuild face_blocks_ purely from geometry. Called at construction and at every
// output write to cover load-balance shuffles between ranks. Throws fatal if a
// block on the requested face is at a non-root refinement level.

void BoundaryFaceOutput::rebuild_face_blocks(Mesh *pm) {
  face_blocks_.clear();
  const int nmb   = pm->pmb_pack->nmb_thispack;
  const int igids = pm->pmb_pack->gids;
  const int rl    = pm->root_level;
  const int nmbx1 = pm->nmb_rootx1;
  const int nmbx2 = pm->nmb_rootx2;
  const int nmbx3 = pm->nmb_rootx3;

  for (int m = 0; m < nmb; ++m) {
    const LogicalLocation &loc = pm->lloc_eachmb[igids + m];

    bool on_face = false;
    switch (face_id_) {
      case 0: on_face = (loc.lx1 == 0);           break;
      case 1: on_face = (loc.lx1 == nmbx1 - 1);   break;
      case 2: on_face = (loc.lx2 == 0);           break;
      case 3: on_face = (loc.lx2 == nmbx2 - 1);   break;
      case 4: on_face = (loc.lx3 == 0);           break;
      case 5: on_face = (loc.lx3 == nmbx3 - 1);   break;
    }
    if (!on_face) continue;

    if (loc.level != rl) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "BoundaryFaceOutput[" << out_params.block_name
                << "]: meshblock gid=" << (igids + m)
                << " on face_id=" << face_id_ << " is at refinement level "
                << loc.level << " (root_level=" << rl
                << "). Only root-level boundary blocks are supported." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    FaceBlock fb;
    fb.m_local = m;
    switch (face_id_) {
      case 0: case 1: fb.lx_a = static_cast<uint32_t>(loc.lx2);
                      fb.lx_b = static_cast<uint32_t>(loc.lx3); break;
      case 2: case 3: fb.lx_a = static_cast<uint32_t>(loc.lx1);
                      fb.lx_b = static_cast<uint32_t>(loc.lx3); break;
      case 4: case 5: fb.lx_a = static_cast<uint32_t>(loc.lx1);
                      fb.lx_b = static_cast<uint32_t>(loc.lx2); break;
    }
    face_blocks_.push_back(fb);
  }

  // (Re)allocate device packing buffers to match nblocks. Only realloc when
  // nblocks actually changes — for root-level-only setups it's essentially
  // static across the run, so this is normally a one-shot at startup.
  const int n = static_cast<int>(face_blocks_.size());
  if (n > 0) {
    if (static_cast<int>(m_local_d_.extent(0)) != n) {
      Kokkos::realloc(slab_d_,    n, BCFACE_NVAR,
                                  static_cast<int>(window_b_),
                                  static_cast<int>(window_a_));
      Kokkos::realloc(m_local_d_, n);
      Kokkos::realloc(m_local_h_, n);
    }
    for (int i = 0; i < n; ++i) m_local_h_(i) = face_blocks_[i].m_local;
    Kokkos::deep_copy(m_local_d_, m_local_h_);
  }
}

//----------------------------------------------------------------------------------------
// LoadOutputData: nothing to stage — the device packing kernel and the disk
// write happen together in WriteOutputFile. Driver calls Load+Write in
// lockstep on the same trigger (see driver.cpp).

void BoundaryFaceOutput::LoadOutputData(Mesh * /*pm*/) {}

//----------------------------------------------------------------------------------------
// WriteOutputFile: produce one per-snapshot file containing the 96-byte BCTABLE
// header, the snapshot time, and the (nvar, N2, N1) payload assembled from this
// rank's contribution to the global face raster.

void BoundaryFaceOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  // Always rebuild block ownership before writing — covers load-balance shuffles
  // of root-level blocks across ranks that AMR refinement elsewhere may have
  // triggered. Cost is O(nmb_thispack); negligible.
  rebuild_face_blocks(pm);
  cached_nmb_thispack_ = pm->pmb_pack->nmb_thispack;

  // ---- filename ----
  // Path layout: bin/<file_id>/<chunk>/<basename>.<file_id>.<NNNNN>.bin
  // where chunk = file_number / BCFACE_CHUNK_SIZE. Sharding keeps any single
  // directory's inode count to ~BCFACE_CHUNK_SIZE entries so glob / ls / rm
  // stay snappy at 100k+ snapshot runs. The chunk subdir is created lazily
  // only when file_number crosses a chunk boundary.
  const int chunk = out_params.file_number / BCFACE_CHUNK_SIZE;
  char chunk_str[12];
  std::snprintf(chunk_str, sizeof(chunk_str), "%05d", chunk);
  if (chunk != last_chunk_dir_) {
    const std::string chunk_path =
        std::string("bin/") + out_params.file_id + "/" + chunk_str;
    mkdir(chunk_path.c_str(), 0775);
    last_chunk_dir_ = chunk;
  }

  char number[12];
  std::snprintf(number, sizeof(number), ".%05d", out_params.file_number);
  std::string fname = std::string("bin/") + out_params.file_id + "/"
                    + chunk_str + "/"
                    + out_params.file_basename + "." + out_params.file_id
                    + number + ".bin";

  // ---- open (collective) ----
  IOWrapper binfile;
  binfile.Open(fname.c_str(), IOWrapper::FileMode::write,
               /*single_file_per_rank=*/false);

  // ---- rank 0 writes header + 1 time value ----
  if (global_variable::my_rank == 0) {
    char hdr[BCFACE_HDR_BYTES];
    std::memset(hdr, 0, sizeof(hdr));

    char *p = hdr;
    auto put_bytes = [&](const void *src, std::size_t n) {
      std::memcpy(p, src, n); p += n;
    };
    auto put32 = [&](uint32_t v) { put_bytes(&v, sizeof(v)); };

    put_bytes(BCFACE_MAGIC, 8);
    put32(BCFACE_VERSION);
    put32(static_cast<uint32_t>(dtype_code_));
    put32(static_cast<uint32_t>(BCFACE_NVAR));
    put32(1u);                                  // nt = 1 (per-snapshot file)
    put32(N2_);
    put32(N1_);
    put32(static_cast<uint32_t>(face_id_));
    // reserved[15] left zeroed by memset above

    binfile.Write_any_type_at(hdr, BCFACE_HDR_BYTES, 0, "byte",
                              /*single_file_per_rank=*/false);

    if (dtype_code_ == 1) {
      double t = static_cast<double>(pm->time);
      binfile.Write_any_type_at(&t, 1, BCFACE_HDR_BYTES, "double", false);
    } else {
      float t = static_cast<float>(pm->time);
      binfile.Write_any_type_at(&t, 1, BCFACE_HDR_BYTES, "float", false);
    }
  }

  const std::size_t payload_off = static_cast<std::size_t>(BCFACE_HDR_BYTES)
                                + static_cast<std::size_t>(elem_size_);

  // ---- if this rank owns boundary blocks, pack on device and write ----
  if (!face_blocks_.empty()) {
    // The kernel below fills slab_d_ (a small (nblocks, 7, window_b, window_a)
    // buffer) by scattering reads out of u0 and b0.{x1f,x2f,x3f}. We then mirror
    // ONLY this slab to host — a few MB regardless of how large u0/b0 are.
    // This replaces the previous full-array host mirror, which was the dominant
    // cost when running at every cycle (dcycle=1).
    //
    // Index conventions match the python concatenator and the pgen reader:
    //   * The derived 'bfc1' variable maps to b0.x1f(m,k,j,i) at "cell i"
    //     (dv(m,0,k,j,i) = b0.x1f(m,k,j,i)), so sampling bfc1 at cell index i
    //     means reading b0.x1f at face index i. Same logic for bfc2 / bfc3.
    //   * The perpendicular-to-face B uses the *outer* face of the first ghost
    //     cell (the face shared with ghost #2). For ix1: b0.x1f(...,is-1);
    //     for ox1: b0.x1f(...,ie+2). The pgen writes the table value back to
    //     b0.x1f(...,ie+i+2) for i=0,1, which closes the loop.
    //   * Parallel-to-face B components use the same cell index as the
    //     conserved variables.
    //
    // One par_for per face — keeps every kernel branchless, matches the pattern
    // used by src/outputs/derived_variables.cpp.

    const int nblocks = static_cast<int>(face_blocks_.size());
    const auto &indcs = pm->mb_indcs;
    const int is = indcs.is, ie = indcs.ie;
    const int js = indcs.js, je = indcs.je;
    const int ks = indcs.ks, ke = indcs.ke;
    const int wa = static_cast<int>(window_a_);
    const int wb = static_cast<int>(window_b_);

    auto u0   = pm->pmb_pack->pmhd->u0;
    auto x1f  = pm->pmb_pack->pmhd->b0.x1f;
    auto x2f  = pm->pmb_pack->pmhd->b0.x2f;
    auto x3f  = pm->pmb_pack->pmhd->b0.x3f;
    auto slab = slab_d_;
    auto ml   = m_local_d_;

    switch (face_id_) {
      case 0: {  // ix1   perp axes: jj = j_index (fast), kk = k_index (slow)
        par_for("bcface_pack_ix1", DevExeSpace(),
          0, nblocks - 1, 0, wb - 1, 0, wa - 1,
          KOKKOS_LAMBDA(int b, int kk, int jj) {
            const int m = ml(b);
            const int k = kk, j = jj;
            const int i_c  = is - 1;
            const int i_bx = is - 1;
            slab(b, 0, kk, jj) = u0(m, IDN, k, j, i_c);
            slab(b, 1, kk, jj) = u0(m, IM1, k, j, i_c);
            slab(b, 2, kk, jj) = u0(m, IM2, k, j, i_c);
            slab(b, 3, kk, jj) = u0(m, IM3, k, j, i_c);
            slab(b, 4, kk, jj) = x1f(m, k, j, i_bx);
            slab(b, 5, kk, jj) = x2f(m, k, j, i_c);
            slab(b, 6, kk, jj) = x3f(m, k, j, i_c);
          });
        break;
      }
      case 1: {  // ox1
        par_for("bcface_pack_ox1", DevExeSpace(),
          0, nblocks - 1, 0, wb - 1, 0, wa - 1,
          KOKKOS_LAMBDA(int b, int kk, int jj) {
            const int m = ml(b);
            const int k = kk, j = jj;
            const int i_c  = ie + 1;
            const int i_bx = ie + 2;
            slab(b, 0, kk, jj) = u0(m, IDN, k, j, i_c);
            slab(b, 1, kk, jj) = u0(m, IM1, k, j, i_c);
            slab(b, 2, kk, jj) = u0(m, IM2, k, j, i_c);
            slab(b, 3, kk, jj) = u0(m, IM3, k, j, i_c);
            slab(b, 4, kk, jj) = x1f(m, k, j, i_bx);
            slab(b, 5, kk, jj) = x2f(m, k, j, i_c);
            slab(b, 6, kk, jj) = x3f(m, k, j, i_c);
          });
        break;
      }
      case 2: {  // ix2   perp axes: jj = i_index (fast), kk = k_index (slow)
        par_for("bcface_pack_ix2", DevExeSpace(),
          0, nblocks - 1, 0, wb - 1, 0, wa - 1,
          KOKKOS_LAMBDA(int b, int kk, int jj) {
            const int m = ml(b);
            const int k = kk, i = jj;
            const int j_c  = js - 1;
            const int j_by = js - 1;
            slab(b, 0, kk, jj) = u0(m, IDN, k, j_c, i);
            slab(b, 1, kk, jj) = u0(m, IM1, k, j_c, i);
            slab(b, 2, kk, jj) = u0(m, IM2, k, j_c, i);
            slab(b, 3, kk, jj) = u0(m, IM3, k, j_c, i);
            slab(b, 4, kk, jj) = x1f(m, k, j_c, i);
            slab(b, 5, kk, jj) = x2f(m, k, j_by, i);
            slab(b, 6, kk, jj) = x3f(m, k, j_c, i);
          });
        break;
      }
      case 3: {  // ox2
        par_for("bcface_pack_ox2", DevExeSpace(),
          0, nblocks - 1, 0, wb - 1, 0, wa - 1,
          KOKKOS_LAMBDA(int b, int kk, int jj) {
            const int m = ml(b);
            const int k = kk, i = jj;
            const int j_c  = je + 1;
            const int j_by = je + 2;
            slab(b, 0, kk, jj) = u0(m, IDN, k, j_c, i);
            slab(b, 1, kk, jj) = u0(m, IM1, k, j_c, i);
            slab(b, 2, kk, jj) = u0(m, IM2, k, j_c, i);
            slab(b, 3, kk, jj) = u0(m, IM3, k, j_c, i);
            slab(b, 4, kk, jj) = x1f(m, k, j_c, i);
            slab(b, 5, kk, jj) = x2f(m, k, j_by, i);
            slab(b, 6, kk, jj) = x3f(m, k, j_c, i);
          });
        break;
      }
      case 4: {  // ix3   perp axes: jj = i_index (fast), kk = j_index (slow)
        par_for("bcface_pack_ix3", DevExeSpace(),
          0, nblocks - 1, 0, wb - 1, 0, wa - 1,
          KOKKOS_LAMBDA(int b, int kk, int jj) {
            const int m = ml(b);
            const int j = kk, i = jj;
            const int k_c  = ks - 1;
            const int k_bz = ks - 1;
            slab(b, 0, kk, jj) = u0(m, IDN, k_c, j, i);
            slab(b, 1, kk, jj) = u0(m, IM1, k_c, j, i);
            slab(b, 2, kk, jj) = u0(m, IM2, k_c, j, i);
            slab(b, 3, kk, jj) = u0(m, IM3, k_c, j, i);
            slab(b, 4, kk, jj) = x1f(m, k_c, j, i);
            slab(b, 5, kk, jj) = x2f(m, k_c, j, i);
            slab(b, 6, kk, jj) = x3f(m, k_bz, j, i);
          });
        break;
      }
      case 5: {  // ox3
        par_for("bcface_pack_ox3", DevExeSpace(),
          0, nblocks - 1, 0, wb - 1, 0, wa - 1,
          KOKKOS_LAMBDA(int b, int kk, int jj) {
            const int m = ml(b);
            const int j = kk, i = jj;
            const int k_c  = ke + 1;
            const int k_bz = ke + 2;
            slab(b, 0, kk, jj) = u0(m, IDN, k_c, j, i);
            slab(b, 1, kk, jj) = u0(m, IM1, k_c, j, i);
            slab(b, 2, kk, jj) = u0(m, IM2, k_c, j, i);
            slab(b, 3, kk, jj) = u0(m, IM3, k_c, j, i);
            slab(b, 4, kk, jj) = x1f(m, k_c, j, i);
            slab(b, 5, kk, jj) = x2f(m, k_c, j, i);
            slab(b, 6, kk, jj) = x3f(m, k_bz, j, i);
          });
        break;
      }
    }  // switch face_id_

    // Mirror only the small slab to host (a few MB at most). deep_copy is
    // blocking on the device queue, so no explicit Kokkos::fence is needed.
    auto slab_h = Kokkos::create_mirror_view(slab_d_);
    Kokkos::deep_copy(slab_h, slab_d_);

    // Build (file_displacement, source_segment_index) pairs in natural order,
    // sort by displacement, then permute the source data into permuted_buf_ in
    // sorted order. MPI requires filetype displacements to be monotonically
    // non-decreasing for file views (MPI standard 14.3.2); the natural
    // (b_idx, v, kk) iteration order does NOT satisfy this because face_blocks_
    // can be in arbitrary (lx_a, lx_b) order. The sort + permute is O(nseg log
    // nseg) + O(total_elems) per cadence — typically well under 1 ms.
    const int nseg = nblocks * BCFACE_NVAR * static_cast<int>(window_b_);
    const std::size_t total_elems =
        static_cast<std::size_t>(nseg) * window_a_;

    // pair: (file displacement in oldtype units, segment index in slab_h order)
    std::vector<std::pair<int, int>> entries;
    entries.reserve(nseg);
    int src_idx = 0;
    for (int b_idx = 0; b_idx < nblocks; ++b_idx) {
      const FaceBlock &fb = face_blocks_[b_idx];
      const int jj_base = static_cast<int>(fb.lx_a) * static_cast<int>(window_a_);
      const int kk_base = static_cast<int>(fb.lx_b) * static_cast<int>(window_b_);
      for (int v = 0; v < BCFACE_NVAR; ++v) {
        for (int kk = 0; kk < static_cast<int>(window_b_); ++kk) {
          const int disp =
              (v * static_cast<int>(N2_) + kk_base + kk) * static_cast<int>(N1_)
              + jj_base;
          entries.emplace_back(disp, src_idx);
          ++src_idx;
        }
      }
    }

    std::sort(entries.begin(), entries.end(),
              [](const std::pair<int,int> &a, const std::pair<int,int> &b) {
                return a.first < b.first;
              });

    // (Re)size persistent buffers
    disps_.resize(nseg);
    const std::size_t total_bytes = total_elems * static_cast<std::size_t>(elem_size_);
    if (permuted_buf_.size() < total_bytes) permuted_buf_.resize(total_bytes);

    // Permute slab_h into permuted_buf_ in sorted order; down-cast to float in
    // the same loop if dtype_code_ == 2.
    const Real *slab_ptr = slab_h.data();
    const std::size_t wa_z = window_a_;     // size_t alias (avoids shadowing the
                                            // `int wa` used by the kernel above)
    if (dtype_code_ == 1) {
      double *dst = reinterpret_cast<double*>(permuted_buf_.data());
      for (int i = 0; i < nseg; ++i) {
        disps_[i] = entries[i].first;
        const Real *seg_src = slab_ptr +
            static_cast<std::size_t>(entries[i].second) * wa_z;
        double *seg_dst = dst + static_cast<std::size_t>(i) * wa_z;
        for (std::size_t j = 0; j < wa_z; ++j) seg_dst[j] = seg_src[j];
      }
    } else {
      float *dst = reinterpret_cast<float*>(permuted_buf_.data());
      for (int i = 0; i < nseg; ++i) {
        disps_[i] = entries[i].first;
        const Real *seg_src = slab_ptr +
            static_cast<std::size_t>(entries[i].second) * wa_z;
        float *seg_dst = dst + static_cast<std::size_t>(i) * wa_z;
        for (std::size_t j = 0; j < wa_z; ++j) {
          seg_dst[j] = static_cast<float>(seg_src[j]);
        }
      }
    }

    // ONE collective MPI-IO call writes the whole rank's contribution via a
    // derived indexed_block filetype. All ranks (including those with no
    // boundary blocks — handled by the `else` branch below) participate.
    const char *dtype_str = (dtype_code_ == 1) ? "double" : "float";
    binfile.Write_indexed_at_all(
        permuted_buf_.data(), total_elems, dtype_str,
        payload_off,
        static_cast<int>(window_a_),
        disps_.data(), nseg,
        /*single_file_per_rank=*/false);
  } else {
    // No boundary blocks on this rank, but Write_indexed_at_all is collective
    // over the file's communicator (set_view + write_at_all). Participate with
    // an empty filetype.
    const char *dtype_str = (dtype_code_ == 1) ? "double" : "float";
    binfile.Write_indexed_at_all(
        nullptr, 0, dtype_str,
        payload_off,
        static_cast<int>(window_a_),
        nullptr, 0,
        /*single_file_per_rank=*/false);
  }

  // ---- close (collective) ----
  binfile.Close(/*single_file_per_rank=*/false);

  // ---- increment counters (mirrors binary.cpp:305-313) ----
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
