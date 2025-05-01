#pragma once

#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <cstdint>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace
{

/// @brief Interpolate cells from Q1 to Q2
/// Launch with blocksize(Qmax, Qmax, Qmax) where Qmax=max(Q1, Q2)
/// If Q2 > Q1, prolongation (coarse->fine), if Q2 < Q1, restriction (fine->coarse)
/// @tparam T: scalar type
/// @tparam Q1: number of dofs in 1D (input space)
/// @tparam Q2: number of dofs in 1D (output space)
/// @param N: number of cells
/// @param cells: list of cell indices of length N
/// @param phi0_in: 1D interpolation table from Q1 to Q2, shape=[Q2, Q1]
/// @param dofmapQ1: list of shape N x (Q1^3)
/// @param dofmapQ2: list of shape N x (Q2^3)
/// @param valuesQ1: vector of values for Q1 (input)
/// @param valuesQ2: vector of values for Q2 (output)
/// @param mult: multiplicity of dofs in dofmap for Q2, needed for restriction.
template <typename T, int Q1, int Q2>
__global__ void interpolate_kernel(int N, const std::int32_t* cells, const T* phi0_in,
                                   const std::int32_t* dofmapQ1, const std::int32_t* dofmapQ2,
                                   const T* valuesQ1, T* valuesQ2, const T* mult)
{
  const int block_id = blockIdx.x;

  if (block_id > N)
    return;

  const int tx = threadIdx.z; // 1d dofs x direction
  const int ty = threadIdx.y; // 1d dofs y direction
  const int tz = threadIdx.x; // 1d dofs z direction

  constexpr int Qmax = (Q2 > Q1) ? Q2 : Q1;
  assert(Qmax == blockSize.x);

  __shared__ T scratch1[Qmax * Qmax * Qmax];
  __shared__ T scratch2[Qmax * Qmax * Qmax];

  // Copy 1D interpolation table
  __shared__ T phi0[Q2][Q1];

  // If Q2 > Q1 use phi, else phi.transpose
  if constexpr (Q2 > Q1)
  {
    if (tx < Q2 and ty < Q1)
      phi0[tx][ty] = phi0_in[tx * Q1 + ty];
  }
  else
  {
    if (tx < Q2 and ty < Q1)
      phi0[tx][ty] = phi0_in[ty * Q2 + tx];
  }

  auto ijk = [](auto i, auto j, auto k) { return i * Qmax * Qmax + j * Qmax + k; };

  if (tx < Q1 and ty < Q1 and tz < Q1)
  {
    // Fetch dofs from valuesQ1 using dofmapQ1
    int dof_thread_id = tx * Q1 * Q1 + ty * Q1 + tz;
    int cell_index = cells[block_id];
    int dof = dofmapQ1[cell_index * Q1 * Q1 * Q1 + dof_thread_id];
    if constexpr (Q2 < Q1)
      scratch2[ijk(tx, ty, tz)] = valuesQ1[dof] / mult[dof];
    else
      scratch2[ijk(tx, ty, tz)] = valuesQ1[dof];
  }
  else
    scratch2[ijk(tx, ty, tz)] = 0;

  __syncthreads();

  T xq = 0;
  for (int ix = 0; ix < Q1; ++ix)
    xq += phi0[tx][ix] * scratch2[ijk(ix, ty, tz)];

  scratch1[ijk(tx, ty, tz)] = xq;
  __syncthreads();

  xq = 0;
  for (int iy = 0; iy < Q1; ++iy)
    xq += phi0[ty][iy] * scratch1[ijk(tx, iy, tz)];

  scratch2[ijk(tx, ty, tz)] = xq;
  __syncthreads();

  xq = 0;
  for (int iz = 0; iz < Q1; ++iz)
    xq += phi0[tz][iz] * scratch2[ijk(tx, ty, iz)];

  // Put result into valuesQ2 using dofmapQ2
  if (tx < Q2 and ty < Q2 and tz < Q2)
  {
    int dof_thread_id = tx * Q2 * Q2 + ty * Q2 + tz;
    int cell_index = cells[block_id];
    int dof = dofmapQ2[cell_index * Q2 * Q2 * Q2 + dof_thread_id];
    if constexpr (Q2 < Q1)
      atomicAdd(&valuesQ2[dof], xq);
    else
      valuesQ2[dof] = xq;
  }
}

} // namespace

/// @brief Matrix-free Interpolator between two P-levels
///
template <typename T>
class Interpolator
{
public:
  /// @brief Set up interpolator from coarse space Q1 to fine space Q2
  /// @param Q1_element - 3D hexahedral element of coarse space (Q1)
  /// @param Q2_element - 3D hexahedral element of fine space (Q2)
  /// @param Q1_dofmap - dofmap Q1 (on device)
  /// @param Q2_dofmap - dofmap Q2 (on device)
  /// @param l_cells - local cells, to interpolate immediately
  /// @param b_cells - boundary cells, to interpolate after vector update
  Interpolator(const basix::FiniteElement<T>& Q1_element, const basix::FiniteElement<T>& Q2_element,
               std::span<const std::int32_t> Q1_dofmap, std::span<const std::int32_t> Q2_dofmap,
               std::span<const std::int32_t> l_cells, std::span<const std::int32_t> b_cells)
      : Q1_dofmap(Q1_dofmap), Q2_dofmap(Q2_dofmap), local_cells(l_cells), boundary_cells(b_cells)
  {
    assert(Q1_element.has_tensor_product_factorisation());
    assert(Q2_element.has_tensor_product_factorisation());

    auto element_Q1_1D = Q1_element.get_tensor_product_representation().at(0).at(0);
    auto element_Q2_1D = Q2_element.get_tensor_product_representation().at(0).at(0);

    nQ1 = element_Q1_1D.dim();
    nQ2 = element_Q2_1D.dim();
    if (nQ1 >= nQ2)
      throw std::runtime_error("Cannot create interpolator");

    spdlog::info("Creating interpolator <{}, {}>", nQ1, nQ2);

    // Checks on dofmap shapes and sizes
    assert(Q1_dofmap.size() % (nQ1 * nQ1 * nQ1) == 0);
    assert(Q2_dofmap.size() % (nQ2 * nQ2 * nQ2) == 0);
    assert(Q2_dofmap.size() / (nQ2 * nQ2 * nQ2) == Q1_dofmap.size() / (nQ1 * nQ1 * nQ1));

    // Get local interpolation matrix for 1D elements (to use in sum factorisation)
    auto [mat, shape] = basix::compute_interpolation_operator(element_Q1_1D, element_Q2_1D);

    // Copy mat to device as phi0
    assert(mat.size() == nQ1 * nQ2);
    phi0.resize(nQ1 * nQ2);
    thrust::copy(mat.begin(), mat.end(), phi0.begin());

    // Compute dofmap multiplicity (histogram) of fine space (Q2)
    // required when restricting fine->coarse so that restriction will be
    // transpose of prolongation, when applied across multiple cells.
    thrust::device_vector<std::int32_t> Q2sorted(Q2_dofmap.size());
    thrust::copy(thrust::device_pointer_cast(Q2_dofmap.data()),
                 thrust::device_pointer_cast(Q2_dofmap.data()) + Q2_dofmap.size(),
                 Q2sorted.begin());
    thrust::sort(Q2sorted.begin(), Q2sorted.end());
    int num_bins = Q2sorted.back() + 1;

    // Resize histogram storage
    Q2mult.resize(num_bins);
    thrust::counting_iterator<int> search_begin(0);
    thrust::upper_bound(Q2sorted.begin(), Q2sorted.end(), search_begin, search_begin + num_bins,
                        Q2mult.begin());
    thrust::adjacent_difference(Q2mult.begin(), Q2mult.end(), Q2mult.begin());
  }

  /// @brief Interpolate from input_values to output_values (both on device)
  /// @tparam Q1 size of input in 1D
  /// @tparam Q2 size of output in 1D
  /// If Q2 > Q1, prolongation (coarse->fine), if Q2 < Q1, restriction (fine->coarse).
  /// @param Q1_vector DeviceVector containing input data
  /// @param in_dofmap on-device dofmap for input vector Q1
  /// @param Q2_vector DeviceVector for output data
  /// @param out_dofmap on-device dofmap for output vector Q2
  template <int Q1, int Q2, typename Vector>
  void impl_interpolate(Vector& Q1_vector, std::span<const std::int32_t> in_dofmap,
                        Vector& Q2_vector, std::span<const std::int32_t> out_dofmap)
  {
    dolfinx::common::Timer tt("Interpolate Kernel: " + std::to_string(Q1 - 1) + "-"
                              + std::to_string(Q2 - 1));

    // Input (Q1) vector is also changed by MPI vector update
    T* Q1_values = Q1_vector.mutable_array().data();
    T* Q2_values = Q2_vector.mutable_array().data();

    assert(in_dofmap.size() % (Q1 * Q1 * Q1) == 0);
    assert(out_dofmap.size() % (Q2 * Q2 * Q2) == 0);

    thrust::device_vector<std::int32_t> cell_list_d(local_cells.begin(), local_cells.end());

    int ncells = local_cells.size();
    constexpr int Qmax = (Q2 > Q1) ? Q2 : Q1;
    dim3 block_size(Qmax, Qmax, Qmax);
    dim3 grid_size(ncells);

    // Start vector update of input vector
    Q1_vector.scatter_fwd_begin();

    spdlog::info("From {} to {} on {} cells", Q1, Q2, ncells);
    spdlog::info("Input dofmap size = {}, output dofmap size = {}", in_dofmap.size(),
                 out_dofmap.size());

    // Only need to zero output vector if interpolating from fine to coarse
    if constexpr (Q2 < Q1)
      Q2_vector.set(0.0);

    // Interpolate from Q1 to Q2
    interpolate_kernel<T, Q1, Q2><<<grid_size, block_size, 0, 0>>>(
        ncells, thrust::raw_pointer_cast(cell_list_d.data()), thrust::raw_pointer_cast(phi0.data()),
        in_dofmap.data(), out_dofmap.data(), Q1_values, Q2_values,
        thrust::raw_pointer_cast(Q2mult.data()));

    check_device_last_error();

    // Wait for update of input vector to complete
    Q1_vector.scatter_fwd_end();

    cell_list_d.resize(boundary_cells.size());
    thrust::copy(boundary_cells.begin(), boundary_cells.end(), cell_list_d.begin());
    ncells = boundary_cells.size();
    if (ncells > 0)
    {
      spdlog::info("From {} dofs/cell to {} on {} (boundary) cells", Q1, Q2, ncells);

      interpolate_kernel<T, Q1, Q2><<<grid_size, block_size, 0, 0>>>(
          ncells, thrust::raw_pointer_cast(cell_list_d.data()),
          thrust::raw_pointer_cast(phi0.data()), in_dofmap.data(), out_dofmap.data(), Q1_values,
          Q2_values, thrust::raw_pointer_cast(Q2mult.data()));
    }
    device_synchronize();
    check_device_last_error();

    spdlog::debug("Done mat-free interpolation");
  }

  template <typename Vector>
  void interpolate(Vector& Q1_vector, Vector& Q2_vector)
  {
    // Implement prolongation from Q3 to Q6 and from Q1 to Q3
    if (nQ1 == 4 and nQ2 == 7)
      impl_interpolate<4, 7>(Q1_vector, Q1_dofmap, Q2_vector, Q2_dofmap);
    else if (nQ1 == 2 and nQ2 == 4)
      impl_interpolate<2, 4>(Q1_vector, Q1_dofmap, Q2_vector, Q2_dofmap);
    else
      throw std::runtime_error("Missing implementation");
  }

  template <typename Vector>
  void reverse_interpolate(Vector& Q2_vector, Vector& Q1_vector)
  {
    // Implement restriction from Q6 to Q3 and from Q3 to Q1
    if (nQ1 == 4 and nQ2 == 7)
      impl_interpolate<7, 4>(Q2_vector, Q2_dofmap, Q1_vector, Q1_dofmap);
    else if (nQ1 == 2 and nQ2 == 4)
      impl_interpolate<4, 2>(Q2_vector, Q2_dofmap, Q1_vector, Q1_dofmap);
    else
      throw std::runtime_error("Missing implementation");
  }

private:
  // Size of 1D elements
  int nQ1, nQ2;

  // Interpolation matrix from Q1->Q2 in 1D
  thrust::device_vector<T> phi0;

  // Multiplicity of dofs in Q2 (fine space)
  thrust::device_vector<T> Q2mult;

  // Dofmaps (on device)
  std::span<const std::int32_t> Q1_dofmap;
  std::span<const std::int32_t> Q2_dofmap;

  // List of local cells, which can be updated before a Vector update
  std::span<const std::int32_t> local_cells;

  // List of cells which are in the "boundary region" which need to wait for a Vector update
  // before interpolation (on device)
  std::span<const std::int32_t> boundary_cells;
};
