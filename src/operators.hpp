// Copyright (C) 2023 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#pragma once

#include <dolfinx.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/dolfinx_fem.h>
#include <dolfinx/fem/petsc.h>
#include <petscmat.h>

/**
 * @brief PETScOperator is a class that provides an operator to perform
 * matrix-vector multiplication using PETSc library. It is used to efficiently
 * compute matrix-vector products in finite element computations. It takes a
 * finite element form and a vector of Dirichlet boundary conditions as input.
 * The matrix-vector multiplication operation is implemented in the operator()
 * method of this class. This class template can be instantiated for a specific
 * data type (currently the type should match PetscScalar).
 */
template <typename T>
class PETScOperator
{
public:
  /**
   * @brief Constructs a PETScOperator object with the given form and Dirichlet
   * boundary conditions. It assembles the matrix and converts it to a HIP/CUDA
   * matrix.
   *
   * @param a         A shared pointer to the finite element form.
   * @param bcs       A vector of shared pointers to Dirichlet BCs.
   *
   * @throw std::runtime_error if the rank of the form is not 2.
   */
  PETScOperator(std::shared_ptr<fem::Form<T, T>> a,
                const std::vector<std::shared_ptr<const fem::DirichletBC<T, T>>>& bcs)
  {
    dolfinx::common::Timer t0("~setup phase PETScOperator");
    static_assert(std::is_same<T, PetscScalar>(), "Type mismatch");

    if (a->rank() != 2)
      throw std::runtime_error("Form should have rank be 2.");

    la::SparsityPattern pattern = fem::create_sparsity_pattern(*a);
    pattern.finalize();

    spdlog::info("Create matrix... {}x{}", pattern.index_map(0)->size_global(),
                 pattern.index_map(1)->size_global());
#ifdef USE_HIP
    _host_mat = la::petsc::create_matrix(a->mesh()->comm(), pattern, "aijhipsparse");
#elif USE_CUDA
    _host_mat = la::petsc::create_matrix(a->mesh()->comm(), pattern, "aijcusparse");
#endif

    std::vector<std::reference_wrapper<const fem::DirichletBC<T, T>>> bcs_ref;
    for (auto bc : bcs)
      bcs_ref.push_back(*bc);

    spdlog::info("Zero matrix...");
    MatZeroEntries(_host_mat);
    auto set_fn = la::petsc::Matrix::set_block_fn(_host_mat, ADD_VALUES);
    spdlog::info("Assemble matrix...");
    fem::assemble_matrix(set_fn, *a, bcs_ref);

    spdlog::info("Assemble matrix 2...");
    auto V = a->function_spaces()[0];
    MatAssemblyBegin(_host_mat, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(_host_mat, MAT_FLUSH_ASSEMBLY);
    auto insert = la::petsc::Matrix::set_fn(_host_mat, INSERT_VALUES);
    fem::set_diagonal<T>(insert, *V, bcs_ref);
    MatAssemblyBegin(_host_mat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(_host_mat, MAT_FINAL_ASSEMBLY);

    PetscReal norm;
    MatNorm(_host_mat, NORM_FROBENIUS, &norm);
    spdlog::info("Mat norm = {}", norm);

    spdlog::info("Convert matrix...");
    // Create HIP/CUDA matrix
    dolfinx::common::Timer t1("~Convert matrix to MATAIJ[HIP/CU]SPARSE");
    int ierr = MatConvert(_host_mat, MATSAME, MAT_INITIAL_MATRIX, &_device_mat);
    spdlog::info("ierr={}", ierr);
    t1.stop();

    // Get communicator from mesh
    _comm = V->mesh()->comm();

    _map = std::make_shared<const common::IndexMap>(pattern.column_index_map());
    const PetscInt local_size = _map->size_local();
    const PetscInt global_size = _map->size_global();
    spdlog::info("Create vecs... {}/{}", local_size, global_size);

#ifdef USE_HIP
    ierr = VecCreateMPIHIPWithArray(_comm, PetscInt(1), local_size, global_size, NULL, &_x_petsc);
    spdlog::info("ierr={}", ierr);
    ierr = VecCreateMPIHIPWithArray(_comm, PetscInt(1), local_size, global_size, NULL, &_y_petsc);
    spdlog::info("ierr={}", ierr);
#elif USE_CUDA
    ierr = VecCreateMPICUDAWithArray(_comm, PetscInt(1), local_size, global_size, NULL, &_x_petsc);
    spdlog::info("ierr={}", ierr);
    ierr = VecCreateMPICUDAWithArray(_comm, PetscInt(1), local_size, global_size, NULL, &_y_petsc);
    spdlog::info("ierr={}", ierr);
#endif
  }

  PETScOperator(const fem::FunctionSpace<T>& V0, const fem::FunctionSpace<T>& V1)
  {
    _comm = V0.mesh()->comm();
    assert(V0.mesh());
    auto mesh = V0.mesh();
    assert(V1.mesh());
    assert(mesh == V1.mesh());

    std::shared_ptr<const fem::DofMap> dofmap0 = V0.dofmap();
    assert(dofmap0);
    std::shared_ptr<const fem::DofMap> dofmap1 = V1.dofmap();
    assert(dofmap1);

    // Create and build  sparsity pattern
    assert(dofmap0->index_map);
    assert(dofmap1->index_map);

    la::SparsityPattern pattern(_comm, {dofmap1->index_map, dofmap0->index_map},
                                {dofmap1->index_map_bs(), dofmap0->index_map_bs()});

    int tdim = mesh->topology()->dim();
    auto map = mesh->topology()->index_map(tdim);
    assert(map);
    std::vector<std::int32_t> c(map->size_local(), 0);
    std::iota(c.begin(), c.end(), 0);
    fem::sparsitybuild::cells(pattern, {c, c}, {*dofmap1, *dofmap0});
    pattern.finalize();

    // Build operator
    _host_mat = dolfinx::la::petsc::create_matrix(_comm, pattern);
    MatSetOption(_host_mat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
    auto set_fn = la::petsc::Matrix::set_block_fn(_host_mat, INSERT_VALUES);
    fem::interpolation_matrix<PetscScalar>(V0, V1, set_fn);

    // Create HIP/CUDA matrix
#ifdef USE_HIP
    MatConvert(_host_mat, MATAIJHIPSPARSE, MAT_INITIAL_MATRIX, &_device_mat);
#elif USE_CUDA
    MatConvert(_host_mat, MATAIJCUSPARSE, MAT_INITIAL_MATRIX, &_device_mat);
#endif

    _map = pattern.index_map(1);
    const PetscInt local_size = _map->size_local();
    const PetscInt global_size = _map->size_global();
#ifdef USE_HIP
    VecCreateMPIHIPWithArray(_comm, PetscInt(1), local_size, global_size, NULL, &_x_petsc);
    VecCreateMPIHIPWithArray(_comm, PetscInt(1), local_size, global_size, NULL, &_y_petsc);
#elif USE_CUDA
    VecCreateMPICUDAWithArray(_comm, PetscInt(1), local_size, global_size, NULL, &_x_petsc);
    VecCreateMPICUDAWithArray(_comm, PetscInt(1), local_size, global_size, NULL, &_y_petsc);
#endif
  }

  /**
   * @brief Destructor that destroys the PETSc objects used in the class.
   */
  ~PETScOperator()
  {
    VecDestroy(&_x_petsc);
    VecDestroy(&_y_petsc);
    MatDestroy(&_device_mat);
  }

  std::shared_ptr<const common::IndexMap> index_map() { return _map; };

  /**
   * @brief Return device PETSc Mat pointer.
   */
  Mat device_matrix() const { return _device_mat; }

  /**
   * @brief Return host PETSc Mat pointer.
   */
  Mat host_matrix() const { return _host_mat; }

  /**
   * @brief The matrix-vector multiplication operator, which multiplies the
   * matrix with the input vector and stores the result in the output vector.
   *
   * @tparam Vector  The type of the input and output vector.
   *
   * @param x        The input vector.
   * @param y        The output vector.
   */
  template <typename Vector>
  void operator()(const Vector& x, Vector& y, bool transpose = false)
  {
    spdlog::info("Hip/CuPlaceArray");

#ifdef USE_HIP
    VecHIPPlaceArray(_x_petsc, x.array().data());
    VecHIPPlaceArray(_y_petsc, y.array().data());
#elif USE_CUDA
    VecCUDAPlaceArray(_x_petsc, x.array().data());
    VecCUDAPlaceArray(_y_petsc, y.array().data());
#endif

    int nx, ny;
    MatGetLocalSize(_device_mat, &nx, &ny);
    spdlog::info("Mat shape = {}x{}", nx, ny);
    VecGetLocalSize(_x_petsc, &nx);
    spdlog::info("Vec size x = {}", nx);
    VecGetLocalSize(_y_petsc, &ny);
    spdlog::info("Vec size y = {}", ny);

    spdlog::info("MatMult");
    if (transpose)
      // y = A^T x
      MatMultTranspose(_device_mat, _x_petsc, _y_petsc);
    else
      // y = A x
      MatMult(_device_mat, _x_petsc, _y_petsc);

    spdlog::info("Hip/CuResetArray");
#ifdef USE_HIP
    VecHIPResetArray(_y_petsc);
    VecHIPResetArray(_x_petsc);
#elif USE_CUDA
    VecCUDAResetArray(_y_petsc);
    VecCUDAResetArray(_x_petsc);
#endif
  }

private:
  Vec _x_petsc = nullptr; // PETSc vector for input
  Vec _y_petsc = nullptr; // PETSc vector for output
  Mat _host_mat;          // Host PETSc matrix
  Mat _device_mat;        // HIP/CUDA matrix
  MPI_Comm _comm;         // MPI communicator
  std::shared_ptr<const common::IndexMap> _map;
};
