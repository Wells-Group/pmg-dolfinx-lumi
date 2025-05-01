#include "../../src/amg.hpp"
#include "../../src/cg.hpp"
#include "../../src/chebyshev.hpp"
#include "../../src/csr.hpp"
#include "../../src/interpolate.hpp"
#include "../../src/laplacian.hpp"
#include "../../src/mesh.hpp"
#include "../../src/operators.hpp"
#include "../../src/pmg.hpp"
#include "../../src/vector.hpp"
#include "poisson.h"

#include <thrust/device_vector.h>

#include <array>
#include <basix/e-lagrange.h>
#include <boost/program_options.hpp>
#include <dolfinx.h>
#include <dolfinx/fem/dolfinx_fem.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/generation.h>
#include <iostream>
#include <memory>
#include <mpi.h>

using namespace dolfinx;
using T = double;

#ifdef USE_HIP
using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;
#else
using DeviceVector = dolfinx::acc::Vector<T, acc::Device::CUDA>;
#endif

namespace po = boost::program_options;

template <typename FineOperator>
void solve(std::shared_ptr<mesh::Mesh<double>> mesh, bool use_amg,
           bool output_to_file, int amg_its, int cheb_iters) {
  if constexpr (std::is_same_v<FineOperator, acc::MatFreeLaplacian<T>>) {
    spdlog::info("------- MatFree -------");
  } else {
    spdlog::info("------- CSR -------");
  }

  int rank = dolfinx::MPI::rank(mesh->comm());
  if (rank == 0)
    spdlog::set_level(spdlog::level::info);

  int size = dolfinx::MPI::size(mesh->comm());
  std::vector<int> order = {1, 3, 6};
  std::vector form_a = {form_poisson_a1, form_poisson_a3, form_poisson_a6};
  std::vector form_L = {form_poisson_L1, form_poisson_L3, form_poisson_L6};

  auto topology = mesh->topology_mutable();
  int tdim = topology->dim();
  int fdim = tdim - 1;
  spdlog::debug("Create facets");
  topology->create_connectivity(fdim, tdim);

  std::vector<std::shared_ptr<fem::FunctionSpace<T>>> V(form_a.size());
  std::vector<std::shared_ptr<fem::Form<T, T>>> a(V.size());
  std::vector<std::shared_ptr<fem::Form<T, T>>> L(V.size());
  std::vector<std::shared_ptr<const fem::DirichletBC<T, T>>> bcs(V.size());

  // List of LHS operators (CSR or MatrixFree), one for each level.
  std::vector<std::shared_ptr<FineOperator>> operators(V.size());

  std::vector<std::shared_ptr<const common::IndexMap>> maps(V.size());

  auto facets = dolfinx::mesh::exterior_facet_indices(*topology);

  std::vector<std::size_t> ndofs(V.size());

  // Prepare and set Constants for the bilinear form
  auto kappa = std::make_shared<fem::Constant<T>>(2.0);
  for (std::size_t i = 0; i < form_a.size(); i++) {
    spdlog::info("Creating FunctionSpace at order {}", order[i]);
    auto element = basix::create_tp_element<T>(
        basix::element::family::P, basix::cell::type::hexahedron, order[i],
        basix::element::lagrange_variant::gll_warped,
        basix::element::dpc_variant::unset, false);

    V[i] = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
        mesh, std::make_shared<const fem::FiniteElement<T>>(element)));
    ndofs[i] = V[i]->dofmap()->index_map->size_global();
    a[i] = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_a[i], {V[i], V[i]}, {}, {{"c0", kappa}}, {}, {}));
  }

  spdlog::info("Compute boundary cells");
  // Compute local and boundary cells (needed for MatFreeLaplacian)
  // FIXME: Generalise to more levels
  auto [lcells, bcells] = compute_boundary_cells(V.back());

  // assemble RHS for each level
  for (std::size_t i = 0; i < V.size(); i++) {
    spdlog::info("Build RHS for order {}", order[i]);

    // auto f = std::make_shared<fem::Function<T>>(V[i]);
    // f->interpolate(
    //     [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
    //     {
    //       std::vector<T> out;
    //       for (std::size_t p = 0; p < x.extent(1); ++p)
    //       {
    //         auto dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
    //         auto dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
    //         out.push_back(1000 * std::exp(-(dx + dy) / 0.02));
    //       }

    //       return {out, {out.size()}};
    //     });

    L[i] = std::make_shared<fem::Form<T, T>>(
        fem::create_form<T>(*form_L[i], {V[i]}, {}, {{"c0", kappa}}, {}, {}));

    auto dofmap = V[i]->dofmap();
    auto bdofs = fem::locate_dofs_topological(*topology, *dofmap, fdim, facets);
    bcs[i] = std::make_shared<const fem::DirichletBC<T, T>>(0.0, bdofs, V[i]);
  }

  // If not using AMG coarse_solver will be a nullptr, and operators[0] will be
  // applied
  std::shared_ptr<CoarseSolverType<T>> coarse_solver;
  if (use_amg)
    coarse_solver =
        std::make_shared<CoarseSolverType<T>>(a[0], bcs[0], amg_its);

  // RHS
  std::size_t ncells = mesh->topology()->index_map(3)->size_global();
  if (rank == 0) {
    std::cout << "-----------------------------------\n";
    std::cout << "Number of ranks : " << size << "\n";
    std::cout << "Number of cells-global : " << ncells << "\n";
    std::cout << "Number of dofs-global : " << ndofs.back() << "\n";
    std::cout << "Number of cells-rank : " << ncells / size << "\n";
    std::cout << "Number of dofs-rank : " << ndofs.back() / size << "\n";
    std::cout << "-----------------------------------\n";
    std::cout << "Hierarchy: " << std::endl;
    for (std::size_t i = 0; i < ndofs.size(); i++) {
      std::cout << "Level " << i << ": " << ndofs[i] << "\n";
    }
    std::cout << "-----------------------------------\n";
    std::cout << std::flush;
  }

  // Data for required quantities for MatFreeLaplacian:

  // Dofmaps for each level
  std::vector<thrust::device_vector<std::int32_t>> dofmapV(V.size());
  std::vector<std::span<std::int32_t>> device_dofmaps;

  // Geometry
  thrust::device_vector<T> geomx_device;
  std::span<T> geom_x;
  thrust::device_vector<std::int32_t> geomx_dofmap_device;
  std::span<std::int32_t> geom_x_dofmap;
  std::vector<thrust::device_vector<T>> geometry_dphi_d(V.size());
  std::vector<std::span<const T>> geometry_dphi_d_span;

  // BCs
  std::vector<thrust::device_vector<std::int8_t>> bc_marker_d(V.size());
  std::vector<std::span<const std::int8_t>> bc_marker_d_span;

  // Copy bc_dofs to device (list of all dofs, with BCs marked with 0)
  for (std::size_t i = 0; i < V.size(); ++i) {
    spdlog::debug("Copy BCs[{}] to device", i);
    auto [dofs, pos] = bcs[i]->dof_indices();
    std::vector<std::int8_t> active_bc_dofs(
        V[i]->dofmap()->index_map->size_local() +
            V[i]->dofmap()->index_map->num_ghosts(),
        0);
    for (std::int32_t index : dofs)
      active_bc_dofs[index] = 1;
    bc_marker_d[i] = thrust::device_vector<std::int8_t>(active_bc_dofs.begin(),
                                                        active_bc_dofs.end());
    bc_marker_d_span.push_back(
        std::span(thrust::raw_pointer_cast(bc_marker_d[i].data()),
                  bc_marker_d[i].size()));
  }

  device_synchronize();

  // Copy constants to device (all same, one per cell, scalar)
  thrust::device_vector<T> constants(
      mesh->topology()->index_map(tdim)->size_local() +
          mesh->topology()->index_map(tdim)->num_ghosts(),
      kappa->value[0]);
  std::span<T> device_constants(thrust::raw_pointer_cast(constants.data()),
                                constants.size());

  if constexpr (std::is_same_v<FineOperator, acc::MatFreeLaplacian<T>>) {
    // Copy dofmaps to device (only for MatFreeLaplacian)

    for (std::size_t i = 0; i < V.size(); ++i) {
      dofmapV[i].resize(V[i]->dofmap()->map().size());
      spdlog::debug("Copy dofmap (V{}) : {}", i, dofmapV[i].size());
      thrust::copy(V[i]->dofmap()->map().data_handle(),
                   V[i]->dofmap()->map().data_handle() +
                       V[i]->dofmap()->map().size(),
                   dofmapV[i].begin());
      device_dofmaps.push_back(std::span<std::int32_t>(
          thrust::raw_pointer_cast(dofmapV[i].data()), dofmapV[i].size()));
    }

    device_synchronize();

    // Copy geometry to device (same for all kernels)
    spdlog::debug("Copy geometry data to device");
    geomx_device.resize(mesh->geometry().x().size());
    spdlog::info("Copy geometry to device :{}", geomx_device.size());
    thrust::copy(mesh->geometry().x().begin(), mesh->geometry().x().end(),
                 geomx_device.begin());
    geom_x = std::span<T>(thrust::raw_pointer_cast(geomx_device.data()),
                          geomx_device.size());

    geomx_dofmap_device.resize(mesh->geometry().dofmap().size());
    thrust::copy(mesh->geometry().dofmap().data_handle(),
                 mesh->geometry().dofmap().data_handle() +
                     mesh->geometry().dofmap().size(),
                 geomx_dofmap_device.begin());
    geom_x_dofmap = std::span<std::int32_t>(
        thrust::raw_pointer_cast(geomx_dofmap_device.data()),
        geomx_dofmap_device.size());

    device_synchronize();
  }

  std::vector<std::shared_ptr<DeviceVector>> bs(V.size());
  for (std::size_t i = 0; i < V.size(); i++) {
    std::shared_ptr<fem::Form<T, T>> a_i = a[i];
    std::vector<std::shared_ptr<const fem::DirichletBC<T, T>>> bc_i = {bcs[i]};

    if constexpr (std::is_same_v<FineOperator, acc::MatFreeLaplacian<T>>) {
      maps[i] = V[i]->dofmap()->index_map;

      spdlog::info("Create operator on V[{}]", i);
      basix::quadrature::type quad_type = basix::quadrature::type::gll;
      operators[i] = std::make_shared<acc::MatFreeLaplacian<T>>(
          order[i], 1, device_constants, device_dofmaps[i], geom_x,
          geom_x_dofmap, mesh->geometry().cmap(), lcells, bcells,
          bc_marker_d_span[i], quad_type);

      device_synchronize();
    } else {
      operators[i] = std::make_shared<acc::MatrixOperator<T>>(a_i, bc_i);
      maps[i] = operators[i]->column_index_map();
    }

    spdlog::debug("Assembling vector on CPU");
    la::Vector<T> b(maps[i], 1);
    b.set(T(0.0));
    fem::assemble_vector(b.mutable_array(), *L[i]);

    spdlog::debug("Apply lifting");
    // Commenting this out because it is very slow for P6, and it does nothing
    // because our BC is set to zero anyway.
    //    fem::apply_lifting<T, T>(b.mutable_array(), {*a[i]}, {{*bcs[i]}}, {},
    //    T(1));
    b.scatter_rev(std::plus<T>());
    bcs[i]->set(b.mutable_array(), std::nullopt);

    spdlog::info("b[{}].norm = {}", i, dolfinx::la::norm(b));

    bs[i] = std::make_shared<DeviceVector>(maps[i], 1);
    bs[i]->copy_from_host(b);
  }

  // Create Matrix-Free Interpolators
  spdlog::debug("Creating Interpolation Operators");
  std::vector<std::shared_ptr<Interpolator<T>>> matfree_interpolators(V.size() -
                                                                      1);

  for (int i = 0; i < V.size() - 1; ++i) {
    matfree_interpolators[i] = std::make_shared<Interpolator<T>>(
        V[i]->element()->basix_element(), V[i + 1]->element()->basix_element(),
        device_dofmaps[i], device_dofmaps[i + 1], lcells, bcells);
  }

  spdlog::info("Create Chebyshev smoothers");

  // Create chebyshev smoother for each level
  std::vector<std::shared_ptr<acc::Chebyshev<DeviceVector>>> smoothers(
      V.size());
  for (std::size_t i = 0; i < V.size(); i++) {
    dolfinx::acc::CGSolver<DeviceVector> cg(maps[i], 1);
    cg.set_max_iterations(20);
    cg.set_tolerance(1e-6);
    cg.store_coefficients(true);

    DeviceVector x(maps[i], 1);

    spdlog::debug("map local size = {}, ghost size = {}", maps[i]->size_local(),
                  maps[i]->num_ghosts());

    x.set(T{0.0});
    DeviceVector y(maps[i], 1);
    y.set(T{1.0});

    [[maybe_unused]] int its = cg.solve(*operators[i], x, y, true, false);
    spdlog::info("CG iterations: {}", its);

    std::vector<T> eign = cg.compute_eigenvalues();
    std::sort(eign.begin(), eign.end());
    spdlog::info("Eigenvalues level {}: {} - {}", i, eign.front(), eign.back());
    std::array<T, 2> eig_range = {0.1 * eign.back(), 1.1 * eign.back()};
    smoothers[i] =
        std::make_shared<acc::Chebyshev<DeviceVector>>(maps[i], 1, eig_range);
    smoothers[i]->set_max_iterations(cheb_iters);
  }

  using SolverType = acc::Chebyshev<DeviceVector>;
  using PMG =
      acc::MultigridPreconditioner<DeviceVector, FineOperator, SolverType,
                                   CoarseSolverType<T>, Interpolator<T>>;

  spdlog::info("Create PMG");
  PMG pmg(maps, 1, bc_marker_d_span[0]);
  pmg.set_solvers(smoothers);
  pmg.set_operators(operators);
  spdlog::info("Set Coarse Solver");
  pmg.set_coarse_solver(coarse_solver);

  // Sets matrix-free kernels to do interpolation
  pmg.set_interpolators(matfree_interpolators);

  // Create solution vector
  spdlog::info("Create x");
  DeviceVector x(maps.back(), 1);
  x.set(T{0.0});

  int niter = 60;
  for (int i = 0; i < niter; i++) {
    pmg.apply(*bs.back(), x, rank == 0);
  }

#ifdef HAS_ADIOS2
  if (output_to_file) {
    auto u = std::make_shared<fem::Function<T>>(V.back());
    auto xv = x.thrust_vector();
    thrust::copy(xv.begin(), xv.end(), u->x()->mutable_array().begin());

    auto element = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::hexahedron, order.back(),
        basix::element::lagrange_variant::gll_warped,
        basix::element::dpc_variant::unset, false);
    auto VL = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
        mesh, std::make_shared<const fem::FiniteElement<T>>(element)));
    auto uL = std::make_shared<fem::Function<T>>(VL);
    uL->interpolate(*u);

    dolfinx::io::VTXWriter<T> write_adios(mesh->comm(), "solution.bp", {uL});
    write_adios.write(0.0);
  }
#endif
}

int main(int argc, char *argv[]) {
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "ndofs", po::value<std::size_t>()->default_value(50000),
      "number of dofs per rank")("amg_its",
                                 po::value<int>()->default_value(10))(
      "cheb_iters", po::value<int>()->default_value(2))(
      "amg", po::bool_switch()->default_value(false))(
      "output", po::bool_switch()->default_value(false))(
      "mesh", po::value<std::string>()->default_value(""));

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(desc)
                .allow_unregistered()
                .run(),
            vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }
  const std::size_t ndofs = vm["ndofs"].as<std::size_t>();
  bool use_amg = vm["amg"].as<bool>();
  const int amg_its = vm["amg_its"].as<int>();
  const int cheb_iters = vm["cheb_iters"].as<int>();
  bool output_to_file = vm["output"].as<bool>();
  const std::string mesh_name = vm["mesh"].as<std::string>();

  init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int size = 0;
    MPI_Comm_size(comm, &size);

    int max_order = 6; // FIXME

    double nx_approx = (std::pow(ndofs * size, 1.0 / 3.0) - 1) / max_order;
    std::int64_t n0 = static_cast<int>(nx_approx);
    std::array<std::int64_t, 3> nx = {n0, n0, n0};

    // Try to improve fit to ndofs +/- 5 in each direction
    if (n0 > 5) {
      std::int64_t best_misfit =
          (n0 * max_order + 1) * (n0 * max_order + 1) * (n0 * max_order + 1) -
          ndofs * size;
      best_misfit = std::abs(best_misfit);
      for (std::int64_t nx0 = n0 - 5; nx0 < n0 + 6; ++nx0)
        for (std::int64_t ny0 = n0 - 5; ny0 < n0 + 6; ++ny0)
          for (std::int64_t nz0 = n0 - 5; nz0 < n0 + 6; ++nz0) {
            std::int64_t misfit = (nx0 * max_order + 1) *
                                      (ny0 * max_order + 1) *
                                      (nz0 * max_order + 1) -
                                  ndofs * size;
            if (std::abs(misfit) < best_misfit) {
              best_misfit = std::abs(misfit);
              nx = {nx0, ny0, nz0};
            }
          }
    }

    spdlog::info("Creating mesh of size: {}x{}x{}", nx[0], nx[1], nx[2]);

    // Create mesh
    std::shared_ptr<mesh::Mesh<T>> mesh;
    {
      std::shared_ptr<mesh::Mesh<T>> base_mesh;
      if (mesh_name.size() == 0) {
        base_mesh = std::make_shared<mesh::Mesh<T>>(mesh::create_box<T>(
            comm, {{{0, 0, 0}, {1, 1, 1}}}, {nx[0], nx[1], nx[2]},
            mesh::CellType::hexahedron));
      } else {
        io::XDMFFile xdmf2(comm, mesh_name, "r");
        fem::CoordinateElement<T> coord_element0(mesh::CellType::hexahedron, 1);
        base_mesh = std::make_shared<mesh::Mesh<T>>(
            xdmf2.read_mesh(coord_element0, mesh::GhostMode::none, "Grid"));
        xdmf2.close();
      }

      // First order coordinate element
      auto element_1 =
          std::make_shared<basix::FiniteElement<T>>(basix::create_tp_element<T>(
              basix::element::family::P, basix::cell::type::hexahedron, 1,
              basix::element::lagrange_variant::gll_warped,
              basix::element::dpc_variant::unset, false));
      dolfinx::fem::CoordinateElement<T> coord_element(element_1);

      mesh = std::make_shared<mesh::Mesh<T>>(
          ghost_layer_mesh(*base_mesh, coord_element));
    }

    // Solve using Matrix-free operators
    solve<acc::MatFreeLaplacian<T>>(mesh, use_amg, output_to_file, amg_its,
                                    cheb_iters);

    // Solve using CSR matrices
    // solve<acc::MatrixOperator<T>>(mesh, use_amg, output_to_file);

    // Display timings
    dolfinx::list_timings(MPI_COMM_WORLD);
  }

  PetscFinalize();
  return 0;
}
