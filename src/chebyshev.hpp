// Copyright (C) 2023 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#pragma once

#include "vector.hpp"
#include <algorithm>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <iostream>

using namespace dolfinx;

namespace dolfinx::acc
{

/// Conjugate gradient method
template <typename Vector>
class Chebyshev
{
  /// The value type
  using T = typename Vector::value_type;

public:
  Chebyshev(std::shared_ptr<const common::IndexMap> map, int bs, std::array<T, 2> eig_range)
      : _eig_range(eig_range), _diag_inv_computed(false)
  {
    _p = std::make_unique<Vector>(map, bs);
    _z = std::make_unique<Vector>(map, bs);
    _q = std::make_unique<Vector>(map, bs);
    _r = std::make_unique<Vector>(map, bs);
    _diag_inv = std::make_unique<Vector>(map, bs);
  }

  void set_max_iterations(int max_iter) { _max_iter = max_iter; }

  template <typename Operator>
  T residual(Operator& A, Vector& x, const Vector& b)
  {
    A(x, *_q);
    acc::axpy(*_r, T(-1), *_q, b);
    return acc::norm(*_r, dolfinx::la::Norm::l2);
  }

  // Solve Ax = b
  template <typename Operator>
  void solve(Operator& A, Vector& x, const Vector& b, bool jacobi, bool verbose)
  {
    spdlog::info("Chebyshev solve");
    // Using "fourth kind" Chebyshev from Phillips and Fischer
    // https://arxiv.org/pdf/2210.03179
    T lmax = _eig_range[1];

    if (!_diag_inv_computed)
    {
      common::Timer tinvdiag("% compute diaginv");
      if (jacobi)
        A.get_diag_inverse(*_diag_inv);
      else
        (*_diag_inv).set(1.0);
      _diag_inv_computed = true;
    }

    // r = b - Ax
    A(x, *_q);
    acc::axpy(*_r, T(-1.0), *_q, b);

    if (verbose)
    {
      T rnorm = acc::norm(*_r);
      spdlog::info("Iteration {}, UNPRECONDITIONED residual norm = {}", 0, rnorm);
    }

    // z = M^-1(r) * 4/(3*lmax)
    // Using M^-1 is Jacobi
    common::Timer tjac("% jacobi");
    acc::pointwise_mult(*_z, *_r, *_diag_inv);
    acc::scale(*_z, T(4.0 / (3.0 * lmax)));
    tjac.stop();
    tjac.flush();

    for (int i = 1; i < _max_iter + 1; i++)
    {
      // x += z
      acc::axpy(x, T(1.0), *_z, x);

      // r -= Az
      A(*_z, *_q);
      acc::axpy(*_r, T(-1.0), *_q, *_r);

      // z = z * (2i-1)/(2i+3) + M^-1(r) * (8i+4)/(2i+3)/lmax
      common::Timer tjac("% jacobi");
      acc::scale(*_z, T(2 * i - 1) / T(2 * i + 3));
      // Using M^-1 is Jacobi
      acc::pointwise_mult(*_q, *_r, *_diag_inv);
      tjac.stop();
      tjac.flush();

      acc::axpy(*_z, T(8 * i + 4) / T(2 * i + 3) / lmax, *_q, *_z);

      if (verbose)
      {
        T rnorm = acc::norm(*_r);
        spdlog::info("Iteration {}, UNPRECONDITIONED residual norm = {}", i, rnorm);
      }
    }
  }

private:
  /// Limit for the number of iterations the solver is allowed to do
  int _max_iter;

  /// Eigenvalues
  std::array<T, 2> _eig_range;

  /// Working vectors
  std::unique_ptr<Vector> _p;
  std::unique_ptr<Vector> _z;
  std::unique_ptr<Vector> _q;
  std::unique_ptr<Vector> _r;
  std::shared_ptr<Vector> _diag_inv;

  // Flag to indicate if diagonal inverse has been computed already
  bool _diag_inv_computed;
};
} // namespace dolfinx::acc
