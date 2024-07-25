# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for jax_cfd.pressure."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax_cfd_nlc.base import array_utils
from jax_cfd_nlc.base import boundaries
from jax_cfd_nlc.base import fast_diagonalization
from jax_cfd_nlc.base import finite_differences as fd
from jax_cfd_nlc.base import grids
from jax_cfd_nlc.base import pressure
from jax_cfd_nlc.base import test_util
import numpy as np

USE_FLOAT64 = True

solve_cg = functools.partial(pressure.solve_cg, atol=1e-6, maxiter=10**5)


class PressureTest(test_util.TestCase):

  def setUp(self):
    jax.config.update('jax_enable_x64', USE_FLOAT64)
    super(PressureTest, self).setUp()

  def poisson_setup(self, bc, offset):
    rs = np.random.RandomState(0)
    b = rs.randn(4, 4).astype(np.float32)
    grid = grids.Grid((4, 4), domain=((0, 4), (0, 4)))  # has step = 1.0
    b = grids.GridArray(b, offset, grid)
    a = array_utils.laplacian_matrix_w_boundaries(grid, offset, bc)
    b_transformed = pressure._rhs_transform(
        bc.trim_boundary(b), bc)
    a_inv = fast_diagonalization.pseudoinverse(
        a, b.dtype, hermitian=True, circulant=False, implementation='matmul')
    x = a_inv(b_transformed)
    x = grids.GridArray(x, b.offset, grid)
    # laplacian is defined only on the interior
    x = grids.GridVariable(x, bc)
    x = fd.laplacian(x).data
    return x, b

  @parameterized.named_parameters(
      dict(testcase_name='_1D_cg',
           shape=(301,),
           solve=solve_cg,
           step=(.1,),
           seed=111),
      dict(testcase_name='_2D_cg',
           shape=(100, 100),
           solve=solve_cg,
           step=(1., 1.),
           seed=222),
      dict(testcase_name='_3D_cg',
           shape=(10, 10, 10),
           solve=solve_cg,
           step=(.1, .1, .1),
           seed=333),
      dict(testcase_name='_1D_fast_diag',
           shape=(301,),
           solve=pressure.solve_fast_diag,
           step=(.1,),
           seed=111),
      dict(testcase_name='_2D_fast_diag',
           shape=(100, 100),
           solve=pressure.solve_fast_diag,
           step=(1., 1.),
           seed=222),
      dict(testcase_name='_3D_fast_diag',
           shape=(10, 10, 10),
           solve=pressure.solve_fast_diag,
           step=(.1, .1, .1),
           seed=333),
  )
  def test_pressure_correction_periodic_bc(
      self, shape, solve, step, seed):
    """Returned velocity should be divergence free."""
    grid = grids.Grid(shape, step)
    bc = boundaries.periodic_boundary_conditions(grid.ndim)

    # The uncorrected velocity is a 1 + a small amount of noise in each
    # dimension.
    ks = jax.random.split(jax.random.PRNGKey(seed), 2 * len(shape))
    v = tuple(
        grids.GridArray(1. + .3 * jax.random.normal(k, shape), offset, grid)
        for k, offset in zip(ks[:grid.ndim], grid.cell_faces))
    v = tuple(grids.GridVariable(u, bc) for u in v)
    v_corrected = pressure.projection(v, solve)

    # The corrected velocity should be divergence free.
    div = fd.divergence(v_corrected)
    for u, u_corrected in zip(v, v_corrected):
      np.testing.assert_allclose(u.offset, u_corrected.offset)
    np.testing.assert_allclose(div.data, 0., atol=1e-4)

  @parameterized.named_parameters(
      dict(testcase_name='_1D_cg',
           shape=(10,),
           solve=solve_cg,
           step=(.1,),
           seed=111),
      dict(testcase_name='_2D_cg',
           shape=(10, 10),
           solve=solve_cg,
           step=(.1, .1),
           seed=222),
      dict(testcase_name='_3D_cg',
           shape=(10, 10, 10),
           solve=solve_cg,
           step=(.1, .1, .1),
           seed=333),
  )
  def test_pressure_correction_dirichlet_velocity_bc(
      self, shape, solve, step, seed):
    """Returned velocity should be divergence free."""
    grid = grids.Grid(shape, step)
    velocity_bc = (boundaries.dirichlet_boundary_conditions(
        grid.ndim),) * grid.ndim

    # The uncorrected velocity is zero + a small amount of noise in each
    # dimension.
    ks = jax.random.split(jax.random.PRNGKey(seed), 2 * grid.ndim)
    v = tuple(
        grids.GridArray(0. + .3 * jax.random.normal(k, shape), offset, grid)
        for k, offset in zip(ks[:grid.ndim], grid.cell_faces))
    # Set boundary velocity to zero
    masks = grids.domain_interior_masks(grid)
    self.assertLen(masks, grid.ndim)
    v = (m * u for m, u in zip(masks, v))
    # Associate boundary conditions
    v = tuple(grids.GridVariable(u, u_bc) for u, u_bc in zip(v, velocity_bc))
    self.assertLen(v, grid.ndim)
    # Apply pressure correction
    v_corrected = pressure.projection(v, solve)

    # The corrected velocity should be divergence free.
    div = fd.divergence(v_corrected)
    for u, u_corrected in zip(v, v_corrected):
      np.testing.assert_allclose(u.offset, u_corrected.offset)
    np.testing.assert_allclose(div.data, 0., atol=1e-4)

  @parameterized.parameters(
      dict(ndim=2, solve=pressure.solve_cg),
      dict(ndim=2, solve=pressure.solve_fast_diag_channel_flow),
      dict(ndim=3, solve=pressure.solve_cg),
      dict(ndim=3, solve=pressure.solve_fast_diag_channel_flow),
  )
  def test_pressure_correction_mixed_velocity_bc(self, ndim, solve):
    """Returned velocity should be divergence free."""
    shape = (20,) * ndim
    grid = grids.Grid(shape, step=0.1)
    velocity_bc = (boundaries.channel_flow_boundary_conditions(ndim),) * ndim

    def rand_array(seed):
      key = jax.random.split(jax.random.PRNGKey(seed))
      return jax.random.normal(key[0], shape)

    v = tuple(
        grids.GridArray(
            1. + .3 * rand_array(seed=d), offset=grid.cell_faces[d], grid=grid)
        for d in range(ndim))

    # Associate and enforce boundary conditions
    v = tuple(grids.GridVariable(u, u_bc).impose_bc()
              for u, u_bc in zip(v, velocity_bc))

    # y-velocity = 0 for the edge y=y_max (homogeneous Diriclet BC)
    # y-velocity on lower y-boundary is not on an edge
    # Note, x- and z-velocity do not have an edge value on the y-boundaries
    self.assertAllClose(v[1].data[:, -1, ...], 0)

    # Apply pressure correction
    v_corrected = pressure.projection(v, solve)

    # The corrected velocity should be divergence free.
    div = fd.divergence(v_corrected)
    for u, u_corrected in zip(v, v_corrected):
      np.testing.assert_allclose(u.offset, u_corrected.offset)
    np.testing.assert_allclose(div.data, 0., atol=1e-4)

  @parameterized.parameters(((1.0, 0.5),), ((1.0, 1.0),), ((1.0, 0.0),))
  def test_poisson_periodic_and_dirichlet(self, offset):
    bc = boundaries.periodic_and_dirichlet_boundary_conditions()
    x, b = self.poisson_setup(bc, offset)
    self.assertAllClose(x, bc.trim_boundary(b).data, atol=1e-5)

  @parameterized.parameters(((1.0, 0.5),), ((0.5, 0.5),))
  def test_poisson_periodic_and_neumann(self, offset):
    bc = boundaries.periodic_and_neumann_boundary_conditions()
    x, b = self.poisson_setup(bc, offset)
    self.assertAllClose(x, b.data - b.data.mean(), atol=1e-5)

  @parameterized.parameters(((1.0, 0.5),), ((1.0, 1.0),), ((1.0, 0.0),))
  def test_poisson_2d_periodic(self, offset):
    bc = boundaries.periodic_boundary_conditions(2)
    x, b = self.poisson_setup(bc, offset)
    self.assertAllClose(x, b.data - b.data.mean(), atol=1e-5)

if __name__ == '__main__':
  absltest.main()
