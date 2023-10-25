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

"""Pseudospectral equations."""

import dataclasses
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import forcings
from jax_cfd.base import grids
from jax_cfd.spectral import forcings as spectral_forcings
from jax_cfd.spectral import time_stepping
from jax_cfd.spectral import types
from jax_cfd.spectral import utils as spectral_utils


TimeDependentForcingFn = Callable[[float], types.Array]
RandomSeed = int
ForcingModule = Callable[[grids.Grid, RandomSeed], TimeDependentForcingFn]

rfft = jax.jit(jnp.fft.rfft)
irfft = jax.jit(jnp.fft.irfft)

@jax.jit
def product_k_kmm(m, k, U_h, V_h, gamma, K):

    return U_h[m + K] * V_h[k-m + 2*K] * gamma[m - K, k-m-K]

@jax.jit
def trace_k(k, U_h, V_h, gamma):

    K = (U_h.shape[0]-1)//2
    ms = jnp.arange(-K, K+1)

    sum_me = jax.vmap(product_k_kmm, in_axes=(0, None, None, None, None, None))(ms, k, U_h, V_h, gamma, K)
    return jnp.sum(sum_me)

@jax.jit
def nl_four_aa(u_h, v_h, gamma):

    K = u_h.shape[0] - 1
    K_aa = 2*K

    U_h = jnp.hstack((jnp.conj(u_h[1:][::-1]), u_h))
    V_h = jnp.hstack((jnp.zeros(K), jnp.conj(v_h[1:][::-1]), v_h))#, jnp.zeros(K)))

    ks = jnp.arange(K_aa+1)

    UV_h = jax.vmap(trace_k, in_axes=(0, None, None, None))(ks, U_h, V_h, gamma)

    UV = irfft(UV_h)
    uv_h = 2 * rfft(UV[::2])

    return uv_h


@dataclasses.dataclass
class KuramotoSivashinsky_nlc(time_stepping.ImplicitExplicitODE):
  """Kuramoto–Sivashinsky (KS) equation split in implicit and explicit parts.

  The KS equation is
    u_t = - u_xx - u_xxxx - 1/2 * (u ** 2)_x

  Implicit parts are the linear terms and explicit parts are the non-linear
  terms.

  Attributes:
    grid: underlying grid of the process
    smooth: smooth the non-linear term using the 3/2-rule
  """
  grid: grids.Grid
  smooth: bool = True

  def __post_init__(self):
    self.kx, = self.grid.rfft_axes()
    self.two_pi_i_k = 2j * jnp.pi * self.kx
    self.linear_term = -self.two_pi_i_k ** 2 - self.two_pi_i_k ** 4

  def explicit_terms(self, uhat, gamma):
    """Non-linear parts of the equation, namely `- 1/2 * (u ** 2)_x`."""
    n = (uhat.shape[0] - 1)*2 
    uhat_squared = nl_four_aa(uhat/n, uhat/n, gamma)*n
    return -0.5 * self.two_pi_i_k * uhat_squared

  def implicit_terms(self, uhat):
    """Linear parts of the equation, namely `- u_xx - u_xxxx`."""
    return self.linear_term * uhat

  def implicit_solve(self, uhat, time_step):
    """Solves for `implicit_terms`, implicitly."""
    # TODO(dresdner) the same for all linear terms. generalize/refactor?
    return 1 / (1 - time_step * self.linear_term) * uhat
