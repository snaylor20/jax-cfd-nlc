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

"""Non-learned "base" physics routines for JAX-CFD."""

import jax_cfd_nlc.base.advection
import jax_cfd_nlc.base.array_utils
import jax_cfd_nlc.base.boundaries
import jax_cfd_nlc.base.diffusion
import jax_cfd_nlc.base.equations
import jax_cfd_nlc.base.fast_diagonalization
import jax_cfd_nlc.base.finite_differences
import jax_cfd_nlc.base.forcings
import jax_cfd_nlc.base.funcutils
import jax_cfd_nlc.base.grids
import jax_cfd_nlc.base.initial_conditions
import jax_cfd_nlc.base.interpolation
import jax_cfd_nlc.base.pressure
import jax_cfd_nlc.base.resize
import jax_cfd_nlc.base.subgrid_models
import jax_cfd_nlc.base.time_stepping
import jax_cfd_nlc.base.validation_problems
