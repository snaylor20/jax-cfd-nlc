"""Decoder modules that help interfacing model states with output data.

All decoder modules generate a function that given an specific model state
return the observable data of the same structure as provided to the Encoder.
Decoders can be either fixed functions, decorators, or learned modules.
"""

from typing import Any, Callable, Optional

import gin
import haiku as hk
import jax.numpy as jnp
from jax_cfd_nlc.base import array_utils
from jax_cfd_nlc.base import grids
from jax_cfd_nlc.base import interpolation
from jax_cfd_nlc.ml import physics_specifications
from jax_cfd_nlc.ml import towers
from jax_cfd_nlc.spectral import utils as spectral_utils


DecodeFn = Callable[[Any], Any]  # maps model state to data time slice.
DecoderModule = Callable[..., DecodeFn]  # generate DecodeFn closed over args.
TowerFactory = towers.TowerFactory


@gin.register
def identity_decoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
) -> DecodeFn:
  """Identity decoder module that returns model state as is."""
  del grid, dt, physics_specs  # unused.
  def decode_fn(inputs):
    return inputs

  return decode_fn


# TODO(dkochkov) generalize this to arbitrary pytrees.
@gin.register
def aligned_array_decoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
) -> DecodeFn:
  """Generates decoder that extracts data from GridVariables."""
  del grid, dt, physics_specs  # unused.
  def decode_fn(inputs):
    return tuple(x.data for x in inputs)

  return decode_fn


@gin.register
def staggered_to_collocated_decoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
):
  """Decoder that interpolates from staggered to collocated grids."""
  del dt, physics_specs  # unused.
  def decode_fn(inputs):
    interp_inputs = [interpolation.linear(c, grid.cell_center) for c in inputs]
    return tuple(x.data for x in interp_inputs)

  return decode_fn


@gin.register
def channels_split_decoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
) -> DecodeFn:
  """Generates decoder that splits channels into data tuples."""
  del grid, dt, physics_specs  # unused.
  def decode_fn(inputs):
    return array_utils.split_axis(inputs, -1)

  return decode_fn


@gin.register
def latent_decoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    tower_factory: TowerFactory,
    num_components: Optional[int] = None,
):
  """Generates trainable decoder that maps latent representation to data tuple.

  Decoder first computes an array of outputs using network specified by a
  `tower_factory` and then splits the channels into `num_components` components.

  Args:
    grid: grid representing spatial discritization of the system.
    dt: time step to use for time evolution.
    physics_specs: physical parameters of the simulation.
    tower_factory: factory that produces trainable tower network module.
    num_components: number of data tuples in the data representation of the
      state. If None, assumes num_components == grid.ndims. Default is None.

  Returns:
    decode function that maps latent state `inputs` at given time to a tuple of
    `num_components` data arrays representing the same state at the same time.
  """
  split_channels_fn = channels_split_decoder(grid, dt, physics_specs)

  def decode_fn(inputs):
    num_channels = num_components or grid.ndim
    decoder_tower = tower_factory(num_channels, grid.ndim, name='decoder')
    return split_channels_fn(decoder_tower(inputs))

  return hk.to_module(decode_fn)()


@gin.register
def aligned_latent_decoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
    tower_factory: TowerFactory,
    num_components: Optional[int] = None,
):
  """Latent decoder that decodes from aligned arrays."""
  split_channels_fn = channels_split_decoder(grid, dt, physics_specs)

  def decode_fn(inputs):
    inputs = jnp.stack([x.data for x in inputs], axis=-1)
    num_channels = num_components or grid.ndim
    decoder_tower = tower_factory(num_channels, grid.ndim, name='decoder')
    return split_channels_fn(decoder_tower(inputs))

  return hk.to_module(decode_fn)()


@gin.register
def vorticity_decoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
) -> DecodeFn:
  """Solves for velocity and converts into GridVariables."""
  del dt, physics_specs  # unused.
  velocity_solve = spectral_utils.vorticity_to_velocity(grid)
  def decode_fn(vorticity):
    # TODO(dresdner) note the main difference is the input, which is in real space instead of vorticity space
    vorticity = jnp.squeeze(vorticity, axis=-1)  # remove channel dim
    vorticity_hat = jnp.fft.rfft2(vorticity)
    uhat, vhat = velocity_solve(vorticity_hat)
    v = (jnp.fft.irfft2(uhat), jnp.fft.irfft2(vhat))
    return v

  return decode_fn


@gin.register
def spectral_vorticity_decoder(
    grid: grids.Grid,
    dt: float,
    physics_specs: physics_specifications.BasePhysicsSpecs,
) -> DecodeFn:
  """Solves for velocity and converts into GridVariables."""
  del dt, physics_specs  # unused.
  velocity_solve = spectral_utils.vorticity_to_velocity(grid)
  def decode_fn(vorticity_hat):
    uhat, vhat = velocity_solve(vorticity_hat)
    v = (jnp.fft.irfft2(uhat), jnp.fft.irfft2(vhat))
    return v

  return decode_fn
