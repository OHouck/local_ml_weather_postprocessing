import optax
import neuralgcm
from typing import Sequence, Callable, Optional, Dict, Any
import jax
import jax.numpy as jnp
# reference code from local version of neuralGCM
from local_neuralGCM.reference_code import metrics, metrics_util, linear_transforms

from dinosaur import pytree_utils
from dinosaur import typing
Pytree = typing.Pytree

class CustomLoss(metrics.TransformedL2Loss):
    def __init__(
        self,
        trajectory_spec: metrics_util.TrajectorySpec,
        lat_bounds: tuple,
        lon_bounds: tuple,
        components: Sequence[linear_transforms.LinearTransformConstructor] = (),
        is_nodal: bool = True,
        is_encoded: bool = False,
        getter: Callable[[Pytree], Pytree] = metrics_util.filter_sim_time,
        time_step: Optional[int | slice] = None,
    ):
        super().__init__(
            trajectory_spec,
            components,
            is_nodal=is_nodal,
            is_encoded=is_encoded,
            getter=getter,
            time_step=time_step,
        )
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds

    # def evaluate_per_variable(
    #     self,
    #     prediction: metrics_util.TrajectoryRepresentations,
    #     target: metrics_util.TrajectoryRepresentations,
    # ) -> Pytree:
    #     prediction = self.get_representation(prediction)
    #     target = self.get_representation(target)
    #     trajectory = self.getter(prediction)
    #     target = self.getter(target)
    def evaluate_per_variable(
        self,
        prediction: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Pytree:
        prediction = self.getter(prediction)
        target = self.getter(target)

        # Apply spatial masking
        # trajectory = self._apply_spatial_mask(trajectory) # XX will likely want to switch back to this
        prediction = self._apply_spatial_mask(prediction)
        target = self._apply_spatial_mask(target)

        errors = jax.tree_util.tree_map(jnp.subtract, trajectory, target)
        transformed_errors = self.transform(errors, target)
        squared_transformed_errors = jax.tree_util.tree_map(jnp.square, transformed_errors)
        return self.mean_per_variable(squared_transformed_errors)

    def _apply_spatial_mask(self, data: Pytree) -> Pytree:
        lat_min, lat_max = self.lat_bounds
        lon_min, lon_max = self.lon_bounds

        coords = self.trajectory_spec.coords if self.is_encoded else self.trajectory_spec.data_coords
        lats = coords.horizontal.latitudes
        lons = coords.horizontal.longitudes

        lat_mask = (lats >= lat_min) & (lats <= lat_max)
        lon_mask = (lons >= lon_min) & (lons <= lon_max)
        # spatial_mask = lat_mask[:, None] & lon_mask[None, :] # XX I think the wrong order
        spatial_mask = lon_mask[:, None] & lat_mask[None, :]

        def apply_mask(x):
            if x.ndim == 3:  # (level, lat, lon)
                return x * spatial_mask[None, :, :]
            elif x.ndim == 4:  # (time, level, lat, lon)
                return x * spatial_mask[None, None, :, :]
            else:
                return x

        return jax.tree_util.tree_map(apply_mask, data)

def compute_loss(model, inputs, forcings, rng, lat_bounds, lon_bounds):
    trajectory_spec = metrics_util.TrajectorySpec(
        trajectory_length=1,  # Adjust as needed
        max_trajectory_length=1,
        steps_per_save=1,
        coords=model.model_coords,
        data_coords=model.data_coords,
    )

    # Define weights for each variable
    weights = {
        'temperature': 1.0, # maybe should be "t"?
        'z': 1.0,
        'u': 1.0,
        'v': 1.0,
        'tracers': {'specific_humidity': 1.0},
        # Add other variables as needed
    }

    components = [
        linear_transforms.LegacyTimeRescaling,
        lambda *args, **kwargs: linear_transforms.PerVariableRescaling(*args, weights=weights, **kwargs)
    ]

    loss_fn = CustomLoss(
        trajectory_spec=trajectory_spec,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        components=components,
    )

    encoded = model.encode(inputs, forcings, rng_key=rng)
    predictions = model.decode(encoded, forcings)

    loss = loss_fn.evaluate(predictions, inputs)
    return loss

# JIT-compile the function
compute_loss_jit = jax.jit(compute_loss, static_argnums=(0, 4, 5))

# old simple loss function
# def compute_loss(model, inputs, forcings, rng, lat_bounds = (0, 180), lon_bounds = (0, 360)):
#     encoded = model.encode(inputs, forcings, rng_key=rng)
#     predictions = model.decode(encoded, forcings)

#     lat_min, lat_max = lat_bounds
#     lon_min, lon_max = lon_bounds

#     # Convert geographic coordinates to array indices
#     # Assuming 128 longitude steps from 0 to 360, and 64 latitude steps from -90 to 90
#     lon_idx_min = int(lon_min * 128 / 360)
#     lon_idx_max = int(lon_max * 128 / 360)
#     lat_idx_min = int((lat_min + 90) * 64 / 180)
#     lat_idx_max = int((lat_max + 90) * 64 / 180)

#     # Calculate slice sizes
#     lon_slice_size = lon_idx_max - lon_idx_min
#     lat_slice_size = lat_idx_max - lat_idx_min

#     # Initialize total loss
#     total_loss = 0.0
#     num_variables = 0

#     # Iterate over all variables in inputs
#     for var_name in inputs.keys():
#         if inputs[var_name].ndim == 3:  # Only process 3D variables (not things like time)
#             # Apply the geographic mask to both inputs and predictions using dynamic_slice
#             # there are 37 pressure levels
#             inputs_masked = jax.lax.dynamic_slice(
#                 inputs[var_name], 
#                 (0, lon_idx_min, lat_idx_min), 
#                 (37, lon_slice_size, lat_slice_size)
#             )
#             predictions_masked = jax.lax.dynamic_slice(
#                 predictions[var_name], 
#                 (0, lon_idx_min, lat_idx_min), 
#                 (37, lon_slice_size, lat_slice_size)
#             )

#             # Normalized mean absolute error
#             var_range = jnp.max(inputs_masked) - jnp.min(inputs_masked)
#             var_loss = jnp.mean(jnp.abs(inputs_masked - predictions_masked) / var_range)
#             total_loss += var_loss
#             num_variables += 1

#     # Calculate the average loss across all variables
#     average_loss = total_loss / num_variables if num_variables > 0 else 0.0
#     return average_loss

# # JIT-compile the function
# compute_loss_jit = jax.jit(compute_loss, static_argnums=(0, 4, 5))

def freeze_non_decoder_params(model, updates):
    total_params = 0
    unfrozen_params = 0

    # First, let's create a mapping from flattened indices to full parameter paths
    flat_params, tree_def = jax.tree_util.tree_flatten(model.params)
    flat_to_full = {}
    for i, (path, _) in enumerate(jax.tree_util.tree_leaves_with_path(model.params)):
        flat_to_full[i] = '/'.join(str(p) for p in path)

    def is_decoder_param(path, _):
        if isinstance(path[0], jax.tree_util.FlattenedIndexKey):
            full_path = flat_to_full.get(path[0].key, "")
        else:
            full_path = '/'.join(str(p) for p in path)

        # name in all decoder parameters
        return 'dimensional_learned_primitive_to_weatherbench_decoder' in full_path

    def maybe_freeze(path, update):
        nonlocal total_params, unfrozen_params
        total_params += jnp.size(update)
        if is_decoder_param(path, update):
            unfrozen_params += jnp.size(update)
            return update
        else:
            return jnp.zeros_like(update)

    frozen_updates, tree_def = jax.tree_util.tree_flatten(updates)
    frozen_updates = [maybe_freeze((jax.tree_util.FlattenedIndexKey(i),), update) 
                      for i, update in enumerate(frozen_updates)]
    frozen_updates = jax.tree_util.tree_unflatten(tree_def, frozen_updates)

    pct_unfrozen = unfrozen_params / total_params if total_params > 0 else 0.0
    return frozen_updates, pct_unfrozen


def find_decoder_params(model):
    # print all parameter names that contain "decode"
    for path, param in model.params.items():
        if 'decode' in str(path):
            print(path)


checkpoint = neuralgcm.demo.load_checkpoint_tl63_stochastic()
initial_model = neuralgcm.PressureLevelModel.from_checkpoint(checkpoint)

ds = neuralgcm.demo.load_data(initial_model.data_coords)
inputs, forcings = initial_model.data_from_xarray(ds.isel(time=0))

optimizer = optax.adam(1e-3)

lat_min = 20
lat_max = 60
lon_min = 200
lon_max = 300

lat_bounds = (lat_min, lat_max)
lon_bounds = (lon_min, lon_max)

rng = jax.random.PRNGKey(0)

opt_state = optimizer.init(initial_model)

model = initial_model

for i in range(5):
    loss, grads = jax.value_and_grad(compute_loss_jit)(model, inputs, forcings, rng, lat_bounds, lon_bounds)
    updates, opt_state = optimizer.update(grads, opt_state)
    frozen_updates, pct_unfrozen = freeze_non_decoder_params(model, updates)

    model = optax.apply_updates(model, frozen_updates)
    print(f'{i=}, {loss=}')
    exit()
# i=0, loss=Array(6.2256584, dtype=float32)
# i=1, loss=Array(4.670498, dtype=float32)
# i=2, loss=Array(3.855668, dtype=float32)
# i=3, loss=Array(3.5485578, dtype=float32)
# i=4, loss=Array(3.4335625, dtype=float32)