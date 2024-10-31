import optax
import numpy as np
import neuralgcm
from typing import Sequence, Callable, Optional, Dict, Any
import jax
import jax.numpy as jnp
# reference code from local version of neuralGCM
from local_neuralGCM.reference_code import metrics, metrics_util, linear_transforms

from dinosaur import pytree_utils
from dinosaur import coordinate_systems
from dinosaur import typing
Pytree = typing.Pytree
tree_map = jax.tree_util.tree_map

class CustomLoss(metrics.TransformedL2Loss):
    def __init__(
        self,
        trajectory_spec: metrics_util.TrajectorySpec,
        lat_bounds: tuple,
        lon_bounds: tuple,
        variables_to_slice: Sequence[str],
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
        self.variables_to_slice = set(variables_to_slice)

        # Create regional mask
        coords = self.trajectory_spec.coords if self.is_encoded else self.trajectory_spec.data_coords
        self.lat_mask, self.lon_mask, self.region_mask = self._create_region_mask(coords)

    def evaluate_per_variable(
        self,
        prediction: Dict[str, Any], # maybe should be trajectories? XX
        target: Dict[str, Any],
    ) -> Pytree:
        trajectory = self.getter(prediction)
        target = self.getter(target)

        # slice the lat lon bounds
        trajectory = self._get_spatial_slice(trajectory) 
        target = self._get_spatial_slice(target)

        # difference between target and the trajectory
        errors = jax.tree_util.tree_map(jnp.subtract, trajectory, target) # change prediction to trajectory if revert to original

        transformed_errors = self.transform(errors, target)

        squared_transformed_errors = jax.tree_util.tree_map(jnp.square, transformed_errors)

        # XX issue is that this fucntion is meant for global loss not loss for a region
        return self.mean_per_variable(squared_transformed_errors)

    def _create_region_mask(self, coords):
        # Get full latitudes and longitudes in degrees
        full_latitudes = coords.horizontal.latitudes  # Shape: (64,)
        full_longitudes = coords.horizontal.longitudes  # Shape: (128,)

        lat_min = lat_bounds[0]
        lat_max = lat_bounds[1]
        lon_min = lon_bounds[0]
        lon_max = lon_bounds[1]

        # Create boolean masks
        lat_mask = (full_latitudes >= lat_min) & (full_latitudes <= lat_max)
        lon_mask = (full_longitudes >= lon_min) & (full_longitudes <= lon_max)

        # Create a 2D mask
        region_mask = np.outer(lon_mask, lat_mask).astype(float)  # Shape: (128, 64)
        # region_mask = region_mask.T  # Shape: (64, 128), matching (lat, lon) don't think we want
        return lat_mask, lon_mask, region_mask

    def _get_spatial_slice(self, data: Pytree) -> Pytree:
        lat_mask = self.lat_mask
        lon_mask = self.lon_mask

        def slice_data(var_name, x):
            if var_name in self.variables_to_slice:
                if x.ndim == 3:  # (level, lon, lat)
                    assert x.shape[1:] == (128, 64), f"Expected shape (_, 128, 64), got {x.shape}"
                    return x[:, lon_mask][:, :, lat_mask]
                elif x.ndim == 4:  # (time, level, lon, lat)
                    assert x.shape[2:] == (128, 64), f"Expected shape (_, _, 128, 64), got {x.shape}"
                    return x[:, :, lon_mask][:, :, :, lat_mask]
            return x

        return {var_name: slice_data(var_name, x) for var_name, x in data.items()}

    def surface_mean(self, trajectory: Pytree) -> Pytree:
        coords = self.trajectory_spec.coords if self.is_encoded else self.trajectory_spec.data_coords
        region_mask = self.region_mask
        if self.is_nodal:
            fn = lambda x: metrics_util.regional_nodal_surface_mean(x, coords, region_mask)
        else:
            print('MODAL coords! probably should not be here!!')
            fn = lambda x: metrics_util.modal_surface_mean(x, coords)
        return tree_map(fn, trajectory)

def compute_loss(model, inputs, forcings, rng, lat_bounds, lon_bounds):
    # Define the number of days to predict
    num_days = 1
    
    # Calculate the number of steps based on the model's timestep
    # Assuming the model's timestep is in hours
    steps_per_day = 24 // model.timestep.astype('timedelta64[h]').astype(int)
    total_steps = num_days * steps_per_day

    trajectory_spec = metrics_util.TrajectorySpec(
        trajectory_length=1,  # Adjust as needed
        max_trajectory_length=1,
        steps_per_save=1,
        coords=model.model_coords,
        data_coords=model.data_coords,
    )

    # Define weights for each variable: these are used to scale the loss for each variable
    weights = {
        'temperature': 1.0,
        'geopotential': 1.0,
        'specific_cloud_ice_water_content': 1.0,
        'specific_cloud_liquid_water_content': 1.0,
        'specific_humidity': 1.0,
        'u_component_of_wind': 1.0,
        'v_component_of_wind': 1.0,
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
        variables_to_slice=['temperature', 'geopotential', 
                            'u_component_of_wind', 'v_component_of_wind', 
                            'specific_humidity', 
                            'specific_cloud_liquid_water_content', 
                            'specific_cloud_ice_water_content'],
    )

    encoded = model.encode(inputs, forcings, rng_key=rng)

     # Unroll the model for 5 days
    # _, predictions = model.unroll(encoded, forcings, steps=total_steps)
    predictions = model.decode(encoded, forcings)

    # initial state repeated for total_steps XX place holder code
    # target = jax.tree_map(lambda x: jnp.repeat(x[jnp.newaxis, ...], total_steps, axis=0), inputs)
    target = inputs

    loss = loss_fn.evaluate_per_variable(predictions, target)
    return loss

# JIT-compile the function
compute_loss_jit = jax.jit(compute_loss, static_argnums=(0, 4, 5))

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

# helper function for debugging
def find_decoder_params(model):
    '''Identify decoder parameters and print them'''
    for path, param in model.params.items():
        if 'decode' in str(path):
            print(path)


checkpoint = neuralgcm.demo.load_checkpoint_tl63_stochastic()
initial_model = neuralgcm.PressureLevelModel.from_checkpoint(checkpoint)

ds = neuralgcm.demo.load_data(initial_model.data_coords)
inputs, forcings = initial_model.data_from_xarray(ds.isel(time=0))

optimizer = optax.adam(1e-3)


# latitude between -90 and 90
# longitude between 0 and 360
# lat_bounds = (-90, 90)
# lon_bounds = (0, 360)

# pakistan
lat_bounds = (20, 60)
lon_bounds = (200, 300)

# convert to radians
lat_bounds = (np.deg2rad(lat_bounds[0]), np.deg2rad(lat_bounds[1]))
lon_bounds = (np.deg2rad(lon_bounds[0]), np.deg2rad(lon_bounds[1]))

rng = jax.random.PRNGKey(0)

opt_state = optimizer.init(initial_model)

# 128 longitude, 64 latitude
model = initial_model

for i in range(5):
    loss, grads = jax.value_and_grad(compute_loss_jit)(model, inputs, forcings, rng, lat_bounds, lon_bounds)
    updates, opt_state = optimizer.update(grads, opt_state)
    frozen_updates, pct_unfrozen = freeze_non_decoder_params(model, updates)

    model = optax.apply_updates(model, frozen_updates)
    print(f'{i=}, loss = {loss.item()}')
    exit()
# i=0, loss=Array(6.2256584, dtype=float32)
# i=1, loss=Array(4.670498, dtype=float32)
# i=2, loss=Array(3.855668, dtype=float32)
# i=3, loss=Array(3.5485578, dtype=float32)
# i=4, loss=Array(3.4335625, dtype=float32)