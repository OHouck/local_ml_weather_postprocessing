import optax
import numpy as np
import neuralgcm
from typing import Sequence, Callable, Optional, Dict, Any
import jax
import jax.numpy as jnp
# reference code from local version of neuralGCM
from local_neuralGCM.reference_code import metrics, metrics_util, linear_transforms, metrics_base

from dinosaur import pytree_utils
from dinosaur import coordinate_systems
from dinosaur import typing
Pytree = typing.Pytree
TrajectoryRepresentations = typing.TrajectoryRepresentations
tree_map = jax.tree_util.tree_map

# neuralgcm files that are changed..
# linear_transforms.py: used jnp instead of np for sqrt function


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
        self.variables_to_slice = variables_to_slice

    # modifed version of method in metrics.py to incorporate regional masking
    def evaluate_per_variable(
        self,
        prediction: TrajectoryRepresentations,
        target: TrajectoryRepresentations,
    ) -> Pytree:
        prediction = self.get_representation(prediction)
        target = self.get_representation(target)
        trajectory = self.getter(prediction)
        target = self.getter(target)

        # Get the masks for the region of interest
        if self.is_encoded:
            coords = self.trajectory_spec.coords
        else:
            coords = self.trajectory_spec.data_coords
        lat_mask, lon_mask, region_mask = self._create_region_mask(coords)
        
        # # Apply mask to relevant variables
        # def apply_mask(x, var_name):
        #     if var_name in self.variables_to_slice:
        #         # Expand mask dimensions to match the variable shape
        #         # Assuming shape is (time, level, lon, lat) or similar
        #         expanded_mask = region_mask[None, None, :, :]  # Add time and level dims
        #         expanded_mask = jnp.broadcast_to(expanded_mask, x.shape)
        #         return x * expanded_mask
        #     return x
        def apply_mask(x, var_name):
            if var_name in self.variables_to_slice:
                print(f"Variable {var_name} shape: {x.shape}")
                *leading_dims, lon, lat = x.shape
                broadcast_shape = (1,) * (len(leading_dims)) + region_mask.shape
                print(f"Mask broadcast shape: {broadcast_shape}")
                expanded_mask = region_mask.reshape(broadcast_shape)
                expanded_mask = jnp.broadcast_to(expanded_mask, x.shape)
                return x * expanded_mask
            return x            

        # Apply mask to both trajectory and target
        trajectory = {k: apply_mask(v, k) for k, v in trajectory.items()}
        target = {k: apply_mask(v, k) for k, v in target.items()}

        # Continue with normal loss computation
        errors = tree_map(jnp.subtract, trajectory, target)
        transformed_errors = self.transform(errors, target)
        squared_transformed_errors = tree_map(jnp.square, transformed_errors)
    
        # When taking mean, we should only consider points within the mask
        def masked_mean(x, var_name):
            if var_name in self.variables_to_slice:
                # Count number of points in mask for proper averaging
                n_points = jnp.sum(region_mask) * jnp.prod(jnp.array(x.shape[:-2]))
                # Sum over spatial dimensions and divide by number of masked points
                return jnp.sum(x) / n_points
            return jnp.mean(x)

        return {k: masked_mean(v, k) for k, v in squared_transformed_errors.items()}

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
            
def compute_loss(model, inputs, forcings, rng, lat_bounds, lon_bounds):

    # compute statustics for input data
    def compute_stats(x):
        return {
            'mean': jnp.mean(x),
            'std': jnp.std(x) + 1e-8 # avoid division by 0
        }

    # Filter out metadata fields and only compute stats for actual variables
    variables_to_normalize = {
        'temperature': inputs['temperature'],
        'geopotential': inputs['geopotential'],
        'specific_cloud_ice_water_content': inputs['specific_cloud_ice_water_content'],
        'specific_cloud_liquid_water_content': inputs['specific_cloud_liquid_water_content'],
        'specific_humidity': inputs['specific_humidity'],
        'u_component_of_wind': inputs['u_component_of_wind'],
        'v_component_of_wind': inputs['v_component_of_wind']
    }
    input_stats = jax.tree_util.tree_map(compute_stats, variables_to_normalize)

    # Define the number of days to predict
    num_days = 1
    
    # Calculate the number of steps based on the model's timestep
    # Assuming the model's timestep is in hours
    steps_per_day = 24 // model.timestep.astype('timedelta64[h]').astype(int)
    total_steps = num_days * steps_per_day

    trajectory_spec = metrics_util.TrajectorySpec(
        trajectory_length=1,  # Adjust as needed only going a short ways forward
        max_trajectory_length=1,
        steps_per_save=1,
        coords=model.model_coords,
        data_coords=model.data_coords,
    )

    # Define variable-specific weights to handle different scales
    importance_weights = {
        'temperature': 1.0,
        'geopotential': 1.0,
        'specific_cloud_ice_water_content': 1.0,
        'specific_cloud_liquid_water_content': 1.0,
        'specific_humidity': 1.0,
        'u_component_of_wind': 1.0,
        'v_component_of_wind': 1.0,
    }

    # to rescale variables and then use importance weights
    components = [
        linear_transforms.LegacyTimeRescaling,
        lambda *args, **kwargs: linear_transforms.PerVariableRescaling(
            *args, 
            weights={k: 1/v['std']**2 for k, v in input_stats.items()}, 
            **kwargs
        ),
        lambda *args, **kwargs: linear_transforms.PerVariableRescaling(
            *args, 
            weights= importance_weights, 
            **kwargs
        )
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
    prediction = model.decode(encoded, forcings)

    # initial state repeated for total_steps XX place holder code
    # target = jax.tree_util.tree_map(lambda x: jnp.repeat(x[jnp.newaxis, ...], total_steps, axis=0), inputs)
    target = inputs

    # Convert prediction and target to TrajectoryRepresentations
    def create_trajectory_representations(data):
        # Get both nodal and modal representations in data and model space
        data_nodal = data
        data_modal = coordinate_systems.maybe_to_modal(data_nodal, model.data_coords)
        model_nodal = model.encode(data_nodal, forcings, rng_key=rng).state  # Get state from encoded output
        model_modal = coordinate_systems.maybe_to_modal(model_nodal, model.model_coords)
        
        return TrajectoryRepresentations(
            data_nodal_trajectory=data_nodal,
            data_modal_trajectory=data_modal,
            model_nodal_trajectory=model_nodal,
            model_modal_trajectory=model_modal
        )

    prediction = create_trajectory_representations(prediction)
    target = create_trajectory_representations(target)

    # check that they are TrajectoryRepresentations
    assert isinstance(prediction, TrajectoryRepresentations)
    assert isinstance(target, TrajectoryRepresentations)

    # loss for all variables
    loss_dict = loss_fn.evaluate_per_variable(prediction, target)
    # Can think more carefully about how to combine loss from different variables
    loss_sum = sum(loss_dict.values())

    return loss_sum

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

# helper function to get number of params in decoder (58k)
def count_decoder_parameters(model):
    '''Count the number of parameters in the decoder that can be retrained.'''
    retrainable_params = 0

    for path, param in jax.tree_util.tree_leaves_with_path(model.params):
        if 'dimensional_learned_primitive_to_weatherbench_decoder' in str(path):
            retrainable_params += jnp.size(param)

    print(retrainable_params)

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