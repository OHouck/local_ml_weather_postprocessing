import gcsfs
import optax
import os
import socket
import pickle
import numpy as np
from datetime import datetime, timedelta
import xarray
import neuralgcm
from typing import Sequence, Callable, Optional, Dict, Any
import jax
import jax.numpy as jnp
# reference code from local version of neuralGCM
from local_neuralGCM.reference_code import metrics, metrics_util, linear_transforms, metrics_base

from dinosaur import pytree_utils
from dinosaur import coordinate_systems
from dinosaur import horizontal_interpolation
from dinosaur import xarray_utils
from dinosaur import spherical_harmonic
from dinosaur import typing
Pytree = typing.Pytree
TrajectoryRepresentations = typing.TrajectoryRepresentations
tree_map = jax.tree_util.tree_map
gcs = gcsfs.GCSFileSystem(token='anon')

# neuralgcm files that are changed..
# linear_transforms.py: used jnp instead of np for sqrt function

#==============================================================================
# Define functions and classes
#==============================================================================

def setup_directories():
    # check if we are on the server or local
    nodename = socket.gethostname()
    if nodename == "oMac.local": # local laptop
        root = os.path.expanduser("~/OneDrive - The University of Chicago/ai_weather_ag/data")
    else:
        raise Exception("Unknown environment, Please specify the root directory")

    dirs = {
        'root': root,
        'raw': os.path.join(root, "raw"),
        'processed': os.path.join(root, "processed"),
        'fig': os.path.join(root, "../figures")
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs
dir = setup_directories()

class RegionalLoss(metrics.TransformedL2Loss):
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
    # also made prediction and target dictionaries and removed the need for TrajectoryRepresentations
    # do not call get_representation 
    def evaluate_per_variable(
        self,
        prediction: Dict,
        target: Dict,
    ) -> Pytree:

        prediction = self.getter(prediction)
        target = self.getter(target)

        # Get the masks for the region of interest
        if self.is_encoded:
            coords = self.trajectory_spec.coords
        else:
            coords = self.trajectory_spec.data_coords
        region_mask = self._create_region_mask(coords)
        
        def apply_mask(x, var_name):
            if var_name in self.variables_to_slice:
                *leading_dims, lon, lat = x.shape 
                broadcast_shape = (1,) * (len(leading_dims)) + region_mask.shape
                expanded_mask = region_mask.reshape(broadcast_shape)
                expanded_mask = jnp.broadcast_to(expanded_mask, x.shape)
                return x * expanded_mask
            return x            

        # Apply mask to both trajectory and target
        prediction = {k: apply_mask(v, k) for k, v in prediction.items()}
        target = {k: apply_mask(v, k) for k, v in target.items()}

        # Continue with normal loss computation
        errors = tree_map(jnp.subtract, prediction, target)
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
        return region_mask
            
def compute_loss(model, initial_state, target, forcings, rng_key, num_outer_steps, num_inner_steps, timedelta, lat_bounds, lon_bounds):

    # Run model forward to create forecast
    _, prediction_trajectory = model.unroll(
        state = initial_state,
        forcings = forcings,
        steps=num_outer_steps,
        timedelta=timedelta,
        start_with_input=True,
    )

    # compute statistics for input data
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

    trajectory_spec = metrics_util.TrajectorySpec(
        trajectory_length=num_outer_steps,  # max "outer steps"
        max_trajectory_length=num_outer_steps, # Max length for stage of experiment
        steps_per_save=num_inner_steps, # number of 1 hour steps between outer steps
        coords=model.model_coords, # model coordinates
        data_coords=model.data_coords, # data coordinates
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

    # to rescale variables and then use weight using importance weights
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

    loss_fn = RegionalLoss(
        trajectory_spec=trajectory_spec,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        components=components,
        variables_to_slice=['temperature', 'geopotential', 
                            'u_component_of_wind', 'v_component_of_wind', 
                            'specific_humidity', 
                            'specific_cloud_liquid_water_content', 
                            'specific_cloud_ice_water_content'],
        is_nodal = True, # data is defined at grid nodes (lat, lon)
        is_encoded = False, # variables represent physical quantities 
    )

    # filter trajectories to only include last time step. OH might want to change 
    prediction_trajectory = {k: v[-1] for k, v in prediction_trajectory.items()}
    target_trajectory = {k: v[-1] for k, v in target.items()}

    # loss for all variables
    loss_dict = loss_fn.evaluate_per_variable(prediction_trajectory, target_trajectory)
    # Can think more carefully about how to combine loss from different variables
    loss_sum = sum(loss_dict.values())
    return loss_sum

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


def pull_and_regrid_era5(model, era5_path, start_date, end_date, num_inner_steps, output_path, save = False):
    start_date_short = start_date.replace('-', '')
    end_date_short = end_date.replace('-', '')
    filename = f'eval_era5_{start_date_short}_{end_date_short}.nc'
    file_path = os.path.join(output_path, filename)

    # Check if the file already exists
    if os.path.exists(file_path):
        print(f'File {filename} already exists. Loading it instead of re-evaluating.')
        return xarray.open_dataset(file_path, engine='netcdf4')

    # Open ERA5 dataset
    full_era5 = xarray.open_zarr(gcs.get_mapper(era5_path), chunks=None)
    
    # Slice the dataset to the desired time range
    sliced_era5 = (
        full_era5 
        [model.input_variables + model.forcing_variables]
        .pipe(
            xarray_utils.selective_temporal_shift,
            variables=model.forcing_variables,
            time_shift='24 hours',
        )
        .sel(time=slice(start_date, end_date, num_inner_steps))
        .compute()
    )

    # Regrid to neuralgcm resolution
    era5_grid = spherical_harmonic.Grid(
        latitude_nodes=full_era5.sizes['latitude'],
        longitude_nodes=full_era5.sizes['longitude'],
        latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        era5_grid, model.data_coords.horizontal, skipna=True
    )
    eval_era5 = xarray_utils.regrid(sliced_era5, regridder)
    eval_era5 = xarray_utils.fill_nan_with_nearest(eval_era5)

    # Save to NetCDF
    if save:

        eval_era5.to_netcdf(f"{output_path}/{filename}")
    
    return eval_era5


#==============================================================================
# Set up parameters
#==============================================================================
# set random key
rng_key = jax.random.PRNGKey(854)

# 1.4 degree pre-trained model checkpoint
model_name = 'neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl'  #@param ['neural_gcm_dynamic_forcing_deterministic_0_7_deg.pkl', 'neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl', 'neural_gcm_dynamic_forcing_deterministic_2_8_deg.pkl', 'neural_gcm_dynamic_forcing_stochastic_1_4_deg.pkl'] {type: "string"}

# set time parameters
start_date = '2020-02-14'
num_days = 2
num_inner_steps = 24 # save model output once every 24 hours
data_inner_steps = 24  # process every 24th hour

# Set region of interest: Note:
# latitude between -90 and 90
# longitude between 0 and 360

# global
# lat_bounds = (-90, 90)
# lon_bounds = (0, 360)

# pakistan
lat_bounds = (20, 60)
lon_bounds = (200, 300)


#==============================================================================
# Set up model and data
#==============================================================================

# set other time parameters based on start date and number of days
end_date = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=num_days)
end_date = end_date.strftime('%Y-%m-%d')
num_outer_steps = num_days * 24 // num_inner_steps # process num_days days
timedelta = np.timedelta64(1, 'h') * num_inner_steps
times = np.arange(num_outer_steps) * num_inner_steps # time axis in hours

# convert coordinate bounds to radians
lat_bounds = (np.deg2rad(lat_bounds[0]), np.deg2rad(lat_bounds[1]))
lon_bounds = (np.deg2rad(lon_bounds[0]), np.deg2rad(lon_bounds[1]))

# Load a non-toy version of the model
# with gcs.open(f'gs://gresearch/neuralgcm/04_30_2024/{model_name}', 'rb') as f:
#   ckpt = pickle.load(f)
# model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)    

# simple demo version for quickest testing, now outdated
checkpoint = neuralgcm.demo.load_checkpoint_tl63_stochastic()
model = neuralgcm.PressureLevelModel.from_checkpoint(checkpoint)
# ds = neuralgcm.demo.load_data(model.data_coords)
# inputs, forcings = model.data_from_xarray(ds.isel(time=0))

era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
output_path = dir['processed']
eval_era5 = pull_and_regrid_era5(model, era5_path, start_date, end_date, num_inner_steps, output_path, save = True)

inputs = model.inputs_from_xarray(eval_era5.isel(time = 0))
input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))
initial_state = model.encode(inputs, input_forcings, rng_key)
forcings = model.forcings_from_xarray(eval_era5.head(time=1))

target_trajectory = model.inputs_from_xarray(
    eval_era5
    .thin(time=(num_inner_steps // data_inner_steps))
    .isel(time=slice(num_outer_steps))
)

# set up optimizer settings
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(model)

# JIT-compile the training function
compute_loss_jit = jax.jit(compute_loss, static_argnums=(5, 6, 7, 8, 9))

#==============================================================================
# Run training loop
#==============================================================================
for i in range(5):
    print(f'Iteration {i+1}')
    loss, grads = jax.value_and_grad(compute_loss_jit)(
        model, initial_state, target_trajectory, forcings, rng_key, num_outer_steps, num_inner_steps, timedelta, lat_bounds, lon_bounds
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    frozen_updates, pct_unfrozen = freeze_non_decoder_params(model, updates)
    model = optax.apply_updates(model, frozen_updates)
    print(f'{i+1=}, loss = {loss.item()}')
 