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
from local_neuralGCM.reference_code import metrics, metrics_util, linear_transforms 

from dinosaur import pytree_utils
from dinosaur import coordinate_systems
from dinosaur import horizontal_interpolation
from dinosaur import xarray_utils
from dinosaur import spherical_harmonic
from dinosaur import typing

# for experimenting with adding finetuning mlp to neuralGCM
from flax import nnx
from flax.core import freeze, unfreeze

Pytree = typing.Pytree
TrajectoryRepresentations = typing.TrajectoryRepresentations
tree_map = jax.tree_util.tree_map
gcs = gcsfs.GCSFileSystem(token='anon')

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


class FineTuningMLP(nnx.Module):
    """
    OH: currently not used still need to implement. 
        For now just ported pytorch simpleMLP class 
        from weatherbench2_finetune.py to flax

    Simple MLP for post-processing neuralGCM output
    """
    input_dim: int
    output_dim: int = 1
    num_hidd_layers: int = 3

    def setup(self):
        self.layers = []

        # first layer   
        self.layers.append(nnx.Dense(features=self.hiddem_dim))
        self.layers.append(nnx.relu)

        # hidden layers
        for _ in range(self.num_hidd_layers - 1):
            self.layers.append(nnx.Dense(features=self.hiddem_dim))
            self.layers.append(nnx.relu)

        # output layer
        self.layers.append(nnx.Dense(features=self.output_dim))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x


class RegionalLoss(metrics.TransformedL2Loss):
    def __init__(
        self,
        trajectory_spec: metrics_util.TrajectorySpec,
        lat_bounds: tuple,
        lon_bounds: tuple,
        variables_to_slice: Sequence[str],
        pressure_weights: Dict[str, jnp.ndarray],
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
        self.pressure_weights = pressure_weights

    def evaluate_per_variable(
        self,
        prediction: Dict,
        target: Dict,
    ) -> Pytree:

        prediction = self.getter(prediction)
        target = self.getter(target)

        if self.is_encoded:
            coords = self.trajectory_spec.coords
        else:
            coords = self.trajectory_spec.data_coords

        region_mask = self._create_region_mask(coords)
        
        def apply_mask(x, var_name):
            if var_name in self.variables_to_slice:
                if var_name == "P_minus_E_cumulative":
                    # shape might be (time, level, lon, lat) in target
                    # and (time, lon, lat) in prediction
                    # ensure we remove level dimension for the target if needed
                    if x.ndim == 4:
                        x = x[:, 0, :, :]
                    *leading_dims, lon, lat = x.shape 
                elif x.ndim == 4:
                    *leading_dims, lon, lat = x.shape 
                else:
                    raise ValueError(f"Unexpected number of dimensions {x.ndim} for variable {var_name}.")

                broadcast_shape = (1,) * (len(leading_dims)) + region_mask.shape
                expanded_mask = region_mask.reshape(broadcast_shape)
                expanded_mask = jnp.broadcast_to(expanded_mask, x.shape)
                return x * expanded_mask
            return x            

        target = {var: target[var] for var in self.variables_to_slice}
        prediction = {var: prediction[var] for var in self.variables_to_slice}

        target = {k: apply_mask(v, k) for k, v in target.items()}
        prediction = {k: apply_mask(v, k) for k, v in prediction.items()}

        errors = tree_map(jnp.subtract, prediction, target)
        transformed_errors = self.transform(errors, target)
        squared_transformed_errors = tree_map(jnp.square, transformed_errors)

        # apply pressure weights
        def apply_pressure_weights(x, var_name):
            if var_name in self.pressure_weights and x.ndim == 4:
                pw = self.pressure_weights[var_name]  # shape (levels,)
                pw = pw.reshape((1, x.shape[1], 1, 1))
                x = x * pw
            return x

        squared_transformed_errors = {
            k: apply_pressure_weights(v, k) 
            for k, v in squared_transformed_errors.items()
        }

        def masked_mean_rmse(x, var_name):
            if var_name not in self.variables_to_slice:
                raise ValueError(f"Variable '{var_name}' not in variables to slice")
            n_points = jnp.sum(region_mask) * jnp.prod(jnp.array(x.shape[:-2]))
            return jnp.sqrt(jnp.sum(x) / n_points)

        return {k: masked_mean_rmse(v, k) for k, v in squared_transformed_errors.items()}
    
    def _create_region_mask(self, coords):
        full_latitudes = coords.horizontal.latitudes  # (lat,)
        full_longitudes = coords.horizontal.longitudes  # (lon,)

        lat_min = self.lat_bounds[0]
        lat_max = self.lat_bounds[1]
        lon_min = self.lon_bounds[0]
        lon_max = self.lon_bounds[1]

        lat_mask = (full_latitudes >= lat_min) & (full_latitudes <= lat_max)
        lon_mask = (full_longitudes >= lon_min) & (full_longitudes <= lon_max)

        region_mask = np.outer(lon_mask, lat_mask).astype(float)  # shape (lon, lat)
        return region_mask


def compute_loss(
    model,
    initial_state,
    target,
    forcings,
    rng_key,
    num_outer_steps,
    num_inner_steps,
    timedelta,
    lat_bounds,
    lon_bounds
):
    # unroll the model for forecasting
    _, prediction_trajectory = model.unroll(
        state = initial_state,
        forcings = forcings,
        steps=num_outer_steps,
        timedelta=timedelta,
        start_with_input=True,
    )

    # compute stats for rescaling
    def compute_stats(x):
        return {'mean': jnp.mean(x), 'std': jnp.std(x) + 1e-8}

    variables_to_normalize = {
        'temperature': initial_state['temperature'],
        'geopotential': initial_state['geopotential'],
        'specific_cloud_ice_water_content': initial_state['specific_cloud_ice_water_content'],
        'specific_cloud_liquid_water_content': initial_state['specific_cloud_liquid_water_content'],
        'specific_humidity': initial_state['specific_humidity'],
        'u_component_of_wind': initial_state['u_component_of_wind'],
        'v_component_of_wind': initial_state['v_component_of_wind'],
        'P_minus_E_cumulative': initial_state['P_minus_E_cumulative'] 
    }
    input_stats = jax.tree_util.tree_map(compute_stats, variables_to_normalize)

    trajectory_spec = metrics_util.TrajectorySpec(
        trajectory_length=num_outer_steps,
        max_trajectory_length=num_outer_steps,
        steps_per_save=num_inner_steps,
        coords=model.model_coords,
        data_coords=model.data_coords,
    )

    importance_weights = {
        'temperature': 1.0,
        'geopotential': 1.0,
        'specific_cloud_ice_water_content': 1.0,
        'specific_cloud_liquid_water_content': 1.0,
        'specific_humidity': 1.0,
        'u_component_of_wind': 1.0,
        'v_component_of_wind': 1.0,
        'P_minus_E_cumulative': 1.0 
    }

    num_levels = 37  
    level_array = jnp.arange(num_levels)
    level_weights = 1.0 / (1.0 + level_array)
    level_weights = level_weights / jnp.sum(level_weights)

    pressure_weight_dict = {
        'temperature': level_weights,
        'geopotential': level_weights,
        'specific_cloud_ice_water_content': level_weights,
        'specific_cloud_liquid_water_content': level_weights,
        'specific_humidity': level_weights,
        'u_component_of_wind': level_weights,
        'v_component_of_wind': level_weights
    }

    components = [
        linear_transforms.LegacyTimeRescaling,
        lambda *args, **kwargs: linear_transforms.PerVariableRescaling(
            *args, 
            weights={k: 1/v['std']**2 for k, v in input_stats.items()}, 
            **kwargs
        ),
        lambda *args, **kwargs: linear_transforms.PerVariableRescaling(
            *args, 
            weights=importance_weights, 
            **kwargs
        )
    ]

    loss_fn = RegionalLoss(
        trajectory_spec=trajectory_spec,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        components=components,
        variables_to_slice=[
            'temperature', 'geopotential', 
            'u_component_of_wind', 'v_component_of_wind', 
            'specific_humidity', 'specific_cloud_liquid_water_content', 
            'specific_cloud_ice_water_content','P_minus_E_cumulative'
        ],
        pressure_weights=pressure_weight_dict,
        is_nodal=True,
        is_encoded=False,
    )

    def subset_trajectory(trajectory: Dict[str, jnp.ndarray], last_n: int = 1) -> Dict[str, jnp.ndarray]:
        return {var_name: data_array[-last_n:] for var_name, data_array in trajectory.items()}

    # only compare last time step
    prediction_trajectory = subset_trajectory(prediction_trajectory, last_n=1)
    target = subset_trajectory(target, last_n=1)

    loss_dict = loss_fn.evaluate_per_variable(prediction_trajectory, target)
    loss_sum = sum(loss_dict.values())
    return loss_sum

def freeze_non_decoder_params(model, updates):
    total_params = 0
    unfrozen_params = 0

    # Flatten param updates for inspection
    flat_updates, tree_def = jax.tree_util.tree_flatten(updates)

    # We will store the updated results in a list, then unflatten at the end
    new_updates = []

    for update in flat_updates:
        total_params += jnp.size(update)

    def is_decoder_param(path_str):
        return 'dimensional_learned_primitive_to_weatherbench_decoder' in path_str

    # We need to track param paths along with updates:
    #   jax.tree_util.tree_leaves_with_path gives us (path, leaf).
    leaves_with_paths = jax.tree_util.tree_leaves_with_path(updates)

    # Build new updates with freezing logic
    for (path, old_u) in leaves_with_paths:
        # path is a tuple of keys/indices describing where we are in the tree
        path_str = '/'.join(str(x) for x in path)
        if is_decoder_param(path_str):
            new_updates.append(old_u)  # allow updates
            unfrozen_params += jnp.size(old_u)
        else:
            new_updates.append(jnp.zeros_like(old_u))  # freeze

    # Re-tree-ify new updates
    frozen_updates = jax.tree_util.tree_unflatten(tree_def, new_updates)

    pct_unfrozen = unfrozen_params / total_params if total_params > 0 else 0.0
    return frozen_updates, pct_unfrozen

def find_decoder_params(model):
    '''Identify decoder parameters and print them'''
    for path, param in model.params.items():
        if 'decode' in str(path):
            print(path)

def count_decoder_parameters(model):
    '''Count the number of parameters in the decoder that can be retrained.'''
    retrainable_params = 0
    for path, param in jax.tree_util.tree_leaves_with_path(model.params):
        if 'dimensional_learned_primitive_to_weatherbench_decoder' in str(path):
            retrainable_params += jnp.size(param)
    print(retrainable_params)

def count_total_parameters(model):
    num_params = 0
    for path, param in jax.tree_util.tree_leaves_with_path(model.params):
        num_params += jnp.size(param)
    print(num_params)


def pull_and_regrid_era5(
    model, 
    era5_path, 
    start_date, 
    end_date, 
    num_inner_steps, 
    output_path, 
    save = False
):
    start_date_short = start_date.replace('-', '')
    end_date_short = end_date.replace('-', '')
    filename = f'eval_era5_{start_date_short}_{end_date_short}.zarr'
    file_path = os.path.join(output_path, filename)

    if os.path.exists(file_path):
        print(f'File {filename} already exists. Loading it instead of re-evaluating.')
        return xarray.open_zarr(file_path)

    full_era5 = xarray.open_zarr(gcs.get_mapper(era5_path), chunks=None)

    era5_vars_to_keep = (
        model.input_variables 
        + model.forcing_variables 
        + ['evaporation', 'total_precipitation']
    )
    
    timestep = 24 // num_inner_steps

    sliced_era5 = (
        full_era5[era5_vars_to_keep]
        .pipe(
            xarray_utils.selective_temporal_shift,
            variables=model.forcing_variables,
            time_shift='24 hours',
        )
        .sel(time=slice(start_date, end_date, timestep))
        .compute()
    )

    # Convert total precipitation & evaporation to kg/m^2
    sliced_era5["total_precipitation"] = sliced_era5["total_precipitation"] * 1000.0
    sliced_era5["evaporation"] = sliced_era5["evaporation"] * 1000.0

    # Multiply by the time step to approximate cumulative effect
    sliced_era5["total_precipitation"] = sliced_era5["total_precipitation"] * timestep
    sliced_era5["evaporation"] = sliced_era5["evaporation"] * timestep

    # cumulative precipitation minus evaporation
    sliced_era5['P_minus_E_cumulative'] = (
        sliced_era5['total_precipitation'].cumsum('time')
        - sliced_era5['evaporation'].cumsum('time')
    )

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

    if save:
        eval_era5.to_zarr(f"{output_path}/{filename}")
    
    return eval_era5

# split the dataset into train/val
def prepare_train_val_data(
    model,
    era5_path: str,
    start_date: str,
    end_date: str,
    num_inner_steps: int,
    num_outer_steps: int,
    output_path: str,
    train_fraction: float = 0.8,
    save: bool = False
):
    """
    1) Pull and regrid ERA5 data with pull_and_regrid_era5.
    2) Split by time into train and validation sets.
    3) Convert each list of (initial_state, target_trajectory, forcings) touples.
    """
    # pull regrid and load in era5 data at selected time range, time resolution,
    # and regrid to model grid. Only includes variables needed for model
    full_era5 = pull_and_regrid_era5(
        model=model,
        era5_path=era5_path,
        start_date=start_date,
        end_date=end_date,
        num_inner_steps=num_inner_steps,
        output_path=output_path,
        save=save
    )

    # Need to split the full data into train and validation sets while still 
    # keeping the time clusters together for forecasts to make sense
    window_length = num_outer_steps + 1
    times = full_era5.time
    n_times = times.size

    cluster_starts = np.arange(0, n_times - window_length + 1, window_length)

    # Randomize cluster order to do a train/val split
    rng_key = jax.random.PRNGKey(2304)
    cluster_starts = jax.random.permutation(rng_key, cluster_starts)

    n_clusters = len(cluster_starts)
    n_train = int(train_fraction * n_clusters)

    train_indices = cluster_starts[:n_train]
    val_indices = cluster_starts[n_train:] 

    train_list = []
    val_list = []

    # Helper function to transform an xarray dataset chunk into
    # (initial_state, target_trajectory, forcings) needed for training.
    def build_data_dict(xds: xarray.Dataset):

        evaluation_vars = [
            'temperature', 'geopotential', 
            'specific_cloud_ice_water_content', 
            'specific_cloud_liquid_water_content', 
            'specific_humidity', 'u_component_of_wind', 
            'v_component_of_wind', 'P_minus_E_cumulative'
        ]

        # Convert to model dictionary
        data_dict = model._data_from_xarray(xds, list(xds.data_vars))

        # The "initial_state" can come from the first time
        init_data = xds.isel(time=0)
        inputs = model.inputs_from_xarray(init_data)
        forcings = model.forcings_from_xarray(init_data)
        initial_state = model.encode(inputs, forcings, jax.random.PRNGKey(0))  

        target_trajectory = data_dict
        all_forcings = model.forcings_from_xarray(xds)

        return initial_state, target_trajectory, all_forcings
    
    for start_idx in train_indices:
        chunk_ds = full_era5.isel(time=slice(start_idx, start_idx + window_length))
        train_list.append(build_data_dict(chunk_ds))

    # Build validation list
    for start_idx in val_indices:
        chunk_ds = full_era5.isel(time=slice(start_idx, start_idx + window_length))
        val_list.append(build_data_dict(chunk_ds))

    # Each element of train_list or val_list is a tuple:
    #   (initial_state, target_trajectory, forcings) for that time cluster

    return train_list, val_list 

if __name__ == "__main__":
    # Set random key
    rng_key = jax.random.PRNGKey(854)

    # 1) Load a simple demo model
    ckpt = neuralgcm.demo.load_checkpoint_tl63_stochastic()

    # P_minus_E code from: https://github.com/neuralgcm/neuralgcm/issues/12
    new_inputs_to_units_mapping = {
      'u': 'meter / second',
      'v': 'meter / second',
      't': 'kelvin',
      'z': 'm**2 s**-2',
      'sim_time': 'dimensionless',
      'tracers': {
          'specific_humidity': 'dimensionless',
          'specific_cloud_liquid_water_content': 'dimensionless',
          'specific_cloud_ice_water_content': 'dimensionless',
      },
      'diagnostics': {'P_minus_E_cumulative': 'kg / (meter**2)'}
    }

    new_model_config_str = '\n'.join([
        ckpt['model_config_str'],
        f'DimensionalLearnedPrimitiveToWeatherbenchDecoder.inputs_to_units_mapping = {new_inputs_to_units_mapping}',
        'DimensionalLearnedPrimitiveToWeatherbenchDecoder.diagnostics_module = @NodalModelDiagnosticsDecoder',
        'StochasticPhysicsParameterizationStep.diagnostics_module = @PrecipitationMinusEvaporationDiagnostics',
        'PrecipitationMinusEvaporationDiagnostics.method = "cumulative"',
        'PrecipitationMinusEvaporationDiagnostics.moisture_species = ("specific_humidity", "specific_cloud_liquid_water_content", "specific_cloud_ice_water_content")'
    ])
    ckpt['model_config_str'] = new_model_config_str
    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)    

    # Example parameters
    start_date = '2020-02-01'
    num_days = 5 
    num_inner_steps = 1
    # add 1 so that we can create clusters that have a one day buffer
    end_date_dt = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=num_days + 1) 
    end_date = end_date_dt.strftime('%Y-%m-%d')
    # OH XX trying to pull for a full month for now
    end_date = '2020-03-01'
    num_outer_steps = num_days * num_inner_steps
    timedelta_h = np.timedelta64(24, 'h') // num_inner_steps

    # Region of interest in degrees, convert to radians:
    lat_bounds_deg = (20, 60)
    lon_bounds_deg = (200, 300)
    lat_bounds = (np.deg2rad(lat_bounds_deg[0]), np.deg2rad(lat_bounds_deg[1]))
    lon_bounds = (np.deg2rad(lon_bounds_deg[0]), np.deg2rad(lon_bounds_deg[1]))

    era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
    output_path = dir['processed']

    # 2) Prepare training and validation data
    train_list, val_list = prepare_train_val_data(
        model=model,
        era5_path=era5_path,
        start_date=start_date,
        end_date=end_date,
        num_inner_steps=num_inner_steps,
        num_outer_steps=num_outer_steps,
        output_path=output_path,
        train_fraction=0.8,
        save=True
    )

    print("Data prepared.")

    print(type(train_list))
    print(len(train_list))
    exit()

    # for now only use the first element of train_list and val_list
    train_init, train_target_trajectory, train_forcings = train_list[0]
    print(f"train init: {train_init['temperature'].shape}")
    val_init, val_target_trajectory, val_forcings = val_list[0]
    print(f" val init: {val_init['temperature'].shape}")
    exit()

    # 3) Set up optimizer
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(model)

    # 4) JIT-compile compute_loss
    compute_loss_jit = jax.jit(compute_loss, static_argnums=(5, 6, 7, 8, 9))

    # 5) Training loop
    num_iterations = 3
    for i in range(num_iterations):
        # Training step
        train_loss, grads = jax.value_and_grad(compute_loss_jit)(
            model,
            train_init,
            train_target_trajectory,
            train_forcings,
            rng_key,
            num_outer_steps,
            num_inner_steps,
            timedelta_h,
            lat_bounds,
            lon_bounds
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        frozen_updates, pct_unfrozen = freeze_non_decoder_params(model, updates)
        model = optax.apply_updates(model, frozen_updates)

        # Validation loss
        val_loss = compute_loss_jit(
            model,
            val_init,
            val_target_trajectory,
            val_forcings,
            rng_key,
            num_outer_steps,
            num_inner_steps,
            timedelta_h,
            lat_bounds,
            lon_bounds
        )

        print(f"Iteration {i+1}, train_loss = {train_loss.item():.6f}, val_loss = {val_loss.item():.6f}")

