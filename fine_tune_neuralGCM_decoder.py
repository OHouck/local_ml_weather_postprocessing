import jax
import jax.numpy as jnp
import optax
import neuralgcm

# @jax.jit
# def compute_loss(model, inputs, forcings, rng):
#     encoded = model.encode(inputs, forcings, rng_key=rng)
#     predictions = model.decode(encoded, forcings)

#     inputs_xarray = model.data_to_xarray(inputs, None)
#     predictions_xarray = model.data_to_xarray(predictions, None)

#     loss = abs(inputs_xarray['temperature'] - predictions_xarray['temperature']).mean()
#     return loss

#     # return abs(inputs['temperature'] - predictions['temperature']).mean()

def compute_loss(model, inputs, forcings, rng, lat_bounds = (0, 180), lon_bounds = (0, 360)):
    encoded = model.encode(inputs, forcings, rng_key=rng)
    predictions = model.decode(encoded, forcings)

    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    # Convert geographic coordinates to array indices
    # Assuming 128 longitude steps from 0 to 360, and 64 latitude steps from -90 to 90
    lon_idx_min = int(lon_min * 128 / 360)
    lon_idx_max = int(lon_max * 128 / 360)
    lat_idx_min = int((lat_min + 90) * 64 / 180)
    lat_idx_max = int((lat_max + 90) * 64 / 180)

    # Calculate slice sizes
    lon_slice_size = lon_idx_max - lon_idx_min
    lat_slice_size = lat_idx_max - lat_idx_min


    # Initialize total loss
    total_loss = 0.0
    num_variables = 0

    # Iterate over all variables in inputs
    for var_name in inputs.keys():
        if inputs[var_name].ndim == 3:  # Only process 3D variables
            # Apply the geographic mask to both inputs and predictions using dynamic_slice
            inputs_masked = jax.lax.dynamic_slice(
                inputs[var_name], 
                (0, lon_idx_min, lat_idx_min), 
                (37, lon_slice_size, lat_slice_size)
            )
            predictions_masked = jax.lax.dynamic_slice(
                predictions[var_name], 
                (0, lon_idx_min, lat_idx_min), 
                (37, lon_slice_size, lat_slice_size)
            )

            # Mean absolute error
            # var_loss = jnp.mean(jnp.abs(inputs_masked - predictions_masked))
            # total_loss += var_loss
            # num_variables += 1

            # Normalized mean absolute error
            var_range = jnp.max(inputs_masked) - jnp.min(inputs_masked)
            var_loss = jnp.mean(jnp.abs(inputs_masked - predictions_masked) / var_range)
            total_loss += var_loss
            num_variables += 1

    # Calculate the average loss across all variables
    average_loss = total_loss / num_variables if num_variables > 0 else 0.0
    return average_loss

# JIT-compile the function
compute_loss_jit = jax.jit(compute_loss, static_argnums=(0, 4, 5))

# this isn't working
def freeze_non_decoder_params(model, updates):
    def is_decoder_param(path, _):
        return 'dimensional_learned_primitive_to_weatherbench_decoder' in '/'.join(path)

    def maybe_freeze(path, update):
        print(f'path: {path}')
        return update if is_decoder_param(path, update) else jnp.zeros_like(update)

    frozen_updates = jax.tree_util.tree_map_with_path(maybe_freeze, updates)
    return frozen_updates

checkpoint = neuralgcm.demo.load_checkpoint_tl63_stochastic()
initial_model = neuralgcm.PressureLevelModel.from_checkpoint(checkpoint)

ds = neuralgcm.demo.load_data(initial_model.data_coords)
inputs, forcings = initial_model.data_from_xarray(ds.isel(time=0))


def print_param_tree(params, prefix=''):
    for key, value in params.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_param_tree(value, prefix + '  ')
        else:
            print(f"{prefix}{key}: {value.shape}")

# print_param_tree(initial_model.params)

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
    frozen_updates = freeze_non_decoder_params(model, updates)
    model = optax.apply_updates(model, frozen_updates)
    print(f'{i=}, {loss=}')
# i=0, loss=Array(6.2256584, dtype=float32)
# i=1, loss=Array(4.670498, dtype=float32)
# i=2, loss=Array(3.855668, dtype=float32)
# i=3, loss=Array(3.5485578, dtype=float32)
# i=4, loss=Array(3.4335625, dtype=float32)