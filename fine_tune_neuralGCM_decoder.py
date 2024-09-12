import jax
import optax
import neuralgcm

@jax.jit
def compute_loss(model, inputs, forcings, rng):
    encoded = model.encode(inputs, forcings, rng_key=rng)
    predictions = model.decode(encoded, forcings)

    # inputs_xarray = model.data_to_xarray(inputs, None)
    # predictions_xarray = model.data_to_xarray(predictions, None)

    # loss = abs(inputs_xarray['temperature'] - predictions_xarray['temperature']).mean()
    # return loss

    return abs(inputs['temperature'] - predictions['temperature']).mean()

checkpoint = neuralgcm.demo.load_checkpoint_tl63_stochastic()
initial_model = neuralgcm.PressureLevelModel.from_checkpoint(checkpoint)

ds = neuralgcm.demo.load_data(initial_model.data_coords)
inputs, forcings = initial_model.data_from_xarray(ds.isel(time=0))

print(compute_loss(initial_model, inputs, forcings, jax.random.key(0)))
# Array(6.2256584, dtype=float32)

optimizer = optax.adam(1e-3)

model = initial_model
opt_state = optimizer.init(initial_model)

for i in range(5):
    loss, grads = jax.value_and_grad(compute_loss)(model, inputs, forcings, jax.random.key(0))
    updates, opt_state = optimizer.update(grads, opt_state)
    model = optax.apply_updates(model, updates)
    print(f'{i=}, {loss=}')
# i=0, loss=Array(6.2256584, dtype=float32)
# i=1, loss=Array(4.670498, dtype=float32)
# i=2, loss=Array(3.855668, dtype=float32)
# i=3, loss=Array(3.5485578, dtype=float32)
# i=4, loss=Array(3.4335625, dtype=float32)