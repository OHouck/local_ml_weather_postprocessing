Files to create forecasts using ECMWF and ai-models packages
- On the server download-forecasts.py should be called with a slurm job using ai_weather_models_download_job.sh
- download_forecasts.py: download IFS, pangu, fourcastnet, era5 
- combine_forecasts.py: combines output of previous script into single .nc file
- compare_forecasts.py: creates rmse plots and maps for different regions 
- can be run together using local_forecast_compare.py
- viz_forecast.py: is currenlty unused old vizualization script that predated compare_forecasts.py

Files needed to create weatherbench output
- weatherbench2_job.sh: to be run on server to submit slurm job to run weatherbench2 script evaluate.py
- the outputs can then be read in and combined using weatherbench2_analysis.py
- there are other misc weatherbench2 scripts included for 

Inference using neuralgcm
- neuralGCM_inference.py: takes inference demo code from neuralGCM documentation.

Files used for retraining neuralgcm decoder
- fine_tune_neuralGCM_decoder.py: main script for creating and testing custom loss 
- it calls from local_neuralGCM which contains the public neuralGCM code. We only modify scripts in the reference_code folder

Files used for fine-tuning models using weatherbench2