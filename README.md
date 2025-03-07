Files to create forecasts using ECMWF and ai-models packages in run_ECMWF_forecasts
NOTE: no longer actively using
- On the server download-forecasts.py should be called with a slurm job using ai_weather_models_download_job.sh
- download_forecasts.py: download IFS, pangu, fourcastnet, era5 
- combine_forecasts.py: combines output of previous script into single .nc file
- compare_forecasts.py: creates rmse plots and maps for different regions 
- can be run together using local_forecast_compare.py
- viz_forecast.py: is currenlty unused old vizualization script that predated compare_forecasts.py

Files needed to run weatherbench2 package in run_weatherbench2 folder
NOTE: no longer actively using
- weatherbench2_job.sh: to be run on server to submit slurm job to run weatherbench2 script evaluate.py
- the outputs can then be read in and combined using weatherbench2_analysis.py
- there are other misc weatherbench2 scripts included for 

Files used for retraining neuralgcm decoder in neuralGCM_retraining
NOTE: not actively using
- fine_tune_neuralGCM_decoder.py: main script for creating and testing custom loss 
- it calls from local_neuralGCM which contains the public neuralGCM code. We only modify scripts in the reference_code folder
- neuralGCM_inference.py: takes inference demo code from neuralGCM documentation. Independent of previous two

Files used for fine-tuning models using weatherbench2 in finetuning
- finetune.py: main script that trains small MLP to correct model outputs
- finetune_job.sh: bash script to run simple finetuning job
- finetune_evaluation.py: takes in original and corrected .zarr forecasts created by finetune.py and makes plots showing comparison. 

metar_scraping.py and metar_xp.py scrape weatherstation data from NOAA and then do some basic eda