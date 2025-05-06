= Forecast Fine‑tuning Report

// -------------------------------------------------------------------
// User‑adjustable parameters
// -------------------------------------------------------------------
#let fig_root = "../figures/finetuning"         // parent folder for all figures
#let model     = "pangu"                        // model name sub‑folder
#let regions   = ("usa_south", "amazon", "india", "pakistan")
#let lead_times = (24, 72, 168)                  // hours ahead
#let subregion  = "10x10"                       // only plot 10×10 results for now
#let var        = "2m_temperature"              // prediction variable
#let train_vars = "2m_temperature"              // (joined by "_" in file names)
#let mlp_tag    = "mlp512x5"                    // hidden‑dim × layers tag used in file names

#let fig_dir = fig_root + "/" + model          // convenience path

// Small wrapper for quickly inserting (and scaling) images
#let pic(path) = image(path, width: 100%)        // full‑width; adjust as desired

// -------------------------------------------------------------------
// Combined comparison plots (across all regions & lead times)
// -------------------------------------------------------------------
= Combined Results

// Overall MSE by lead‑time/region
#pic(fig_dir + "/comparison/mse_comparison_" + model + "_trained_with_" + train_vars + "_output" + var + "_" + mlp_tag + ".png")

// Sub‑region improvement plot
#pic(fig_dir + "/comparison/subregion_mse_improvement_" + train_vars + "_" + var + "_" + mlp_tag + ".png")

// -------------------------------------------------------------------
// Region‑specific sections (only 10×10 sub‑region)
// -------------------------------------------------------------------
#for region in regions {
  == Region #region

  /// Monthly MSE time‑series for each lead time
  #for lt in lead_times {
    === Time‑series · #lt h

    #pic(fig_dir + "/" + region + "/time_series/mse_time_series_" + subregion + "_" + var + "_trained_with_" + train_vars + "_" + lt.str() + "h.png")
  }

  /// Spatial maps (raw values & MSE) for each lead time
  #let map_tags = (
    "raw_map_original",
    "raw_map_corrected",
    "raw_map_difference",
    "mse_map_original",
    "mse_map_corrected",
    "mse_map_difference",
  )

  #for lt in lead_times {
    === Maps · #lt h

    #for tag in map_tags {
      #pic(fig_dir + "/" + region + "/maps/" + tag + "_" + subregion + "_" + var + "_trained_with_" + train_vars + "_" + lt.str() + "h.png")
    }
  }
}

// -------------------------------------------------------------------
// Notes
// -------------------------------------------------------------------
// • Adjust `fig_root` if this Typst file is not located next to the Python
//   project root.
// • Add or remove regions/lead times by editing the parameter lists above.
// • Typst loops (`#for`) keep the document short and automatically expand
//   when new plots are generated.
// • `pic` helper sets every figure to full text width; tweak the width or
//   wrap inside `figure()` if you need captions.
