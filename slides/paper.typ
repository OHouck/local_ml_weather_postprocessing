// Frontmatter //
#set page( margin: (left: 25mm, right: 25mm, top: 25mm, bottom: 30mm),
    numbering: "1", number-align: center)
#set text(font: "New Computer Modern", lang: "en")
#set cite(style: "chicago-author-date")
#set heading(numbering: "1.1")
#import "@preview/subpar:0.2.2" // for making subfigures

// TITLE //
#line(length: 100%, stroke: 2pt)
#pad(top: 4pt, bottom: 5pt, align(center)[#text(weight: "bold", 1.8em, smallcaps[Tailoring machine learning weather predictions for local impacts])])
#line(length: 100%, stroke: 2pt)

// AUTHORS--- need to pick your own number of rows and columns //
#pad(x: 1em, top: 5pt, align(center)[#grid( columns: (auto, auto), gutter: 30pt, align: center,

[*Ozma Houck*\
Harris School of Public Policy\
University of Chicago\
`ohouck@uchicago.edu`],

[#pad(bottom: -20pt, grid(columns:(40%,4%), gutter:1%, [*James Franke*], [#link("http://orcid.org/0000-0001-8598-750X")[#image("figs/orcid.svg")]] )) \
Department of the Geophysical Sciences\
University of Chicago\
`jfranke@uchicago.edu`],  

)])

#pad(top:10pt, bottom:5pt, align(center)[#datetime.today().display()])
#set par.line(numbering: "1")

#pad(x: 3em, bottom: 0.4em, align(center)[#heading(outlined: false, numbering: none, text(0.85em, smallcaps[Abstract]),)
  #set par(justify: true)
  #set text(hyphenate: false)
  Machine learning (ML) based weather models are improving quickly but are expensive to train for anything but general use cases. We demonstrate the feasibility of improving the performance of these weather forecasts for specific, local use cases using minimal compute power. 
  We show ML weather models can be post-processed in the same way as traditional numerical weather prediction (NWP) based forecasts and discuss which modeling decisions lead to the best error prediction. By applying this method regionally across total land surface, we show that there is substantial variation how much post-processing increases forecast skill. This serves as a novel diagnostic of local vs larger-scale impacts on weather variability (or predictability). We find that our post-processing method performs relatively the best for areas near the equator and areas with large changes in elevation. 
])
  
// Main text //
= Introduction

Weather forecasts are a critical input to decision-making across a wide range of economic sectors. 
Accurate predictions of temperature, precipitation, and wind allow farmers to optimize planting and irrigation decisions, energy utilities to manage intermittent renewable generation, and emergency managers to issue timely warnings for extreme events. 
The economic value of improved forecasts has been documented extensively: experimental evidence from India demonstrates substantial welfare gains when farmers receive accurate weather information @burligValueForecastsExperimental2024, mortality studies show that forecast errors have meaningful impacts on health outcomes@shraderFatalErrorsMortality2023, analyses of electricity markets reveal how forecast uncertainty affects renewable energy integration @weberIntermittencyUncertaintyImpacts2024, and how forecast inaccuracy distorts local labor markets @songValueWeatherForecasts2024.

Regional meteorological offices in wealthy countries synthesize global forecasts models along with their own methods and expert opinions to create forecasts for their service areas. People living in areas with fewer resources often do not have access to these tailored forecasts and would benefit from affordable and easy to implementable forecast post-processing methods

Over the past fifty years, steady advances in computational resources and atmospheric science have driven substantial improvements in physics-based Numerical Weather Prediction (NWP) models. 
These models discretive the atmosphere onto three-dimensional grids and numerically solve the partial differential equations governing atmospheric dynamics. 
More recently, the rapid development of machine learning methods has led to the creation of data-driven weather models that learn to predict atmospheric states directly from historical data. A proliferation of such models---including FourCastNet @pathakFourCastNetGlobalDatadriven2022, Pangu-Weather @biAccurateMediumrangeGlobal2023, NeuralGCM @kochkovNeuralGeneralCirculation2024, GenCast @priceProbabilisticWeatherForecasting2024, AIFS @langAIFSECMWFsDatadriven2024, and Aurora @bodnarFoundationModelEarth2025 ---has demonstrated that neural networks can match or exceed the average accuracy of traditional NWP systems on standard benchmarks@raspWeatherBench2Benchmark2024a @olivettiDatadrivenModelsBeat2024.


A critical distinction between NWP and machine learning (ML) approaches lies in computational cost. Generating a global weather forecast using NWP requires supercomputing resources. In contrast, once trained, ML models can produce forecasts in minutes on a personal laptop. This reduction in inference cost opens new possibilities for forecast customization. #text(red)[democratization?] Rather than relying solely on forecasts distributed by major weather agencies, users with specific local needs could potentially generate their own predictions tailored to their particular applications. 

While ML forecasts are cheap to generate, training a global ML weather model from scratch remains prohibitively expensive for almost all end users---requiring thousands of GPU-days and tens of terabytes of training data. 
Fine-tuning existing models is possible but still computationally demanding, and the process is often model-specific and technically challenging @xiaFineTunedPanguWeather2025.
An alternative approach is to use a separate model to correct systematic biases in an existing forecast system. This strategy has a long history in operational meteorology and offers several advantages for our purposes. We train light weight neural networks to post-process existing forecasts and shrink forecast error. This method can be applied to either NWP or ML based forecast models and allows the user to set customized target objectives.

Statistical post-processing of weather forecasts has a long history in operational meteorology @vannitsemStatisticalPostprocessingWeather2021. Traditional approaches include the linear regression based Model Output Statistics (MOS) which was later extended to be used with ensemble forecasts @glahnUseModelOutput1972 @gneitingCalibratedProbabilisticForecasting2005. Before the successful development of fully ML based weather models, neural networks were shown as a way of implementing this method of ensemble post processing @raspNeuralNetworksPostprocessing2018.

Multi-layer perceptron and Convolutional U-Net neural network architectures are able to improve 2m temperature and 10m wind speed predictions for both NWP and ML based forecasting models @hanDeepLearningMethod2021 @bremnesEvaluationForecastsGlobal2024 @trottaPostprocessingImprovesAccuracy2025. 

Most recent post-processing studies are case studies focused on particular regions, primarily in Western Europe where dense observation networks are available. #footnote[A notable exception is the ecPoint model developed by ECMWF to improve precipitation forecasts @hewsonLowcostPostprocessingTechnique2021a @trottaRainForestsMachineLearning2024.] Given the differences in global atmospheric processes, different areas may have different post-processing needs. Forecast accuracy is positively correlated with country GDP @linsenmeierGlobalInequalitiesWeather2023a. The relative accuracy of short-term versus long-term forecasts varies systematically between tropical and extratropical regions @keaneMidLatitudeTropicalScales2025. 

For a nine day 2m temperature forecast, we show mean improvement of XX for Pangu and XX for IFS. For a 10m wind speed forecast, the mean improvement for Pangu and IFS are XX and XX respectively. Furthermore, we document spatial heterogeneity in forecast improvement and argue that this stems from differences in how much of the residual forecast error is driven by local vs. global weather patterns. When training our models, we compare the performance of different choices for model architectures input data. 


= Results 


= Data and Methods

We focus on post-processing one NWP model and one ML model. For the NWP model we choose to use ECMWF's IFS HRES model; it is one of the most widely used global NWP models. Following @raspNeuralNetworksPostprocessing2018, we use IFS HRES t=0 analysis data to be the ground truth for IFS forecasts. For the ML model we used Pangu-Weather introduced in @biAccurateMediumrangeGlobal2023. Pangu has been shown to perform similarly to other ML weather models and has the advantage of having 4 full years of forecast data available for download on weatherbench (@raspNeuralNetworksPostprocessing2018, @olivettiDatadrivenModelsBeat2024). We additionally use ERA5 to compute static fields characterizing surface orography at quarter-degree resolution.

For both Pangu and IFS forecasts, we use forecasts at 0.25° resolution spanning 2018--2022 initialized daily at 00:00 UTC. We focus on lead times of 1, 5, and 9 days (24, 120, and 216 hours), capturing both short-range and extended-range forecast skill. We use 2018--2021 for training and hyperparameter selection, and 2022 for out-of-sample evaluation.

== Post-Processing Framework

We train neural networks to predict the bias in existing weather forecasts. We follow the post-processing approach where the corrected forecast is the sum of the original forecast and a learned correction term. (XX footnote for paper that found this was better) 
// XX equation part could be moved to appendix, not sure?
Formally, if $hat(y)_(t,h)$ denotes the raw forecast for time $t$ at lead time $h$, and $f_theta$ is our neural network, the corrected forecast is:
$ tilde(y)_(t,h) = hat(y)_(t,h) + f_theta (bold(x)_(t,h), h, d_t) $
where $bold(x)_(t,h)$ contains the forecast fields used as predictors, $h$ is the lead time, and $d_t$ is the day of year.
Crucially, our models use only the weather forecasts themselves as inputs---we do not incorporate additional observational data or auxiliary predictors beyond temporal encodings.

== Model Architectures

We design and evaluate two neural network architectures: a multilayer perceptron (MLP) and a U-Net. Both architectures incorporate lead time and seasonal information to enable training across multiple forecast horizons and to capture systematic seasonal biases.

*Temporal Encodings.* Following @bremnesEvaluationForecastsGlobal2024, we encode the day of year using sinusoidal features:
$ "doy"_sin = sin((2 pi dot d_t) / 365), quad "doy"_cos = cos((2 pi dot d_t) / 365) $
where $d_t$ is the day of year for forecast valid time $t$. These features are concatenated to the input representation in both architectures. Lead time is encoded via a learned embedding layer that maps discrete lead time indices to a continuous vector representation, which is similarly concatenated to the input.

*MLP Architecture.* The MLP consists of fully connected layers with ReLU activations and dropout regularization. Input forecast fields are flattened and concatenated with the temporal encodings before being passed through the network. The architecture depth and width are treated as hyperparameters.

*U-Net Architecture.* The U-Net preserves spatial structure through an encoder-decoder architecture with skip connections. The encoder applies convolutional blocks followed by max pooling, while the decoder uses transposed convolutions for upsampling. Each convolutional block consists of two sequences of 2D convolution, batch normalization, ReLU activation, and 2D dropout. Based on preliminary experiments, we cap the maximum number of channels at 128 to limit model complexity and avoid over fitting during training. Temporal encodings are broadcast spatially and concatenated as additional input channels.

*Training Procedure.* Models are trained to minimize mean squared error between corrected forecasts and ground truth observations using the Adam optimizer. To prevent overfitting, we use early stopping apply L2 weight decay and use dropout during training. The learning rate is adjusted dynamically using a reduce-on-plateau scheduler that decreases the learning rate when validation loss stagnates.

*Hyperparameter Optimization*. We conduct hyperparameter optimization separately for each variable and architecture combination using Bayesian optimization with the Tree-structured Parzen Estimator algorithm, implemented via the `hyperopt` package. Optimization is performed on a 6° × 6° region in central India using the training period data, with an 80/20 train/validation split.

//XX maybe this can be cut?
For the MLP, we optimize the number of hidden layers, nodes per layer, initial learning rate, batch size, weight decay, early stopping patience and improvement threshold, dropout rate, and lead time embedding dimension. For the U-Net, we optimize the base number of channels, initial learning rate, batch size, weight decay, early stopping parameters, dropout rate, and lead time embedding dimension. The optimal hyperparameters are then applied when training models for all regions.

// TODO: Add justification for why single-region hyperparameter tuning generalizes

== Global Results


// Currently just included 2m temperature results...
We find substantial heterogeneity in post-processing skill across geographic regions and forecast lead times. Figure (XX Global Map Fig) plots the RMSE % improvement in 2m temperature forecasts across global land surface at a 0.25 degree resolution. In the figure, each black 6x6 degree box denotes a separate post-processing model that was trained to correct the forecast for that specific region. We are able to improve forecast accuracy in almost all regions, but the magnitude of improvement varies greatly by pixel even within the same region. 

// I draw a cubit line of best fit. some non linearity with distance from equator not so much for sdor. Is this worth mentioning here?
// Should we include a map of SDOR values or will readers know have a good understanding of topography? 
// in results the results section, should we make claims of local vs large-scale sources of forecast error?
To investigate possible explanations for this variation, we first compare how forecast improvement changes with distance to the equator  Figure XX is a binned scatter plot showing that for both 2m temperature and 10m wind speed, the initial forecast skill deteriorates with distance from the equator and that this effect is most pronounced for longer lead times. Post-processing improvement also increases with lead time, but unlike the original forecast error, the improvements are the largest near the equator. For 2m temperature, the relationship between latitude and forecast improvement seems less linear. For nine day forecasts, we see larger improvements between 40 and 50 degrees. These bands include much of The United States, Canada, and Europe. #text(red)[Do we have any suggestions why this might be? we don't see raw forecast accuracy increase in these areas, why only forecast improvement?]

Next, we compare original forecast accuracy and the improvement of our post processing model to the standard deviation of orography (SDOR) in a pixel. Figure XX shows the relationship between forecast improvement and SDOR. While 2m temperature original forecast accuracy and improvement are relatively flat with for different amounts of elevation change, the accuracy of 10m wind speed forecasts decrease with SDOR but the RMSE improvement of our models increases with SDOR.


#figure(
  image("figs/pangu/lead_time_compare_binscatter_equator.png"),
  caption: [Binscatter by lead time equator distance]
)

#figure(
  image("figs/pangu/lead_time_compare_binscatter_sdor.png"),
  caption: [Binscatter by lead time sdor]
)


== Architecture and Domain Comparison

// Do we mention why we choose central india? I choose it initially since its an ag relevant region that is currently underserved by weather forecasts but that's kind of beyond the scope of the this paper
When creating our post processing model, we compare different model architectures and data input options in order to document possible tradeoffs between model performance and computational requirements. We compare two neural network architectures, a multilayer perceptron (MLP) and a U-Net convolutional network. Figure XX shows the RMSE improvement percent for different modeling choices. All models were trained to correct Pangu on the same 6x6 degree region in central India. Each model used its own set of optimal hyperparameters which were chosen by Bayesian optimization (details in Appendix XX). This allows models with more input parameters to also be larger. and find that for the small regional patches we consider (6° × 6°), the simpler MLP architecture performs as well as the more complex U-Net while requiring substantially fewer computational resources. On a 36GB M3 Max Macbook, and MLP with a single input variable takes XX time to train while a UNET taxes XX. In the physics based models, specific humidity and temperature are thought to predict temperature better than temperature alone (XX need a cite? is this correct?). However, when correcting for forecast bias, we find that including additional input variables such as bottom of atmosphere temperature and specific humidity do not improve the predictions of 2m temperature.

#figure(
  image("figs/pangu/arch_comparison_2m_temperature_india_6x6.png"),
  caption: [MLP with one variable is best]
)
// XX what is the correct way to write lat lon?
We also tested if increasing the training region domain would improve post processing skill. In principle, giving the model more data from a larger region might allow it to better correct forecast errors by learning more about about the local weather patterns. Figure XX plots the RMSE percent improvement for a central 6x6 degree patch as the geographic size of the training domain is increased #footnote[Repeating this analysis with UNETs instead of MLPs yields similar results]. We show the results across lead times and for two patches, one in Finland centered at (65, 29) and one in the Amazon Rainforest centered at (-5, 295). We choose these patches as candidates to represent high latitude (XX is there a name for weather patterns driven by larger atmospheric processes?) BLANK driven weather and equatorial convection dominated weather respectively. Because our post processing is at the level of 0.25 degree pixels, the physical size of these pixels shrinks closer to the poles. We thought that this might also affect how increasing the domain size affects training performance. 

Perhaps surprisingly, we find that increasing the training domain did not affect post processing performance in either region. Instead we notice a small degradation in forecast improvement. Similar to the general global results, we see greater improvements in longer forecast lead times. Additionally, the previously discussed non-linear effect between 9 day forecast improvement and latitude is supported by the large improvements seen across all domain in the 9 day Finland forecast.
#text(red)[This last part is still clunky. OH should think more about how to present these results.] 

#figure(
  image("figs/pangu/region_size_comparison_2m_temperature_mlp_pangu.png"),
  caption: [Effect of Expanding Training Region: 2m Temperature]
)

= Discussion

After training post-processing models across global land surface, we observed systematic differences in forecast improvement. We interpret these differences as being informative for how local an area's weather patterns are. The improvement from post-processing generally increases with lead time, as systematic biases accumulate in longer-range forecasts. Spatially, we find that post-processing yields the largest gains in tropical regions near the equator and in areas with significant elevation variability. This geographic pattern provides insight into the sources of forecast error. We hypothesize that regions where local post-processing is most effective are those where forecast errors are primarily driven by local weather patterns and surface characteristics that global models fail to capture, rather than by larger-scale atmospheric dynamics.
why do we think the region size doesnt help?
-> maybe network size but that will asymptote 

When experimenting with different training methods, we found that adding additional weather variables as model inputs did not improve prediction skill and increasing the complexity of the model also did not improve. This has practical implications for users seeking to implement post-processing with limited computing infrastructure. We view the models presented here as evidence of what can be done given limited compute power and we hope that future effort will be put into further improving upon our modeling framework.

We also found that increasing the training region domain did not improve the prediction skill in the target area. We found this same pattern in both MLP and UNET architectures. This could be from the fact that the MLP does not make explicit use of spatial relationships and the UNET model uses a fixed spatial context window.(XX finish)

While we believe that the performance of a model like ours can be further improved, forecast improvements of the magnitude presented here have the potentially to meaningfully impact the lives of many people. In the context of Chinese cities, @songValueWeatherForecasts2024 estimates that the 3.9% improvement in one-day maximum temperature forecasts between 2011-2015 generated a social benefit of 25.3 billion Yuan (4.03 billion USD) from the labor sector alone. @shraderFatalErrorsMortality2023 estimate that a 50% improvements in forecast accuracy in would lead 2,200 fewer people dieing from extreme heat in the United States. #text(red)[OH get example from energy sector]

We applied our correction to raw global forecast outputs. Most people in wealthy countries receive weather forecasts created by regional meteorological office. In this paper, we do not benchmark our forecast improvement results to those achieved by local meteorologist. Instead, we believe flexible and computationally cheap post-processing methods would be a useful addition to meteorologists tool kit. We believe this to be especially useful in areas with sparse or underfunded local meteorological offices which characteries large sections lower and middle income countries.#text(red)[OH I don't have a cite for this besides always hearing Amir talking this]


Unaddressed in this paper are challenges of dissemination forecast information. Rural residents of low and middle income countries often do not have access to quality weather forecasts. Without improvements to forecast access, the benefits of forecast accuracy improvement will be limited. Targeted SMS based forecast delivery have effective in helping farmers prepare for and and adapt to weather shocks. @rudderLearningWeatherForecasts2024 @burligValueForecastsExperimental2024 We see high value in future work that further documents how people use weather forecasts and how forecast presentation and delivery affects outcomes.

// Why custom loss is the next big thing
The post-processing method discussed in this paper makes more useful local forecasts by focusing on the prediction of the relevant weather variables in a relevant domain. However, most cases, weather forecasts are inputs into decisions related to electrical grid management, agriculture input use, and emergency weather alerts. We hope that the method presented here can be built upon and be augmented with custom loss functions in order to cheaply create forecasts tailored for specific use cases. This is an active area of research, but most existing efforts either treat all forecast errors equally, or rely on compute intensive model finetuning methods. 





= Appendix
== MLP Architecture Details

The MLP processes flattened spatial fields concatenated with temporal encodings. Let $N_"lat" times N_"lon" times N_"var"$ denote the spatial dimensions and number of input variables. The input dimension is:
$ d_"input" = N_"lat" dot N_"lon" dot N_"var" + 2 + d_"lead" $
where the 2 accounts for sine and cosine day-of-year features, and $d_"lead"$ is the lead time embedding dimension (only added when training across multiple lead times).

The network architecture is:
+ Input layer: $d_"input" -> d_"hidden"$
+ $L-1$ hidden blocks, each containing:
  - Linear layer: $d_"hidden" -> d_"hidden"$
  - ReLU activation
  - Dropout with probability $p$
+ Output layer: $d_"hidden" -> d_"output"$

where $d_"hidden"$, $L$, and $p$ are hyperparameters.

== U-Net Architecture Details

The U-Net processes input fields as multi-channel 2D images, preserving spatial structure throughout. Temporal encodings (day-of-year sine/cosine and lead time embeddings) are broadcast to match spatial dimensions and concatenated as additional input channels.

The encoder consists of $K$ levels, where $K$ is determined automatically based on the spatial dimensions to ensure the bottleneck has spatial extent of at least $4 times 4$:
$ K = min(5, floor(log_2 (min(N_"lat", N_"lon"))) - 1) $

Each encoder level $k$ contains:
+ A convolutional block with $c_k$ output channels, where $c_1 = d_"hidden"$ and $c_k = min(2 dot c_(k-1), 128)$
+ Max pooling with kernel size 2 (except at the final level)

Each convolutional block consists of:
+ 2D Convolution ($3 times 3$ kernel, padding 1)
+ Batch Normalization
+ ReLU activation
+ 2D Dropout with probability $p$
+ 2D Convolution ($3 times 3$ kernel, padding 1)
+ Batch Normalization
+ ReLU activation
+ 2D Dropout with probability $p$

The decoder mirrors the encoder structure, using transposed convolutions ($2 times 2$ kernel, stride 2) for upsampling. Skip connections concatenate encoder features with corresponding decoder features before each decoder convolutional block. A final $1 times 1$ convolution maps to the output variable dimension.

== Training Configuration

*Optimizer.* We use the Adam optimizer with default momentum parameters ($beta_1 = 0.9$, $beta_2 = 0.999$).

*Learning Rate Schedule.* The learning rate is adjusted using ReduceLROnPlateau with factor 0.5 and patience equal to half the early stopping patience. The minimum learning rate is set to $10^(-7)$.

*Early Stopping.* Training terminates when validation loss fails to improve by at least $delta_"min"$ for $P$ consecutive epochs, where both $delta_"min"$ and $P$ are hyperparameters. The model checkpoint with lowest validation loss is retained.

*Mixed Precision Training.* When training on CUDA-enabled GPUs, we employ automatic mixed precision (AMP) to accelerate training without sacrificing model quality.

== Hyperparameter Search Spaces

@tab-hyperparam-mlp and @tab-hyperparam-unet detail the hyperparameter search spaces for the MLP and U-Net architectures, respectively.

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    stroke: none,
    table.hline(),
    table.header(
      [*Hyperparameter*], [*Distribution*], [*Range*],
    ),
    table.hline(),
    [Hidden dimension], [Categorical], [{64, 128, 256, 512, 1024}],
    [Number of layers], [Categorical], [{2, 3, 4, 5, 6}],
    [Learning rate], [Log-uniform], [$[10^(-4), 10^(-2)]$],
    [Batch size], [Categorical], [{64, 128, 256}],
    [Weight decay], [Log-uniform], [$[10^(-6), 10^(-3)]$],
    [Patience (epochs)], [Categorical], [{15, 20, 25, 30}],
    [Min delta], [Log-uniform], [$[10^(-5), 10^(-3)]$],
    [Lead time embedding dim], [Categorical], [{8, 16, 32}],
    [Dropout rate], [Uniform], [$[0.1, 0.3]$],
    table.hline(),
  ),
  caption: [MLP hyperparameter search space],
) <tab-hyperparam-mlp>

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    stroke: none,
    table.hline(),
    table.header(
      [*Hyperparameter*], [*Distribution*], [*Range*],
    ),
    table.hline(),
    [Base channels], [Categorical], [{64, 128}],
    [Learning rate], [Log-uniform], [$[10^(-4), 10^(-2)]$],
    [Batch size], [Categorical], [{64, 128, 256}],
    [Weight decay], [Log-uniform], [$[10^(-6), 10^(-3)]$],
    [Patience (epochs)], [Categorical], [{15, 20, 25, 30}],
    [Min delta], [Log-uniform], [$[10^(-5), 10^(-3)]$],
    [Lead time embedding dim], [Categorical], [{8, 16}],
    [Dropout rate], [Uniform], [$[0.05, 0.20]$],
    table.hline(),
  ),
  caption: [U-Net hyperparameter search space],
) <tab-hyperparam-unet>


= Appndix Figures

- Binscatters showing orginal pangu and ifs and corrected pangu. 
- not sure how useful these are
- Pangu and IFS are very similar after 1 day. IFS is better than pangu for 24h extra tropical predictions
- IFS is worse than pangu for 1 day tropical wind speed but better for high latitude 1 day wind speed prediction. These one day wind speed plots also show interesting cupc relationship with distance from equator (not sure what to make of this).
#grid(
  columns: 1,
  rows: 2,
figure(
  image("figs/model_compare_binscatter_2m_temperature_equator.png"),
  caption: [Model Compare abs(latitude) 2m Temperature]
), 
figure(
  image("figs/model_compare_binscatter_10m_wind_speed_equator.png"),
  caption: [Model Compare abs(latitude) 10m Wind Speed]
) 
)

#grid(
  columns: 1,
  rows: 2,
figure(
  image("figs/model_compare_binscatter_2m_temperature_sdor.png"),
  caption: [Model Compare sdor 2m Temperature]
), 
figure(
  image("figs/model_compare_binscatter_10m_wind_speed_sdor.png"),
  caption: [Model Compare sdor 10m Wind Speed]
) 
)



- Global 9 day wind speed improvement map: 
  - Map needs to be cleaned up
  - Can eye ball the things shown explicitly in the binscatters.
#figure(
  image("figs/pangu/global_improvement_map_pixel_10m_wind_speed_pangu_lt24.png")
)
#figure(
  image("figs/pangu/global_improvement_map_pixel_10m_wind_speed_pangu_lt120.png")
)
#figure(
  image("figs/pangu/global_improvement_map_pixel_10m_wind_speed_pangu_lt216.png")
)

#figure(
  image("figs/pangu/global_improvement_map_pixel_2m_temperature_pangu_lt24.png")
)
#figure(
  image("figs/pangu/global_improvement_map_pixel_2m_temperature_pangu_lt120.png")
)
#figure(
  image("figs/pangu/global_improvement_map_pixel_2m_temperature_pangu_lt216.png")
)

#text(red)[something like this should be promoted to the main, figure 2 or something]
== Appendix full binscatters and region size compairsons
#figure(
  image("figs/pangu/scatter_equator_2m_temperature_all_metrics_binscatter.png")
)
#figure(
  image("figs/pangu/scatter_equator_10m_wind_speed_all_metrics_binscatter.png")
)
#figure(
  image("figs/pangu/scatter_sdor_2m_temperature_all_metrics_binscatter.png")
)
#figure(
  image("figs/pangu/scatter_sdor_10m_wind_speed_all_metrics_binscatter.png")
)
#figure(
  image("figs/pangu/region_size_comparison_10m_wind_speed_mlp_pangu.png"),
  caption: [Effect of Expanding Training Region: 10m Wind Speed]
)

//the result highlights that (@figure1)

= Bibliograpy

#bibliography("post_processing_ml_forecasts.bib", style: "chicago-author-date")