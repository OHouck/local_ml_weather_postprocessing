= Ozma Houck EEE Check in Report 2-5-2026

=== Main new project: How do weather forecast errors impact transmission congestion and LMPs 
- In which areas do forecast errors most impact electricity prices? This is useful when designing forecasting models and when making decisions about where to invest in grid improving technologies.
- In ERCOT, the marginal cost of generation without transmission constraints is called the system lambda. For the real-time market, it is available every 5 minutes. The difference between LMP and system lambda is due to transmission congestion.
  - Real-Time LMPs and System Lambdas are available at 5-minute intervals for all nodes in ERCOT.
- What's been done:
  - Have least cost dispatch simulations working on an ERCOT simulation model. Has a simplified transmission network.
  - Started to pull weather forecasts and ground truth weather station data.
- Proposed modeling plan: 
  - Use Temperature and Wind Speed Forecast Errors to predict the Real Time LMP - System Lambda difference at a node 
  - Interested in training a graph neural net using the high voltage transmission network as the graph structure. 
- Question: How important is it to untangle how much of the LMP - System Lambda difference is due to firms bidding above marginal cost when they have market power? 

=== Other Updates
- Made progress on project that I started last summer about predicting the effect of dry spell lengths at different times on wheat yields in India. Have code to calculate and measure dry spells. Trying to work with Amir to get access to yield data from CIL. 
- Almost done with ML Weather Forecasting Paper draft. As a commitment device: Goal is to have the paper submitted by the next check in!
=== Mostly for Fun: Very Early Result of Training Joint Temperature and Wind Speed Forecasting Model over Texas
- Model jointly predicts temperature and wind speed over Texas and penalizes same sign errors half as much as opposite sign errors.
- Rationale: in the summer, positive temperature errors and positive wind speed errors could "cancel out" the effects on transmission congestion. 
- Model does a moderate job at improving the targeted metric (top plot) without horribly degrading individual variable performance (bottom 4 plots).


// Show maps of original errors and improvements from post processing
#align(center)[
  #image("figures/map_improvement_joint_temp_wind_loss_joint_temp_wind_pangu_texas_lt24h.png", width: 50%)
]
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  // Row 1
  image("figures/map_original_rmse_2m_temperature_pangu_texas_lt24h.png"),
  image("figures/map_improvement_rmse_2m_temperature_pangu_texas_lt24h.png"),
  // Row 2
  image("figures/map_original_rmse_10m_wind_speed_pangu_texas_lt24h.png"),
  image("figures/map_improvement_rmse_10m_wind_speed_pangu_texas_lt24h.png"),
)

