# how it works - steps

## system & global parameters
- pid (ki, kd, kp)
- temperature (sp, start)
- rl agent
- anomaly (start_min, start_max, magnitude_min, magnitude_max)
- reward weights

## class objects
- system_identifier
- rl agent/model
- environment
- pid controller
- anomaly generator

## system initializer
## data handling & processing
- init(filepath)
- load_data()
## baseline using pid controller
- init(kp, ki, kd)
- reset()
- set_control_signal(sp, delta_t, curr_temp)
## fluctuations & anomalies
- init()
- reset()
- generate_anomaly()
## reinforcement learning algorithm
## data analysis & visualizations
- calculate_accurary_metric()
- calculate_performance_metric()
## logging & monitoring
## libraries (DONE)
- stable baselines3 (sb3)
- numpy, pandas, matplotlib/seaborn
- gymnasium
## readme
- install
- dependencies
- usage
- license
