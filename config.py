from scipy.stats import uniform

# dir & file paths
DATASET_PATH = '/rl_control_system/data/MZVAV-2-1.csv'
LOG_DIRECTORY = './logs/'
SAVE_PATH = './test_trained_models'
RESULTS_DIRECTORY = './results/'

# mlp regressor hyperparameters - randomizedsearchcv
MLP_HIDDEN_LAYER_SIZES = [(64,), (128,), (64, 32), (128, 64), (128, 128)]
MLP_ACTIVATION = ['relu', 'tanh']
MLP_SOLVER = ['adam', 'sgd']
MLP_ALPHA = uniform(loc=0.0001, scale=0.01)
MLP_LEARNING_RATE_INIT = uniform(loc=0.0001, scale=0.001)
MLP_N_ITER = 10

# used for initial optimization; update these values once hyperparameter are optimized
MLP_HIDDEN_LAYER_SIZES_OPTIMAL = (64, 32)
MLP_ACTIVATION_OPTIMAL = 'relu'
MLP_SOLVER_OPTIMAL = 'adam'
MLP_ALPHA_OPTIMAL = 0.00456
MLP_LEARNING_RATE_INIT_OPTIMAL = 0.000559

# rl ppo agent hyperparameters
RL_BATCH_SIZE = 256  # larger batch size = stable gradient estimates
RL_N_STEPS = 2048  # number of actions, obersvations before the agent updates the policy
RL_LEARNING_RATE = 0.000065  # smaller = slower = stabler rate
RL_ALGORITHM_TYPE = 'PPO'
RL_GAMMA = 0.95
RL_TOTAL_TIMESTEPS = 1_000_000  # num_episodes = total_timesteps/sim_steps before stopping the training

# simulation parameters
DELTA_T_IN_SEC = 60.0  # step interval in sec
N_EVAL_EPISODES = 100  # num_of_epd; total steps in 1 eval run = episode * steps
SIMULATED_DURATION_STEPS = 500  # max length of each episode; step_counter < this

# pid controller parameters
PID_KP = 0.8  # Proportional gain: responsiveness to current error
# Integral gain: responsiveness to accumulated error (helps eliminate steady-state error)
PID_KI = 0.0
# Derivative gain: responsiveness to rate of change of error (helps dampen oscillations)
PID_KD = 0.18

# system parameters
CRITICAL_TEMP_LOW = 65.0
CRITICAL_TEMP_HIGH = 85.0
TEMP_SETPOINT = 75.0
SUPPLY_AIR_TEMP_MIN_LIMIT = 68.0
SUPPLY_AIR_TEMP_MAX_LIMIT = 78.0
RECOVERY_TIME_LIMIT = 30.0

# env parameters
N_ENVIRONMENT = 1
SETPOINT_MIN_FALLBACK = 70.0
SETPOINT_MAX_FALLBACK = 75.0
OUTDOOR_AIR_TEMP_MIN_FALLBACK = 0.0
OUTDOOR_AIR_TEMP_MAX_FALLBACK = 110.0
OUTDOOR_AIR_TEMP_MIN_WALK_CLAMP = -5.0
OUTDOOR_AIR_TEMP_MAX_WALK_CLAMP = 115.0
INITIAL_TEMP_DEVIATION_RANGE = 2.0

# anomaly parameters
ANOMALY_MAGNITUDE_MIN = 10.0
ANOMALY_MAGNITUDE_MAX = 15.0
ANOMALY_DURATION_MIN = 40.0
ANOMALY_DURATION_MAX = 90.0
ANOMALY_START_MIN = 90.0
ANOMALY_START_MAX = 400.0
ANOMALY_DETECTION_THRESHOLD = 3.0
ANOMALY_DETECTION_WINDOW = 5

# noise parameters
NOISE_MAGNITUDE = 1.0

# reward weights
REWARD_CRITICAL_VIOLATION_PENALTY = -10_000  # pentalized for critical violation
REWARD_CONTROL_EFFORT_PENALTY = -0.01  # penatalized for large control signals
REWARD_ERROR_PENALTY = -2.0  # pentalized for errors/anomalites
REWARD_TIMEOUT_PENALTY = -1000  # pentalized for exceeding time
REWARD_ANOMALY_CORRECTED_REWARD = 500  # rewarded if anomaly corrected
