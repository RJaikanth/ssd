clear

export CUDA_LAUNCH_BLOCKING=1
export CONFIG_DIR="./config"

python main.py --train --exp_config=experiments/test_run.json