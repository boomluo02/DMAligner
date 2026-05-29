import argparse
import os
import time
import yaml
from easydict import EasyDict as edict
from tools.utils_1 import save_yaml

def parse_args():
    parser = argparse.ArgumentParser(description="BCD training script.")
    # env
    parser.add_argument(
        "--base_config",
        type=str,
        # default="config/train_config.yaml",
        # default="config/test_config.yaml",
        default="config/test_config_davis.yaml",
        # default="config/test_config_sintel.yaml",
        help="base config file path",
    )
    parser.add_argument("--use_wandb", action="store_true", help="use wandb for logging")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--data", type=str, default="bcd_data", help="Data type. bcd_data or render_data")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--random_mask", action="store_true", help="Randomly mask img during training.")
    parser.add_argument("--decoupled_attn", action="store_true", help="Use decoupled attention for training.")
    args = parser.parse_args()

    return args


def set_config():
    cfg_args = parse_args()
    # for safety, huggingface token won't be saved in config file
    hub_token = cfg_args.hub_token

    with open(cfg_args.base_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        replace_none_with_python_none(config)
        cfg = edict(config)

        # env
        cfg.env.run_id = get_timestamp()
        cfg.env.base_config = cfg_args.base_config
        cfg.env.debug = cfg_args.debug
        cfg.env.seed = cfg_args.seed
        cfg.env.resume = cfg_args.resume
        cfg.env.data = cfg_args.data
        cfg.env.decoupled_attn = cfg_args.decoupled_attn

        # data
        cfg.data.random_mask = cfg_args.random_mask

        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1:
            cfg.env.local_rank = env_local_rank

        # wandb
        if cfg_args.use_wandb:
            cfg.tracker.report_to = 'wandb'

        # debug mode
        if cfg_args.debug:
            cfg.env.log_dir = cfg.env.log_dir + "_debug"
            cfg.env.seed = 42
            if cfg.env.mode == 'train':
                cfg.train.epochs = 5
                cfg.train.batch_size = 1
                cfg.train.validation_epochs = 2
                cfg.train.checkpointing_steps = 20

        cfg.env.signature = f"{cfg.env.mode}-{cfg.env.run_id}" if not cfg.env.debug else f"debug-{cfg.env.run_id}"
        # cfg.env.log_dir = os.path.join(cfg.env.log_dir, cfg.env.signature)
        cfg.env.log_dir = f"{cfg.env.log_dir}/{cfg.env.mode}/{cfg.env.signature}"
        cfg.env.output_dir = os.path.join(cfg.env.log_dir, cfg.env.output_dir)
        if not os.path.exists(cfg.env.output_dir):
            os.makedirs(cfg.env.output_dir, exist_ok=True)

        if not os.path.exists(cfg.env.log_dir):
            os.makedirs(cfg.env.log_dir, exist_ok=True)

        if cfg.env.mode == 'train':
            save_dir = f"{cfg.env.log_dir}/val_output"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            if cfg.train.val_save_output:
                cfg.train.val_save_dir = save_dir
        else:
            save_dir = f"{cfg.env.log_dir}/test_output"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            cfg.env.test_save_dir = save_dir

        save_yaml(cfg)

        return cfg, hub_token

# Function to convert string 'None' to actual None
def replace_none_with_python_none(d):
    for k, v in d.items():
        if isinstance(v, dict):
            replace_none_with_python_none(v)
        elif v in ['None', 'none', "", '']:  # Check if value is string 'None'
            d[k] = None  # Replace with Python None

def get_timestamp():
    # wait 5 seconds to make sure timestamp is same in DDP
    time.sleep(5)
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    return timestamp