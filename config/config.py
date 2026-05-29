import argparse
import os
import random
import time
import yaml
from easydict import EasyDict as edict
from tools.utils import save_yaml


def parse_args():
    parser = argparse.ArgumentParser()

    # env
    parser.add_argument(
        "--base_config",
        type=str,
        default="config/config.yaml",
        help="base config file path",
    )
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=random.randint(0, 1000000000))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mode", type=str, default="train") # train or test

    # DDP
    parser.add_argument("--use_ddp", action="store_true")  # default False

    args = parser.parse_args()
    return args


def set_config():
    cfg_args = parse_args()

    with open(cfg_args.base_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cfg = edict(config)

        # env
        cfg.env.run_id = time.strftime("%Y%m%d%H%M%S", time.localtime())
        cfg.env.base_config = cfg_args.base_config
        cfg.env.use_wandb = cfg_args.use_wandb
        cfg.env.debug = cfg_args.debug
        cfg.env.seed = cfg_args.seed
        cfg.env.resume = cfg_args.resume
        cfg.env.device = cfg_args.device
        cfg.env.mode = cfg_args.mode

        # DDP
        cfg.DDP.use_ddp = cfg_args.use_ddp

        # debug mode
        if cfg_args.debug:
            cfg.env.log_dir = "debug"
            cfg.env.seed = 42
            cfg.train.epochs = 5
            cfg.train.batch_size = 2
            cfg.train.val_interval = 2

        cfg.env.signature = f"{cfg.env.mode}-{cfg.env.run_id}"
        cfg.env.log_dir = os.path.join(cfg.env.log_dir, cfg.env.signature)

        if cfg.DDP.use_ddp:
            cfg.env.signature += "-DDP"
            cfg.env.log_dir += "-DDP"

        if not os.path.exists(cfg.env.log_dir):
            os.makedirs(cfg.env.log_dir)

        if cfg.train.val_save_output:
            save_dir = f"{cfg.env.log_dir}/val_output"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cfg.train.val_save_dir = save_dir

        if cfg.test.save_test_img:
            TEST_IMG_DIR = f"{cfg.env.log_dir}/test_img"
            if not os.path.exists(TEST_IMG_DIR):
                os.makedirs(TEST_IMG_DIR)
            cfg.test.test_img_dir = TEST_IMG_DIR

        cfg.env.CKPT_DIR = os.path.join(cfg.env.log_dir, cfg.env.CKPT_DIR)

        if not os.path.exists(cfg.env.CKPT_DIR):
            os.makedirs(cfg.env.CKPT_DIR)

        save_yaml(cfg)

        return cfg
