# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import copy
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.data.items()}
        return item


class WandbManager:
    def __init__(self) -> None:
        assert (
            self._wandb_available
        ), "wandb is not installe, please install via pip install wandb"
        import wandb

        self._wandb = wandb
        self._initialized = False

    def _wandb_available():
        # any value of WANDB_DISABLED disables wandb
        if os.getenv("WANDB_DISABLED", "").upper():
            print(
                "Not using wandb for logging, if this is not intended, unset WANDB_DISABLED env var"
            )
            return False
        return importlib.util.find_spec("wandb") is not None

    def setup(self, args, configs, **kwargs):
        if not isinstance(args, dict):
            args = args.__dict__
        project_name = args.get("project", "debug")
        wandb_dir = configs["output_path"]
        os.makedirs(wandb_dir, exist_ok=True)

        args_dict = OmegaConf.to_container(configs)
        self._wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            entity=args.get("entity", None),
            mode=args.get("mode", "disabled"),
            name=args.get("experiment_id", "test"),
            # track hyperparameters and run metadata
            config=args_dict,
            dir=wandb_dir,  # setup a directory for the runs and results, so that it can be resumed.
            resume=not args.get("force_rerun"),
        )
        self._initialized = True

    def log(self, logs, head=None):
        if head is not None:
            copied_logs = copy.deepcopy(logs)
            for key in logs.keys():
                copied_logs[f"{head}/{key}"] = logs[key]
                copied_logs.pop(key, None)
            logs = copied_logs
        self._wandb.log(logs)

    def finish(self):
        self._wandb.finish()

    def summarize(self, outputs):
        # add values to the wandb summary => only works for scalars
        for k, v in outputs.items():
            self._wandb.run.summary[k] = v.item()


def setup_logging(config):
    dict_config = dict(config)
    logger_conf = dict_config["logging"]
    model_config = dict_config["dataset"]
    if logger_conf["writer"] == "wandb":
        if logger_conf["mode"] == "offline":
            os.environ["WANDB_MODE"] = "offline"
        from utils import WandbManager

        writer = WandbManager()
        writer.setup(
            {**logger_conf, "experiment_id": dict_config["experiment_id"]},
            configs=config,
        )
        # writer.setup({ **logger_conf, **model_config, 'experiment_id': dict_config['experiment_id'], 'seed': dict_config['seed'] }, configs=config)
    else:
        raise NotImplementedError("Specified writer not recognized!")
    return writer


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def set_weight_decay(optimizer, weight_decay):
    for param_group in optimizer.param_groups:
        param_group["weight_decay"] = weight_decay


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
