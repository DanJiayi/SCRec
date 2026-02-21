# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
import traceback

import hydra

import torch
from ID_generation.preprocessing.data_process import preprocessing
from ID_generation.train_rqvae import train as train_sid
from ID_generation.utils import process_data_split, process_embeddings
from omegaconf import DictConfig
from src.training import train_tiger

from utils import set_seed


class set_dir:
    def __init__(self, config):
        self.directory = "./ID_generation/preprocessing/raw_data/"
        self.directory_processed = "./ID_generation/preprocessing/processed/"
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(self.directory_processed, exist_ok=True)

        if config["test_method"] in ["tiger", "liger"]:
            self.rqvae_save_dir = "./ID_generation/ID/"
            os.makedirs(self.rqvae_save_dir, exist_ok=True)

            id_filename = (
                f"{config['dataset']['name']}_{config['dataset']['content_model']}"
            )
            self.id_save_location = os.path.join(
                self.rqvae_save_dir, id_filename + f"_{config['seed']}.pkl"
            )

        self.embedding_save_name = f"_{config['dataset']['content_model']}"
        embedding_suffix = config["dataset"].get("embedding_suffix", "")  # e.g. "_new" for _embeddings_new.pt
        # When using converted embeddings, filename may use a different model name (e.g. sentence-t5-base)
        embedding_file_model = config["dataset"].get("embedding_file_model") or config["dataset"]["content_model"]
        embedding_filename = f"{config['dataset']['name']}_{embedding_file_model}"
        self.embedding_save_path = os.path.join(
            self.directory_processed, embedding_filename + "_embeddings" + embedding_suffix + ".pt"
        )

        self.result_save_dir = f"./results/{config['test_method']}/"
        os.makedirs(self.result_save_dir, exist_ok=True)

    def set_config(self, config):
        config["dataset"]["raw_data_path"] = self.directory
        config["dataset"]["processed_data_path"] = self.directory_processed
        config["output_path"] = os.path.join(
            self.result_save_dir,
            f"{config['dataset']['type']}_{config['dataset']['name']}",
            f"{config['experiment_id']}_seed_{config['seed']}",
        )
        os.makedirs(config["output_path"], exist_ok=True)
        return config


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(config: DictConfig) -> None:

    # print(config)
    device = (
        torch.device(f"cuda:{config['device_id']}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    set_seed(config["seed"])

    PATH_CONFIG = set_dir(config)
    config = PATH_CONFIG.set_config(config)
    config["logging"]["project"] = "liger"
    is_steam = config["dataset"]["type"] == "steam"

    try:
        data_file, id2meta_file, item2attribute_file = preprocessing(config["dataset"])
        # id2meta_file: the file that save item_id to meta info, we will later use it for sentence T5 embedding generation
        # data_file: the file that save the user-item interactions.

        train_config = {
            **config["dataset"],
            **{
                k: v
                for k, v in config.items()
                if k not in ["logging", "dataset", "method"]
            },
        }
        method_config = {
            **config["method"],
            **{
                k: v
                for k, v in config.items()
                if k not in ["logging", "dataset", "method"]
            },
        }

        # load id split
        id_split, user_sequence = process_data_split(
            config, data_file, id2meta_file, is_steam=is_steam
        )

        # load item embedding
        item_embedding = process_embeddings(
            config, device, id2meta_file, PATH_CONFIG.embedding_save_path
        )

        train_sid(
            config, device, item_embedding, id_split, PATH_CONFIG.id_save_location
        )

        # Optional CSA module switch. We keep the default behavior (no CSA)
        # to ensure existing scripts such as scripts/ours.sh produce
        # identical results unless the user explicitly enables CSA via
        # `--use_csa=True`.
        use_csa = config.get("use_csa", False)

        train_tiger(
            config,
            train_config,
            method_config,
            id_split,
            user_sequence,
            item_embedding,
            PATH_CONFIG.id_save_location,
            device=device,
            use_csa=use_csa,
        )

    except BaseException:
        traceback.print_exc(file=sys.stderr)
        raise

    finally:
        # fflush everything
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    main()
