# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from genrec.pipeline import Pipeline
from genrec.utils import parse_command_line_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CSA', help='Model name')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    # 添加cache_dir参数支持
    parser.add_argument('--cache_dir', type=str, default=None, help='Cache directory path')
    # 添加text_embedding_dim参数支持
    parser.add_argument('--text_embedding_dim', type=int, default=None, help='Text embedding dimension for PCA')
    # 显式解析 epochs，避免被 parse_command_line_args 漏掉或覆盖顺序问题
    # parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    
    # 如果指定了cache_dir，添加到配置中
    if args.cache_dir:
        command_line_configs['cache_dir'] = args.cache_dir
    
    # 如果指定了text_embedding_dim，添加到配置中
    if args.text_embedding_dim:
        command_line_configs['text_embedding_dim'] = args.text_embedding_dim

    # # 显式传入 epochs，保证命令行 --epochs=N 一定生效（优先级最高）
    # if args.epochs is not None:
    #     command_line_configs['epochs'] = args.epochs

    pipeline = Pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_dict=command_line_configs
    )
    pipeline.run()
