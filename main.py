
import argparse

from genrec.pipeline import Pipeline
from genrec.utils import parse_command_line_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CSA', help='Model name')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    # Add support for the cache_dir argument
    parser.add_argument('--cache_dir', type=str, default=None, help='Cache directory path')
    # Add support for the text_embedding_dim argument
    parser.add_argument('--text_embedding_dim', type=int, default=None, help='Text embedding dimension for PCA')
    # Parse epochs explicitly to avoid being missed by parse_command_line_args or affected by override order
    # parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    
    # If cache_dir is specified, add it to the config
    if args.cache_dir:
        command_line_configs['cache_dir'] = args.cache_dir
    
    # If text_embedding_dim is specified, add it to the config
    if args.text_embedding_dim:
        command_line_configs['text_embedding_dim'] = args.text_embedding_dim

    # # Pass epochs explicitly to guarantee command-line --epochs=N takes effect (highest priority)
    # if args.epochs is not None:
    #     command_line_configs['epochs'] = args.epochs

    pipeline = Pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint,
        config_dict=command_line_configs
    )
    pipeline.run()
