import importlib
import os
import re
import sys
import yaml
import html
import hashlib
import datetime
import requests
import tiktoken
from typing import Union, Optional
import logging
from logging import getLogger
import numpy as np

from decoder.model import AbstractModel
from dataloader.dataset import AbstractDataset
from accelerate.utils import set_seed


def init_seed(seed, reproducibility):
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """

    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_local_time():
    r"""Get current time

    Returns:
        str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M")
    return cur


def get_command_line_args_str():
    return '_'.join(sys.argv).replace('/', '|')


# --- MODIFICATION START: full replacement of get_file_name function ---
def get_file_name(config: dict, suffix: str = ''):
    # 1. Generate a short hash based on the full config to ensure uniqueness
    #    This hash is the final identifier used to distinguish different parameter combinations
    config_hash = hashlib.md5(str(config).encode(encoding="utf-8")).hexdigest()[:8] # Use 8-digit hash

    # 2. Build a concise and readable filename prefix
    #    Include model, dataset, category, and major sweep parameters for quick identification
    parts = []
    if 'model' in config:
        parts.append(config['model'])
    if 'dataset' in config:
        parts.append(config['dataset'])
    if 'category' in config: # Dataset category is usually an important discriminator
        parts.append(config['category'])
    
    # Add major sweep parameters and keep values concise in filename (replace decimal points, etc.)
    # Check key existence to avoid KeyError
    if 'lr' in config:
        parts.append(f"lr{str(config['lr']).replace('.', 'p')}") # e.g.: lr0p0003
    if 'n_codebook' in config:
        parts.append(f"ncb{config['n_codebook']}") # e.g.: ncb32
    if 'temperature' in config:
        parts.append(f"temp{str(config['temperature']).replace('.', 'p')}") # e.g.: temp0p07
    if 'n_edges' in config:
        parts.append(f"ne{config['n_edges']}") # e.g.: ne100
    if 'propagation_steps' in config:
        parts.append(f"ps{config['propagation_steps']}") # e.g.: ps3

    # If no specific parameters are available to build a prefix, use run_id as the base
    if not parts:
        base_filename = config.get('run_id', 'run')
    else:
        base_filename = "_".join(parts)
    
    # 3. Compose filename: [base_prefix]-[local_time]-[config_hash][suffix]
    # This keeps readability while ensuring uniqueness with timestamp and hash
    logfilename = "{}-{}-{}{}".format(
        base_filename,
        config['run_local_time'], # e.g.: Jun-21-2025_01-50
        config_hash,
        suffix
    )

    # 4. Check final filename length and truncate to keep it well below filesystem limits (e.g., 255 chars)
    # MAX_SAFE_FILENAME_LEN is a conservative limit and can be adjusted for your filesystem
    MAX_SAFE_FILENAME_LEN = 150 
    if len(logfilename) > MAX_SAFE_FILENAME_LEN:
        # Truncation strategy: keep enough prefix and then append timestamp, hash, and suffix to preserve uniqueness
        
        # Compute the required length for unique parts (hash + timestamp + suffix + separators)
        required_unique_part_len = len(config_hash) + len(config['run_local_time']) + len(suffix) + 2 # +2 for hyphens
        
        # Compute allowed prefix length
        prefix_len = MAX_SAFE_FILENAME_LEN - required_unique_part_len
        
        if prefix_len > 0:
            base_filename_short = base_filename[:prefix_len]
        else:
            base_filename_short = "" # If even the prefix does not fit, keep it empty

        logfilename = "{}-{}-{}{}".format(
            base_filename_short,
            config['run_local_time'],
            config_hash,
            suffix
        )
        # Confirm length again after truncation; if still too long, force-truncate to safe length
        if len(logfilename) > MAX_SAFE_FILENAME_LEN:
             logfilename = logfilename[:MAX_SAFE_FILENAME_LEN] 

    return logfilename


def init_logger(config: dict):
    LOGROOT = config['log_dir']
    os.makedirs(LOGROOT, exist_ok=True)
    dataset_name = os.path.join(LOGROOT, config["dataset"])
    os.makedirs(dataset_name, exist_ok=True)
    model_name = os.path.join(dataset_name, config["model"])
    os.makedirs(model_name, exist_ok=True)

    logfilename = get_file_name(config, suffix='.log')
    if len(logfilename) > 255:
        logfilename = logfilename[:245] + logfilename[-10:]
    logfilepath = os.path.join(LOGROOT, config["dataset"], config["model"], logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fileformatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    logging.basicConfig(level=logging.INFO, handlers=[sh, fh])

    if not config['accelerator'].is_main_process:
        from datasets.utils.logging import disable_progress_bar
        disable_progress_bar()


def log(message, accelerator, logger, level='info'):
    if accelerator.is_main_process:
        if level == 'info':
            logger.info(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'debug':
            logger.debug(message)
        else:
            raise ValueError(f'Invalid log level: {level}')


def get_tokenizer(model_name: str):
    """
    Retrieves the tokenizer for a given model name.

    Args:
        model_name (str): The model name.

    Returns:
        AbstractTokenizer: The tokenizer for the given model name.

    Raises:
        ValueError: If the tokenizer is not found.
    """
    try:
        tokenizer_class = getattr(
            importlib.import_module(f'decoder.models.{model_name}.tokenizer'),
            f'{model_name}Tokenizer'
        )
    except:
        raise ValueError(f'Tokenizer for model "{model_name}" not found.')
    return tokenizer_class


def get_model(model_name: Union[str, AbstractModel]) -> AbstractModel:
    """
    Retrieves the model class based on the provided model name.

    Args:
        model_name (Union[str, AbstractModel]): The name of the model or an instance of the model class.

    Returns:
        AbstractModel: The model class corresponding to the provided model name.

    Raises:
        ValueError: If the model name is not found.
    """
    if isinstance(model_name, AbstractModel):
        return model_name

    try:
        # Special handling for multimodal models
        if model_name == 'MultimodalRPG':
            from decoder.models.multimodal_rpg import MultimodalRPG
            return MultimodalRPG
        elif model_name == 'MultimodalRPGSimple':
            from decoder.models.multimodal_rpg_simple import MultimodalRPGSimple
            return MultimodalRPGSimple
        else:
            model_class = getattr(
                importlib.import_module('decoder.models'),
                model_name
            )
    except:
        raise ValueError(f'Model "{model_name}" not found.')
    return model_class


def get_dataset(dataset_name: Union[str, AbstractDataset]) -> AbstractDataset:
    """
    Get the dataset object based on the dataset name or directly return the dataset object if it is already provided.

    Args:
        dataset_name (Union[str, AbstractDataset]): The name of the dataset or the dataset object itself.

    Returns:
        AbstractDataset: The dataset object.

    Raises:
        ValueError: If the dataset name is not found.
    """
    if isinstance(dataset_name, AbstractDataset):
        return dataset_name

    try:
        if dataset_name == 'AmazonReviews2014':
            # Use AmazonDataProcessor for full data processing
            from dataloader.amazon_data_processor import AmazonDataProcessor
            
            # Create a wrapper class to make it compatible with the AbstractDataset interface
            class AmazonReviews2014Wrapper(AbstractDataset):
                def __init__(self, config):
                    super().__init__(config)
                    self.category = config['category']
                    self.cache_dir = config.get('cache_dir', 'cache')
                    
                    # Use AmazonDataProcessor for data processing
                    processor = AmazonDataProcessor(
                        category=self.category,
                        cache_dir=self.cache_dir,
                        config_path=None,
                        config=config
                    )
                    processor.run_full_pipeline()
                    
                    # Load processed data
                    self.all_item_seqs = processor.all_item_seqs
                    self.id_mapping = processor.id_mapping
                    self.item2meta = processor.item2meta
                    
                def _download_and_process_raw(self):
                    # Already completed in __init__
                    pass
            
            return AmazonReviews2014Wrapper
        else:
            # Processing logic for other datasets
            dataset_module = importlib.import_module(f'dataloader.{dataset_name}.dataset')
            dataset_class = getattr(dataset_module, dataset_name)
    except:
        raise ValueError(f'Dataset "{dataset_name}" not found.')
    return dataset_class


def get_trainer(model_name: Union[str, AbstractModel]):
    """
    Returns the trainer class based on the given model name.

    Parameters:
        model_name (Union[str, AbstractModel]): The name of the model or an instance of the AbstractModel class.

    Returns:
        trainer_class: The trainer class corresponding to the given model name. If the model name is not found, the default Trainer class is returned.
    """
    from decoder.trainer import Trainer
    if isinstance(model_name, str):
        try:
            trainer_class = getattr(
                importlib.import_module(f'decoder.models.{model_name}.trainer'),
                f'{model_name}Trainer'
            )
            return trainer_class
        except:
            return Trainer
    else:
        return Trainer


def get_pipeline(model_name: Union[str, AbstractModel]):
    """
    Returns the pipeline class based on the given model name.

    Parameters:
        model_name (Union[str, AbstractModel]): The name of the model or an instance of the AbstractModel class.

    Returns:
        pipeline_class: The pipeline class corresponding to the given model name. If the model name is not found, the default Pipeline class is returned.
    """
    from decoder.pipeline import Pipeline
    if isinstance(model_name, str):
        try:
            pipeline_class = getattr(
                importlib.import_module(f'decoder.models.{model_name}.pipeline'),
                f'{model_name}Pipeline'
            )
            return pipeline_class
        except:
            return Pipeline
    else:
        return Pipeline

def get_total_steps(config, train_dataloader):
    """
    Calculate the total number of steps for training based on the given configuration and dataloader.

    Args:
        config (dict): The configuration dictionary containing the training parameters.
        train_dataloader (DataLoader): The dataloader for the training dataset.

    Returns:
        int: The total number of steps for training.

    """
    if config['steps'] is not None:
        return config['steps']
    else:
        return len(train_dataloader) * config['epochs']


def convert_config_dict(config: dict) -> dict:
    """
    Convert the values in a dictionary to their appropriate types.

    Args:
        config (dict): The dictionary containing the configuration values.

    Returns:
        dict: The dictionary with the converted values.

    """
    for key in config:
        v = config[key]
        if not isinstance(v, str):
            continue
        try:
            new_v = eval(v)
            if new_v is not None and not isinstance(
                new_v, (str, int, float, bool, list, dict, tuple)
            ):
                new_v = v
        except (NameError, SyntaxError, TypeError):
            if isinstance(v, str) and v.lower() in ['true', 'false']:
                new_v = (v.lower() == 'true')
            else:
                new_v = v
        config[key] = new_v
    return config


def get_config(
    model_name: Union[str, AbstractModel],
    dataset_name: Union[str, AbstractDataset],
    config_file: Union[str, list[str], None],
    config_dict: Optional[dict]
) -> dict:
    """
    Get the configuration for a model and dataset.
    Overwrite rule: config_dict > config_file > model config.yaml > dataset config.yaml > default.yaml

    Args:
        model_name (Union[str, AbstractModel]): The name of the model or an instance of the model class.
        dataset_name (Union[str, AbstractDataset]): The name of the dataset or an instance of the dataset class.
        config_file (Union[str, list[str], None]): The path to additional configuration file(s) or a list of paths to multiple additional configuration files. If None, default configurations will be used.
        config_dict (Optional[dict]): A dictionary containing additional configuration options. These options will override the ones loaded from the configuration file(s).

    Returns:
        dict: The final configuration dictionary.

    Raises:
        FileNotFoundError: If any of the specified configuration files cannot be found.

    Note:
        - If `model_name` is a string, the function will attempt to load the model's configuration file located at `genrec/models/{model_name}/config.yaml`.
        - If `dataset_name` is a string, the function will attempt to load the dataset's configuration file located at `genrec/datasets/{dataset_name}/config.yaml`.
        - The function will merge the configurations from all the specified configuration files and the `config_dict` parameter.
    """
    final_config = {}
    logger = getLogger()

    # Load default configs
    current_path = os.path.dirname(os.path.realpath(__file__))
    config_file_list = [os.path.join(current_path, 'default.yaml')]

    if isinstance(dataset_name, str):
        config_file_list.append(
            os.path.join(current_path, f'../dataloader/{dataset_name}/config.yaml')
        )
        final_config['dataset'] = dataset_name
    else:
        logger.info(
            'Custom dataset, '
            'whose config should be manually loaded and passed '
            'via "config_file" or "config_dict".'
        )
        final_config['dataset'] = dataset_name.__class__.__name__

    if isinstance(model_name, str):
        config_file_list.append(
            os.path.join(current_path, f'models/config.yaml')
        )
        final_config['model'] = model_name
    else:
        logger.info(
            'Custom model, '
            'whose config should be manually loaded and passed '
            'via "config_file" or "config_dict".'
        )
        final_config['model'] = model_name.__class__.__name__

    if config_file:
        if isinstance(config_file, str):
            config_file = [config_file]
        config_file_list.extend(config_file)

    for file in config_file_list:
        cur_config = yaml.safe_load(open(file, 'r'))
        if cur_config is not None:
            final_config.update(cur_config)

    if config_dict:
        final_config.update(config_dict)

    final_config['run_local_time'] = get_local_time()

    final_config = convert_config_dict(final_config)
    return final_config


def parse_command_line_args(unparsed: list[str]) -> dict:
    """
    Parses command line arguments and returns a dictionary of key-value pairs.

    Args:
        unparsed (list[str]): A list of command line arguments in the format '--key=value'.

    Returns:
        dict: A dictionary containing the parsed key-value pairs.

    Example:
        >>> parse_command_line_args(['--name=John', '--age=25', '--is_student=True'])
        {'name': 'John', 'age': 25, 'is_student': True}
    """
    args = {}
    for text_arg in unparsed:
        if '=' not in text_arg:
            raise ValueError(f"Invalid command line argument: {text_arg}, please add '=' to separate key and value.")
        key, value = text_arg.split('=')
        key = key[len('--'):]
        try:
            value = eval(value)
        except:
            pass
        args[key] = value
    return args


def download_file(url: str, path: str) -> None:
    """
    Downloads a file from the given URL and saves it to the specified path.

    Args:
        url (str): The URL of the file to download.
        path (str): The path where the downloaded file will be saved.
    """
    logger = getLogger()
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded {os.path.basename(path)}")
    else:
        logger.error(f"Failed to download {os.path.basename(path)}")


def list_to_str(l: Union[list, str], remove_blank=False) -> str:
    """
    Converts a list or a string to a string representation.

    Args:
        l (Union[list, str]): The input list or string.

    Returns:
        str: The string representation of the input.
    """
    ret = ''
    if isinstance(l, list):
        ret = ', '.join(map(str, l))
    else:
        ret = l
    if remove_blank:
        ret = ret.replace(' ', '')
    return ret


def clean_text(raw_text: str) -> str:
    """
    Cleans the raw text by removing HTML tags, special characters, and extra spaces.

    Args:
        raw_text (str): The raw text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text=re.sub(r'[^\x00-\x7F]', ' ', text)
    return text

def init_device():
    """
    Set the visible devices for training. Supports multiple GPUs.

    Returns:
        torch.device: The device to use for training.

    """
    import torch
    use_ddp = True if os.environ.get("WORLD_SIZE") else False # Check if DDP is enabled
    if torch.cuda.is_available():
        return torch.device('cuda'), use_ddp
    else:
        return torch.device('cpu'), use_ddp

def config_for_log(config: dict) -> dict:
    config = config.copy()
    config.pop('device', None)
    config.pop('accelerator', None)
    
    # Filter out unsupported data types and keep only basic types
    filtered_config = {}
    for k, v in config.items():
        if isinstance(v, (int, float, str, bool)):
            filtered_config[k] = v
        elif isinstance(v, list):
            # Convert list to string
            filtered_config[k] = str(v)
        elif v is None:
            filtered_config[k] = "None"
        else:
            # Skip other complex types
            continue
    
    return filtered_config


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
