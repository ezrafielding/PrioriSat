import os
import logging
import dotenv

dotenv.load_dotenv('./.env')

from rdkit import RDLogger

from args import get_GAN_config
import wandb

os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY") # Set your key here for wandb logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Remove flooding logs.
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from solver_gan import Solver
from torch.backends import cudnn
import datetime

def get_date_postfix(time=False):
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}'.format(dt.date())
    if time:
        post_fix = '{}_{:02d}-{:02d}'.format(
            dt.date(), dt.hour, dt.minute)

    return post_fix

def main(config):
    # For fast training.
    cudnn.benchmark = True
    
    # Tokenizer warning disable
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    # Timestamp
    config.saving_dir = os.path.join(config.saving_dir, get_date_postfix(True))
    config.log_dir = os.path.join(config.saving_dir, 'logs')
    config.model_dir = os.path.join(config.saving_dir, 'models')

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    # Logger
    if config.mode in 'train':
        log_p_name = os.path.join(config.log_dir, get_date_postfix(True) + '_test_logger.log' if config.mode == 'test' else '_logger.log')
        logging.basicConfig(filename=log_p_name, level=logging.INFO)
        logging.info(config)

    # Solver for training and testing StarGAN.
    if config.mode == 'train':
        solver = Solver(config, logging)
    elif config.mode == 'test':
        solver = Solver(config, None)
    else:
        raise NotImplementedError

    solver.train_and_validate()


if __name__ == '__main__':
    wandb.login()
    config = get_GAN_config()

    main(config)
