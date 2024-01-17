import argparse


def str2bool(v):
    return v.lower() in ['true']


def get_GAN_config():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lm_model', type=str, default='roberta-base', help='LM model')
    parser.add_argument('--N', type=int, default=50, help='max number of nodes')
    parser.add_argument('--max_len', type=int, default=128, help='max number of tokens input to LM')
    parser.add_argument('--z_dim', type=int, default=8, help='dimension of latent vector')
    parser.add_argument('--mha_dim', type=int, default=768, help='dimension of vectors uses in multi-head attentin')
    parser.add_argument('--n_heads', type=int, default=8, help='number of heads to be used in multi-head attention')
    parser.add_argument('--gen_dims', default=[[128, 256, 768], [512, 512]], help='hidden dimensions of MLP layer in G before and after attention')
    parser.add_argument('--disc_dims', default=[[128, 128], [512, 768], [512, 256, 128]], help='hidden dimensions of MLP layer in D before and after attention')
    parser.add_argument('--lambda_gp', type=float, default=5, help='weight for gradient penalty')
    parser.add_argument('--lambda_rew', type=float, default=0.5, help='weight for reward loss')
    parser.add_argument('--lambda_wgan', type=float, default=1, help='whether or not to use wgan loss')
    parser.add_argument('--post_method', type=str, default='hard_gumbel', choices=['sigmoid', 'soft_gumbel', 'hard_gumbel'])

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs for training D')
    parser.add_argument('--g_lr', type=float, default=2e-4, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=2e-4, help='learning rate for D')
    parser.add_argument('--b_lr', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    parser.add_argument('--n_critic', type=int, default=4, help='number of D updates per each G update')

    # Test configuration.
    parser.add_argument('--test_epochs', type=int, default=100, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--bert_unfreeze', type=int, default=0)

    # Use either of these two datasets.
    parser.add_argument('--data_dir', type=str, default='data/graphgen')

    # Directories.
    parser.add_argument('--saving_dir', type=str, default='results')

    # Step size.
    parser.add_argument('--model_save_step', type=int, default=20)
    parser.add_argument('--lr_update_step', type=int, default=1000)
    
    # For training
    config = parser.parse_args()
    # Pass the path to the model checkpoints here to restore training from those checkpoints
    config.restore_G = None
    config.restore_D = None
    config.restore_B_D = None
    config.restore_B_G = None
    
    # Modes for Model and Dataset
    # Model Modes:
    #   - mode 0: using multi head attention on the hidden vector and directly use it for the subsequent layers
    #   - mode 1: using multi head attention on the hidden vector and concat it to the hidden vector for subsequent layers
    #   - mode 2: using cls token's embedding from LM to concat to the hidden vector for subsequent layers
    #
    # Dataset Modes:
    #   - ds_mode 0: text with number in numeric format
    #   - ds_mode 1: text with number in text format
    #   - ds_mode 2: just the number in numeric format
    #
    config.model_mode = 1
    config.ds_mode = 0

    # Wandb
    config.name = 'wandb_name'
    
    # Involve bert unfreeze
    config.bert_unfreeze = 0
    
    # For testing
    config.test_category_wise = 1

    return config