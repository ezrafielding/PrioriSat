from collections import defaultdict

import os
import time
import datetime

import torch
import torch.nn.functional as F
from transformers import BertModel, RobertaModel
from score import score

from models_gan import Generator, Discriminator, gumbel_sigmoid
from graph_data import get_loaders, SyntheticGraphDataset
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, '../GraphGen')

import wandb

class Solver(object):
    """Solver for training and testing LIC-GAN."""

    def __init__(self, config, log=None):
        """Initialize configurations."""
        print(config.batch_size)
        # Log
        self.log = log

        # Data loader.
        self.ds_mode = config.ds_mode
        self.train_data, self.val_data, self.test_data = get_loaders(config.data_dir, 
                                                                    config.N, 
                                                                    config.max_len, 
                                                                    config.lm_model, 
                                                                    config.batch_size,
                                                                    num_workers=1,
                                                                    ds_mode=self.ds_mode)

        # Model configurations.
        self.N = config.N
        self.z_dim = config.z_dim
        self.mha_dim = config.mha_dim
        self.n_heads = config.n_heads
        self.gen_dims = config.gen_dims
        self.disc_dims = config.disc_dims
        self.la = config.lambda_wgan
        self.la_gp = config.lambda_gp
        self.la_rew = config.lambda_rew
        self.post_method = config.post_method
        self.model_mode = config.model_mode
        
        self.lm_model = config.lm_model
        self.max_len = config.max_len

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_epochs = config.num_epochs
        self.num_steps = len(self.train_data)
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.b_lr = config.b_lr
        self.dropout = config.dropout
        self.n_critic = config.n_critic
        self.lr_update_step = config.lr_update_step
        
        # Training or testing.
        self.mode = config.mode

        # Miscellaneous.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bert_unfreeze = config.bert_unfreeze
        print('Device gan: ', self.device)
        print('Mode: ', self.mode)
        print('Wandb Name: ', config.name)
        self.test_category_wise = config.test_category_wise

        # Directories.
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir

        # Step size.
        self.model_save_step = config.model_save_step

        # Build the model.
        self.build_model()
        self.restore_G = config.restore_G
        self.restore_D = config.restore_D
        self.restore_B_G = config.restore_B_G
        self.restore_B_D = config.restore_B_D
        
        if self.mode == 'train':
            self.run = wandb.init(
            # Set the project where this run will be logged
                name=config.name,
                project="pgm-proj",
                # Track hyperparameters and run metadata
                config={
                    key: val for key, val in config.__dict__.items() if not key.startswith('__') and not callable(key) and not key.endswith('dir')
                }
            )
            for metric in ['l_D/R', 'l_D/F', 'l_D', 'l_G', 'l_D_gp', 'l_R', 'l_R/N', 'l_R/M']:
                self.run.define_metric(f'train/{metric}', step_metric="step")
            for metric in ['l_D/R', 'l_D/F', 'l_D', 'l_G', 'l_D_gp', 'property_match', 'closeness', 'l_R', 'l_R/N', 'l_R/M',\
                           'n_match', 'm_match', 'min_deg_match', 'max_deg_match', 'diam_match', 'cc_match', 'cycle_match']:
                self.run.define_metric(f'val/{metric}', step_metric="epoch")


    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.N,
                           self.z_dim,
                           self.gen_dims,
                           self.mha_dim,
                           self.n_heads,
                           self.dropout,
                           self.model_mode)
        self.D = Discriminator(self.N,
                               self.disc_dims, 
                               self.mha_dim,
                               self.n_heads,
                               self.dropout,
                               self.model_mode)

        if 'roberta' in self.lm_model:
            self.bert_D = RobertaModel.from_pretrained(self.lm_model)
            self.bert_G = RobertaModel.from_pretrained(self.lm_model)
        elif 'bert' in self.lm_model:
            self.bert_D = BertModel.from_pretrained(self.lm_model)
            self.bert_G = BertModel.from_pretrained(self.lm_model)
        else:
            raise ValueError('Invalid LM model')
        for name, param in self.bert_D.named_parameters():
            if self.bert_unfreeze == 0 or 'pooler' not in name:
                param.requires_grad = False
        for name, param in self.bert_G.named_parameters():
            if self.bert_unfreeze == 0 or 'pooler' not in name:
                param.requires_grad = False

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, betas=(0, 0.9))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, betas=(0, 0.9))
        if self.bert_unfreeze:
            self.b_d_optimizer = torch.optim.Adam(self.bert_D.parameters(), self.b_lr, betas=(0, 0.9))
            self.b_g_optimizer = torch.optim.Adam(self.bert_G.parameters(), self.b_lr, betas=(0, 0.9))
        # self.g_scheduler = torch.optim.lr_scheduler.LinearLR(self.g_optimizer,
        #                                                      1.,
        #                                                      1./self.num_epochs,
        #                                                      self.num_epochs)
        # self.d_scheduler = torch.optim.lr_scheduler.LinearLR(self.d_optimizer,
        #                                                      1.,
        #                                                      1./self.num_epochs,
        #                                                      self.num_epochs)
        # self.r_scheduler = torch.optim.lr_scheduler.LinearLR(self.r_optimizer,
        #                                                      1.,
        #                                                      1./self.num_epochs,
        #                                                      self.num_epochs)
        self.print_network(self.G, 'G', self.log)
        self.print_network(self.D, 'D', self.log)
        self.print_network(self.bert_G, self.lm_model+'_G', self.log)
        self.print_network(self.bert_D, self.lm_model+'_D', self.log)

        self.G.to(self.device)
        self.D.to(self.device)
        self.bert_G.to(self.device)
        self.bert_D.to(self.device)
        
    @staticmethod
    def print_network(model, name, log=None):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        if log is not None:
            log.info(model)
            log.info(name)
            log.info("The number of parameters: {}".format(num_params))

    # def restore_model(self, resume_iters):
    #     """Restore the trained generator and discriminator."""
    #     print('Loading the trained models from step {}...'.format(resume_iters))
    #     G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(resume_iters))
    #     D_path = os.path.join(self.model_dir, '{}-D.ckpt'.format(resume_iters))
    #     self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    #     self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def update_lr(self, g_lr, d_lr, r_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
        if self.bert_unfreeze:
            for param_group in self.b_d_optimizer.param_groups:
                param_group['lr'] = b_lr
            for param_group in self.b_g_optimizer.param_groups:
                param_group['lr'] = b_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        if self.bert_unfreeze:
            self.b_d_optimizer.zero_grad()
            self.b_g_optimizer.zero_grad()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size()) + [dim]).to(self.device)
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.)
        return out

    def sample_z(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    @staticmethod
    def postprocess(inputs, method, temperature=1.):
        def listify(x):
            return x if type(x) == list or type(x) == tuple else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == 'soft_gumbel':
            softmax = [gumbel_sigmoid(e_logits, t=temperature, hard=False)
                       for e_logits in listify(inputs)]
        elif method == 'hard_gumbel':
            softmax = [gumbel_sigmoid(e_logits, t=temperature, hard=True)
                       for e_logits in listify(inputs)]
        elif method == 'sigmoid':
            softmax = [F.sigmoid(e_logits / temperature)
                       for e_logits in listify(inputs)] 
        else:
            raise NotImplementedError
        

        return delistify([e for e in (softmax)])

    def train_and_validate(self):
        self.start_time = time.time()

        # Start training from scratch or resume training.
        start_epoch = 0
        if self.restore_D:
            self.D.load_state_dict(torch.load(self.restore_D, map_location=lambda storage, loc: storage))
        if self.restore_G:
            self.G.load_state_dict(torch.load(self.restore_G, map_location=lambda storage, loc: storage))
        if self.restore_B_G:
            self.bert_G.pooler.load_state_dict(torch.load(self.restore_B_G, map_location=lambda storage, loc: storage))
        if self.restore_B_D:
            self.bert_D.pooler.load_state_dict(torch.load(self.restore_B_D, map_location=lambda storage, loc: storage))

        # Start training.
        if self.mode == 'train':
            print('Validating before training...')
            self.train_or_valid(epoch_i=-1, train_val_test='val')
            
            print('Start training...')
            for i in range(start_epoch, self.num_epochs):
                self.train_or_valid(epoch_i=i, train_val_test='train')
                self.train_or_valid(epoch_i=i, train_val_test='val')
                if (i+1) % self.model_save_step == 0:
                    self.train_or_valid(epoch_i=i, train_val_test='test')
        
                # self.g_scheduler.step()
                # self.d_scheduler.step()
                # if i == start_epoch:
                #     self.la = 1
            wandb.finish()
        elif self.mode == 'test':
            # assert self.resume_epoch is not None
            self.train_or_valid(epoch_i=start_epoch, train_val_test='test')
        else:
            raise NotImplementedError

    def get_gen_adj_mat(self, adj_mat, method=None):
        if self.post_method == 'hard_gumbel':
            return adj_mat
        
        if method is not None:
            adj_mat = self.postprocess(adj_mat, method)
        adj_mat = torch.nan_to_num(adj_mat, nan=0., posinf=0., neginf=0.)
        adj_mat = (adj_mat + adj_mat.permute(0, 2, 1)) / 2
        adj_mat = torch.round(adj_mat)
        assert torch.all(torch.eq(adj_mat, adj_mat.permute(0, 2, 1)))
        return adj_mat

    def save_checkpoints(self, epoch_i):
        G_path = os.path.join(self.model_dir, '{}-G.ckpt'.format(epoch_i + 1))
        D_path = os.path.join(self.model_dir, '{}-D.ckpt'.format(epoch_i + 1))
        B_D_path = os.path.join(self.model_dir, '{}-B_D.ckpt'.format(epoch_i + 1))
        B_G_path = os.path.join(self.model_dir, '{}-B_G.ckpt'.format(epoch_i + 1))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        if self.bert_unfreeze:
            torch.save(self.bert_D.pooler.state_dict(), B_D_path)
            torch.save(self.bert_G.pooler.state_dict(), B_G_path)
        print('Saved model checkpoints into {}...'.format(self.model_dir))
        if self.log is not None:
            self.log.info('Saved model checkpoints into {}...'.format(self.model_dir))

    def train_or_valid(self, epoch_i, train_val_test='val'):
        # The first several epochs using RL to purse stability (not used).
        # if epoch_i < 0:
        #     cur_la = 0
        # else:
        cur_la = self.la

        # Recordings
        losses = defaultdict(list)
        scores = defaultdict(list)
        mask_props = defaultdict(list)
        node_cat = []

        # Iterations
        the_step = self.num_steps
        if train_val_test == 'val':
            the_step = len(self.val_data)
            print('[Validating]')
        if train_val_test == 'test':
            print('[Testing]')
            the_step = len(self.test_data)

        val_data_iter = iter(self.val_data)
        test_data_iter = iter(self.test_data)
        train_data_iter = iter(self.train_data)
        for a_step in tqdm(range(the_step)):
            if train_val_test == 'val':
                adj_mat, node_inp, ids, mask, desc, props = next(val_data_iter)
                z = self.sample_z(adj_mat.shape[0])
            elif train_val_test == 'test':
                adj_mat, node_inp, ids, mask, desc, props = next(test_data_iter)
                z = self.sample_z(adj_mat.shape[0])
            elif train_val_test == 'train':
                adj_mat, node_inp, ids, mask, desc, props = next(train_data_iter)
                z = self.sample_z(adj_mat.shape[0])
            else:
                raise NotImplementedError
            
            if train_val_test == 'train':
                self.reset_grad()

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            adj_mat = adj_mat.to(self.device)
            node_inp = node_inp.to(self.device)
            ids = ids.to(self.device)
            mask = mask.to(self.device)
            z = torch.from_numpy(z).to(self.device).float()

            # Current steps
            cur_step = self.num_steps * epoch_i + a_step
            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #
        
            # Compute losses with real inputs.
            if train_val_test != 'train':
                with torch.no_grad():
                    bert_D_out = self.bert_D(ids, attention_mask=mask).last_hidden_state[:,:self.N,:]
                    logits_real, features_real = self.D(adj_mat, node_inp, bert_D_out)
                    # Z-to-target
                    if self.bert_unfreeze:
                        bert_G_out = self.bert_G(ids, attention_mask=mask).last_hidden_state[:,:self.N,:]
                    else:
                        bert_G_out = bert_D_out
                    adjM_logits, node_logits = self.G(z, bert_G_out)
            else:
                bert_D_out = self.bert_D(ids, attention_mask=mask).last_hidden_state[:,:self.N,:]
                logits_real, features_real = self.D(adj_mat, node_inp, bert_D_out)
                # Z-to-target
                if self.bert_unfreeze:
                    bert_G_out = self.bert_G(ids, attention_mask=mask).last_hidden_state[:,:self.N,:]
                else:
                    bert_G_out = bert_D_out
                adjM_logits, node_logits = self.G(z, bert_G_out)
        
            # Postprocess with sigmoid
            adjM_hat = self.postprocess(adjM_logits, self.post_method)
            node_hat = self.postprocess(node_logits, self.post_method)
            # node_mask = node_hat.view(node_hat.size(0), -1, 1) @ node_hat.view(node_hat.size(0), 1, -1)
            # adjM_hat = adjM_hat * node_mask
            if train_val_test != 'train':
                with torch.no_grad():
                    logits_fake, features_fake = self.D(adjM_hat, node_hat, bert_D_out)
            else:
                logits_fake, features_fake = self.D(adjM_hat, node_hat, bert_D_out)

            # Compute losses for gradient penalty.
            eps = torch.rand(logits_real.size(0), 1, 1).to(self.device)
            x_int0 = (eps * adj_mat + (1. - eps) * adjM_hat).requires_grad_(True)
            eps = eps.view(-1, 1)
            y_int0 = (eps * node_inp + (1. - eps) * node_hat).requires_grad_(True)
            grad0, grad1 = self.D(x_int0, y_int0, bert_D_out)
            grad_penalty = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(grad0, y_int0)

            d_loss_real = torch.mean(logits_real)
            d_loss_fake = torch.mean(logits_fake)
            loss_D = -d_loss_real + d_loss_fake + self.la_gp * grad_penalty

            losses['l_D/R'].append(d_loss_real.item())
            losses['l_D/F'].append(d_loss_fake.item())
            losses['l_D'].append(loss_D.item())

            # Optimise discriminator.
            if train_val_test == 'train':
                loss_D.backward(retain_graph=True)
                self.d_optimizer.step()
                if self.bert_unfreeze:
                    self.b_d_optimizer.step()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #
            self.reset_grad()
            
            # Z-to-target
            if self.bert_unfreeze:
                bert_G_out = self.bert_G(ids, attention_mask=mask).last_hidden_state[:,:self.N,:]
            adjM_logits, node_logits = self.G(z, bert_G_out)
            # Postprocess with sigmoid
            node_hat = self.postprocess(node_logits, self.post_method)
            adjM_hat = self.postprocess(adjM_logits, self.post_method)
            # node_mask = node_hat.view(node_hat.size(0), -1, 1) @ node_hat.view(node_hat.size(0), 1, -1)
            # adjM_hat = adjM_hat * node_mask
            if self.bert_unfreeze:
                bert_D_out = self.bert_D(ids, attention_mask=mask).last_hidden_state[:,:self.N,:]
            logits_fake, features_fake = self.D(adjM_hat, node_hat, bert_D_out)
            
            # Reward Losses
            node_pred = node_hat.sum(dim=1)/self.N # in [0, 1], represent percentage 
            # nodes_pred shape: [abtch_size,]
            # assume undirected graph adj_mat is symmetric
            # `adj_mat` has shape [batch_size, self.N, self.N]
            node_true = node_inp.sum(dim=1)/self.N
            #torch.tensor([/self.N for p in props], dtype=torch.float).to(self.device)
            node_loss = F.mse_loss(node_pred, node_true)
            
            edge_loss = (adj_mat.sum(dim=(1,2)) - adjM_hat.sum(dim=(1,2))) / (self.N * (self.N - 1))
            edge_loss = (edge_loss ** 2).mean()
            
            # Reward loss
            loss_R = node_loss + edge_loss

            # Losses Update
            loss_G = -logits_fake


            loss_G = torch.mean(loss_G)
            # loss_V = torch.mean(loss_V)
            # loss_RL = torch.mean(loss_RL)
            losses['l_G'].append(loss_G.item())
            losses['l_R'].append(loss_R.item())
            losses['l_R/N'].append(node_loss.item())
            losses['l_R/M'].append(edge_loss.item())
            
            if train_val_test == 'train':
                if train_val_test == 'train':
                    wandb.log({
                        f'step': cur_step+1,
                        f'{train_val_test}/l_D/R': d_loss_real.item(), 
                        f'{train_val_test}/l_D/F': d_loss_fake.item(), 
                        f'{train_val_test}/l_D': loss_D.item(),
                        f'{train_val_test}/l_G': loss_G.item(),
                        f'{train_val_test}/l_R': loss_R.item(),
                        f'{train_val_test}/l_R/N': node_loss.item(),
                        f'{train_val_test}/l_R/M': edge_loss.item(),
                        f'{train_val_test}/l_D/GP': grad_penalty.item(),
                    })
                
            # train_step_V = loss_V
            if train_val_test == 'train' and cur_step % self.n_critic == 0:
                # Optimise generator.
                loss_G.backward(retain_graph=True)
                self.g_optimizer.step()
                if self.bert_unfreeze:
                    self.b_g_optimizer.step()
            
            # optimise the rewardnet
            if train_val_test == 'train' and self.la_rew > 0:
                calc_loss_R = self.la_rew * loss_R
                calc_loss_R.backward()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            # Get scores.
            if train_val_test in ['val', 'test']:
                # torch.cuda.empty_cache()
                if self.mode == 'test' or (epoch_i + 1) % 2 == 0:
                    node_mask = node_hat.view(node_hat.size(0), -1, 1) @ node_hat.view(node_hat.size(0), 1, -1)
                    adjM_hat = adjM_hat * node_mask
                    mats = self.get_gen_adj_mat(adjM_hat)
                    np_mats = mats.detach().cpu().numpy().astype(int)
                    np_nodes = node_hat.detach().cpu().numpy().astype(int)
                    results, m_props = score(props, np_mats, np_nodes)
                    for k, v in m_props.items():
                        mask_props[k].extend(v)
                    for k, v in results.items():
                        scores[k].extend(v)
                    node_cat.extend(node_inp.detach().cpu().numpy().sum(axis=1).tolist())
                        
                if a_step +1 == the_step:
                    if self.mode != 'test' and (epoch_i + 1) % 2 != 0:
                        mats = self.get_gen_adj_mat(adjM_hat, self.post_method)
                        np_mats = mats.detach().cpu().numpy().astype(int)
                        np_nodes = node_hat.detach().cpu().numpy().astype(int)
                    log = '5 sample adjacenecy matrices\n'
                    for i in range(5):
                        log += '-'*50 + '\n'
                        log += 'Text: {}\n'.format(desc[i])
                        res = [SyntheticGraphDataset._get_eval_str_fn()[j](np_mats[i], np_nodes[i]) for j in range(7)]
                        log += 'Results: {}\n'.format(res)
                        log += '-'*50 + '\n'
                    if self.log is not None:
                        self.log.info(log)

                    # Save checkpoints.
                    if self.mode == 'train':
                        if (epoch_i + 1) % self.model_save_step == 0:
                            self.save_checkpoints(epoch_i=epoch_i)

                    # Print out training information.
                    et = time.time() - self.start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}]:".format(et, epoch_i + 1, self.num_epochs)

                    is_first = True
                    new_dict = {'epoch': epoch_i + 1}
                    for tag, value in losses.items():
                        if is_first:
                            log += "\n{}: {:.2f}".format(tag, np.mean(value))
                            is_first = False
                        else:
                            log += ", {}: {:.2f}".format(tag, np.mean(value))
                        if self.mode == 'train':
                            new_dict[f'{train_val_test}/{tag}'] = np.mean(value)
    
                    if self.mode == 'test' or (epoch_i + 1) % 2 == 0:
                        is_first = True
                        for tag, value in scores.items():
                            if tag in mask_props:
                                res = np.sum(value) / max(np.sum(mask_props[tag]), 1e-10)
                            else:
                                res = np.mean(value)
                            if is_first:
                                log += "\n{}: {:.2f}".format(tag, res)
                                is_first = False
                            else:
                                log += ", {}: {:.2f}".format(tag, res)
                            if self.mode == 'train':
                                new_dict[f'{train_val_test}/{tag}'] = res
                    
                    if self.mode == 'test' and self.test_category_wise:
                        categories = [0, 5, 10, 25, 50]
                        log += '\n' + '-'*50
                        for c in range(1, len(categories)):
                            c_mask = np.where((np.array(node_cat) <= categories[c]) & (np.array(node_cat) > categories[c-1]), 1, 0)
                            log += f"\nNodes in ({categories[c-1]}, {categories[c]}]: {np.sum(c_mask)}\n"
                            is_first = True
                            for tag, value in scores.items():
                                if tag in mask_props:
                                    new_mask = np.array(mask_props[tag]) * c_mask
                                else:
                                    new_mask = c_mask
                            
                                res = np.sum(np.array(value)*c_mask) / max(np.sum(new_mask), 1e-10)
                                
                                if is_first:
                                    log += "{}: {:.2f}".format(tag, res)
                                    is_first = False
                                else:
                                    log += ", {}: {:.2f}".format(tag, res)
                        log += '\n' + '-'*50
                        
                    if self.mode == 'train':
                        wandb.log(new_dict)
                        
                    print(log)

                    if self.log is not None:
                        self.log.info(log)

