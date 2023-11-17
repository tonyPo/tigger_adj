from copy import deepcopy
import torch
import os
import numpy as np
from tigger_package.tab_ddpm.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from tigger_package.tab_ddpm.utils_train_tab_ddpm import get_model
from tigger_package.tab_ddpm.lib import prepare_fast_dataloader, Dataset
import pandas as pd

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=torch.device('cuda:1')):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss'])
        self.log_every = 100
        self.print_every = 500
        self.ema_every = 1000

    def _anneal_lr(self, step):
        frac_done = step / self.steps
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, out_dict):
        x = x.to(self.device)
        for k in out_dict:
            out_dict[k] = out_dict[k].long().to(self.device)
        self.optimizer.zero_grad()
        loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
        loss = loss_multi + loss_gauss
        loss.backward()
        self.optimizer.step()

        return loss_multi, loss_gauss

    def run_loop(self):
        step = 0
        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0

        curr_count = 0
        while step < self.steps:
            x, out_dict = next(self.train_iter)
            out_dict = {'y': out_dict}
            batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_multi += batch_loss_multi.item() * len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)

            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)
                gloss = np.around(curr_loss_gauss / curr_count, 4)
                if (step + 1) % self.print_every == 0:
                    print(f'Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss}')
                self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            self.update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

            step += 1
            
    def update_ema(self, target_params, source_params, rate=0.999):
        """
        Update target parameters to be closer to those of source parameters using
        an exponential moving average.
        :param target_params: the target parameter sequence.
        :param source_params: the source parameter sequence.
        :param rate: the EMA rate (closer to 1 means slower).
        """
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(rate).add_(src.detach(), alpha=1 - rate)

def train(config,  path, nodes, embed=None):

    dataset = Dataset.make_dataset(
        nodes = nodes,
        embed = embed,
        dataset_config = config['dataset_params'],
        model_config = config['model_params'],
    )

    # determine number of category dimensions
    K = np.array(dataset.get_category_sizes('train'))
    if len(K) == 0:
        K = np.array([0])
    print(f"Category dims K: {K}")

    num_numerical_features = dataset.X_num['train'].shape[1] if dataset.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features
    config['model_params']['d_in'] = d_in
    print(f"Total number of dim in: {d_in}")
    
    print(config['model_params'])
    model = get_model(
        'mlp',  # no other type implemented
        config['model_params'],
        num_numerical_features,
        category_sizes=dataset.get_category_sizes('train')
    )  # create the denoising / reverse diffusion process
    model.to(config['device'])

    train_loader = prepare_fast_dataloader(dataset, split='train', batch_size=config['batch_size'])

    # create diffusion model with both forward as backward 
    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        gaussian_loss_type=config['gaussian_loss_type'],
        num_timesteps=config['num_timesteps'],
        scheduler=config['scheduler'],
        device=config['device']
    )
    diffusion.to(config['device'])
    diffusion.train()

    trainer = Trainer(
        diffusion,
        train_loader,
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        steps=config['steps'],
        device=config['device']
    )
    trainer.run_loop()

    # trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
    # torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
    # torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))
    return trainer.loss_history