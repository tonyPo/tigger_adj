from copy import deepcopy
import torch
import numpy as np
import pandas as pd
from tigger_package.tab_ddpm.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from tigger_package.tab_ddpm.modules import MLPDiffusion
from tigger_package.tab_ddpm.lib import prepare_fast_dataloader, Dataset


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


class Tab_ddpm_controller:
    """instantiates model and manages training and sampling"""
    def __init__(self, config_path, config_dict) -> None:
        self.config_path = config_path + config_dict['ddpm_config_path']
        for key, val in config_dict.items():
            setattr(self, key, val)
            
        self.model = None # diffusion model reverse process
        self.diffusion = None  # diffision model forward + reverse process
        self.dataset = None  # dataset object
 
    def train(self, embed, nodes):

        self.dataset = Dataset.make_dataset(
            nodes = nodes,
            embed = embed,
            dataset_config = self.dataset_params,
            model_config = self.model_params,
        )

        # determine number of category dimensions
        K = np.array(self.dataset.get_category_sizes('train'))
        if len(K) == 0:
            K = np.array([0])
        print(f"Category dims K: {K}")

        num_numerical_features = self.dataset.X_num['train'].shape[1] if self.dataset.X_num is not None else 0
        d_in = np.sum(K) + num_numerical_features
        self.model_params['d_in'] = d_in
        print(f"Total number of dim in: {d_in}")
        
        print(self.model_params)
        self.model = MLPDiffusion(**self.model_params) # create the denoising / reverse diffusion process
        self.model.to(self.device)

        train_loader = prepare_fast_dataloader(self.dataset, split='train', batch_size=self.batch_size)

        # create diffusion model with both forward as backward 
        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=K,
            num_numerical_features=num_numerical_features,
            denoise_fn=self.model,
            gaussian_loss_type=self.gaussian_loss_type,
            num_timesteps=self.num_timesteps,
            scheduler=self.scheduler,
            device=self.device
        )
        self.diffusion.to(self.device)
        self.diffusion.train()
        

        trainer = Trainer(
            self.diffusion,
            train_loader,
            lr=self.lr,
            weight_decay=self.weight_decay,
            steps=self.steps,
            device=self.device
        )
        trainer.run_loop()
        
        if self.verbose >= 2:
            self.plot_history(trainer.loss_history)

        # trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
        # torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
        # torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))
        return ("no_model", trainer.loss_history)


    def sample_model(self, num_samples, name=None):
        """ samples new nodes"""
        empirical_class_dist = torch.tensor([1])  # only 1 class
        synth_node_cat, synth_node_num, synth_embed = None, None, None
        x_gen, y_gen = self.diffusion.sample_all(num_samples, self.batch_size, empirical_class_dist.float(), ddim=False)
        X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

        num_numerical_features = self.dataset.X_num['train'].shape[1] if self.dataset.X_num is not None else 0

        X_num_ = X_gen
        if num_numerical_features < X_gen.shape[1]:
            # use goodode funciton to map to one hot
            X_cat = X_num_[:, num_numerical_features:]
            synth_node_cat = pd.DataFrame(X_cat)

        if num_numerical_features != 0:
            X_num = X_num_[:, :num_numerical_features]
            synth_embed = pd.DataFrame(X_num[:,:self.dataset.embed_dim])
            synth_node_num = pd.DataFrame(X_num[:,self.dataset.embed_dim:], columns=self.dataset.num_cols)

        synth_node = pd.concat([d for d in [synth_node_cat, synth_node_num] if d is not None], axis=1) 
        #reset to original order
        synth_node = synth_node[self.dataset.cols]

        synth_nodes = pd.concat([synth_embed, synth_node], axis=1) 
        synth_nodes = self.post_process(synth_nodes)
        synth_nodes.to_parquet(name)     

    @staticmethod    
    def to_good_ohe(ohe, X):
        indices = np.cumsum([0] + ohe._n_features_outs)
        Xres = []
        for i in range(1, len(indices)):
            x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
            t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
            Xres.append(np.where(t >= 0, 1, 0))
        return np.hstack(Xres)
    
    def post_process(self, synth_nodes):
        """Some samples are extremely off with a factor higher then 1000. These samples are 
        unrealistic and therefore every datapoint outside the boundary [-1, 2] is discarded.
        Note that input is normalised between 0 and 1. 
        the remaining points that are outside the boundary [0,1] are adjust to fit them
        in the range [0,1]
        """
        for c in synth_nodes.columns:
            discard = synth_nodes[(synth_nodes[c]> 2) | (synth_nodes[c]< -1)]
            synth_nodes = synth_nodes.drop(discard.index)
            
            adjust_upper = synth_nodes[(synth_nodes[c]> 1)]
            synth_nodes.loc[adjust_upper.index, c] = 1
            
            adjust_lower = synth_nodes[(synth_nodes[c]< 0)]
            synth_nodes.loc[adjust_lower.index, c] = 0
        return synth_nodes
     
    def load_model(self, name):
        raise("not implemented")
    
    def plot_history(self, hist):
        hist.loss.plot()
    