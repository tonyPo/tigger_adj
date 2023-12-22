#%%
from copy import deepcopy
import torch
import numpy as np
import pandas as pd
if __name__ == "__main__":
    import os
    os.chdir('../..')
    print(os.getcwd())
    import yaml
from tigger_package.tab_ddpm.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion
from tigger_package.tab_ddpm.modules import MLPDiffusion
from tigger_package.tab_ddpm.lib import prepare_fast_dataloader, Dataset
#%%


class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, steps, device=torch.device('cuda:1'),
                 valiation_iter=None):
        self.diffusion = diffusion
        self.ema_model = deepcopy(self.diffusion._denoise_fn)
        for param in self.ema_model.parameters():
            param.detach_()

        self.train_iter = train_iter
        self.validation_iter = valiation_iter
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.loss_history = pd.DataFrame(columns=['step', 'mloss', 'gloss', 'loss', 'val_loss'])
        self.log_every = 100
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
            for x, out_dict in iter(self.train_iter):
                out_dict = {'y': out_dict}
                batch_loss_multi, batch_loss_gauss = self._run_step(x, out_dict)
                self.update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters())

                self._anneal_lr(step)  # adjust learning rate

                curr_count += len(x)  # Add batch size to counter
                curr_loss_multi += batch_loss_multi.item() * len(x)  # add multinominal loss to counter
                curr_loss_gauss += batch_loss_gauss.item() * len(x)  # add categorical loss to counter
                
            if (step + 1) % self.log_every == 0:
                mloss = np.around(curr_loss_multi / curr_count, 4)  # calculate avertage multinominal loss
                gloss = np.around(curr_loss_gauss / curr_count, 4)  # calculate average categorical loss
                val_loss = self.run_validation_step()
                
                if (step + 1) % self.log_every == 0:
                    print(f'\r Step {(step + 1)}/{self.steps} MLoss: {mloss} GLoss: {gloss} Sum: {mloss + gloss} val:{val_loss:.4f}', end="")
                
                self.loss_history.loc[len(self.loss_history)] =[step + 1, mloss, gloss, mloss + gloss, val_loss]
                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_multi = 0.0

            step += 1
            
    def run_validation_step(self):
        self.diffusion.eval()
        loss = 0
        cnt = 0
        out_dict = {}
        for x, out_val in iter(self.validation_iter):
            x = x.to(self.device)     
            out_dict['y'] = out_val.long().to(self.device)
            loss_multi, loss_gauss = self.diffusion.mixed_loss(x, out_dict)
            loss += (loss_multi.item() + loss_gauss.item()) * len(x)
            cnt += len(x)
            
        self.diffusion.train()    
        return loss / cnt
            
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
    def __init__(self, embed, nodes, config_path, config_dict) -> None:
        self.config_path = config_path + config_dict['ddpm_config_path']
        for key, val in config_dict.items():
            setattr(self, key, val)
            
        self.model = None # diffusion model reverse process
        self.diffusion = None  # diffision model forward + reverse process
        self.dataset = None  # dataset object
        self.temp_cols = set()  # set with col id that are added for boolean cols.
        self.nodes = self.expand_boolean_cols(nodes)
        self.embed = embed
 
    def fit(self):
        self.dataset = Dataset.make_dataset(
            nodes = self.nodes,
            embed = self.embed,
            dataset_config = self.dataset_params,
            model_config = self.model_params,
        )

        # determine number of category dimensions
        num_cat = self.dataset.X_cat['train'].shape[1] if self.dataset.X_cat is not None else 0
        print(f"Category dims K: {num_cat}")

        num_numerical_features = self.dataset.X_num['train'].shape[1] if self.dataset.X_num is not None else 0
        d_in = num_cat + num_numerical_features
        self.model_params['d_in'] = d_in
        self.model_params['num_classes'] = num_numerical_features
        print(f"Total number of dim in: {d_in}")
        
        print(self.model_params)
        self.model = MLPDiffusion(**self.model_params) # create the denoising / reverse diffusion process
        self.model.to(self.device)

        train_loader = prepare_fast_dataloader(self.dataset, split='train', batch_size=self.batch_size)
        validation_loader = prepare_fast_dataloader(self.dataset, split='val', batch_size=self.batch_size)

        # create diffusion model with both forward as backward 
        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=self.dataset.cnts_per_cat,
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
            device=self.device,
            valiation_iter=validation_loader
        )
        trainer.run_loop()
        
        if self.verbose >= 2:
            self.plot_history(trainer.loss_history)

        # trainer.loss_history.to_csv(os.path.join(parent_dir, 'loss.csv'), index=False)
        # torch.save(diffusion._denoise_fn.state_dict(), os.path.join(parent_dir, 'model.pt'))
        # torch.save(trainer.ema_model.state_dict(), os.path.join(parent_dir, 'model_ema.pt'))
        return trainer.loss_history


    def sample_model(self, num_samples, name=None):
        """ samples new nodes"""
        empirical_class_dist = torch.tensor([1])  # only 1 class
        synth_node_cat, synth_node_num, synth_embed = None, None, None
        
        x_gen, y_gen = self.diffusion.sample_all(num_samples*2, self.batch_size, empirical_class_dist.float(), ddim=False)
        X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

        num_numerical_features = self.dataset.X_num['train'].shape[1] if self.dataset.X_num is not None else 0

        X_num_ = X_gen
        if num_numerical_features < X_gen.shape[1]:
            # use goodode funciton to map to one hot
            X_cat = X_num_[:, num_numerical_features:]
            cat_cols = sum(self.dataset_params['cat_cols'], [])
            synth_node_cat = pd.DataFrame(X_cat, columns =  cat_cols)

        if num_numerical_features != 0:
            X_num = X_num_[:, :num_numerical_features]
            synth_embed = pd.DataFrame(X_num[:,:self.dataset.embed_dim])
            synth_node_num = pd.DataFrame(X_num[:,self.dataset.embed_dim:], columns=self.dataset.num_cols)

        synth_node = pd.concat([d for d in [synth_node_cat, synth_node_num] if d is not None], axis=1) 
        #reset to original order
        synth_node = synth_node[self.dataset.cols]

        synth_nodes = pd.concat([synth_embed, synth_node], axis=1) 
        synth_nodes = self.post_process(synth_nodes)
        synth_nodes = synth_nodes.iloc[:num_samples,:]
        synth_nodes = self.remove_temp_cols(synth_nodes)
        synth_nodes.to_parquet(name) 
        return synth_nodes    

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
        hist[['loss', 'val_loss']].plot(logy=True)
        
    def expand_boolean_cols(self, nodes):
        for col in self.dataset_params.get('boolean_cols',[]):
            add_col_name = col + "_1"
            self.temp_cols.add(add_col_name)  # add cols to temp list
            self.dataset_params['cat_cols'].append([col, add_col_name])
            nodes[add_col_name] = nodes[col] ^ 1
        return nodes
    
    def remove_temp_cols(self, nodes):
        for col in self.temp_cols:
            nodes.drop(col, axis=1, inplace=True)
        return nodes
            
   
if __name__ == "__main__":
    node_df = pd.DataFrame([(i/10, (i+1)/10, int(i%2==0), int(i%3==0)) for i in range(100)],
                columns=['attr1','attr2', 'attr3', 'attr4'])
    embed_df = pd.DataFrame([(i/10, (i+1)/10) for i in range(100)],
                columns=['emb1','emb2'])
    
    config_dict = yaml.safe_load('''
        ddpm_config_path: ""
        verbose: 2
        num_timesteps: 10
        gaussian_loss_type: "mse"
        scheduler: "cosine"
        model_type: "mlp"
        device: "cpu"
        batch_size: 32
        lr: 0.01
        weight_decay: 0.001
        steps: 400
        dataset_params:
            task_type: "binclass"
            val_fraction: 0.1
            cat_cols: []
            boolean_cols: ['attr3', 'attr4']
        model_params:
            is_y_cond: false
            rtdl_params:
                d_layers: [
                    256,
                    256
                ]
                dropout: 0.1
    ''')
    
    ddpm = Tab_ddpm_controller(embed_df, node_df, "temp/", config_dict)
    ddpm.fit()
    synth_node = ddpm.sample_model(10)
    print(synth_node)
     
# %%
