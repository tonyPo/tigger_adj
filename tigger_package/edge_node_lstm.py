import os
import torch
import torch.nn as nn
import torch.nn.functional as nnf
# current_file_directory = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(current_file_directory)


# ##########################
# TO DO
# Check optimer
# Check the gather constructie
# ######################



class EdgeNodeLSTM(nn.Module):
    def __init__(self, vocab, gnn_dim, nb_layers, num_components, edge_attr_dim,
                 node_attr_dim, nb_lstm_units=100, clust_dim=3, mu_hidden_dim=100,
                 batch_size=3, kl_weight=0.001, device='cpu', dropout=0, seed=1):
        super(EdgeNodeLSTM, self).__init__()
        torch.manual_seed(seed)
        self.vocab = vocab
        self.nb_lstm_layers = nb_layers  # number of LSTM layers
        self.nb_lstm_units = nb_lstm_units  # dimension of hidden layer h
        self.clust_dim = clust_dim  # dimension of the cluster embedding
        self.batch_size = batch_size
        self.edge_attr_dim = edge_attr_dim  # number of edge attributes
        self.node_attr_dim = node_attr_dim  # nomber of node attributes
        # don't count the padding tag for the classifier output
        self.nb_events = len(self.vocab) - 1
        self.gnn_dim = gnn_dim
        self.mu_hidden_dim = mu_hidden_dim  # dimension between cluster embedding and z_gnn
        self.num_components = num_components  # number of clusters
        self.kl_weight = kl_weight
        self.dropout = dropout
        print("Number of components,", num_components)
        
        # create cluster embedding
        self.cluster_embeddings = nn.Embedding(
            num_embeddings=self.num_components,
            embedding_dim=self.clust_dim,
            padding_idx=self.vocab['<PAD>']
        ).to(device)
        
        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.clust_dim + self.gnn_dim + self.node_attr_dim,   ## cluster + GNN + edge embedding
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True,
            dropout = self.dropout            
        ).to(device)
        
        # output layer which projects back to tag space
        self.embedding_hidden = nn.Linear(self.gnn_dim,self.gnn_dim).to(device)  #re-embedding gnn embedding?
        self.hidden_to_ne_hidden = nn.Linear(self.nb_lstm_units, 200).to(device)   # Z_cluster
        self.clusterid_hidden = nn.Linear(200,self.num_components).to(device)  # Z_cluster to cluster distribution
        self.cluster_mu = nn.Linear(200,self.mu_hidden_dim*self.num_components).to(device)  # mu's per cluster
        self.cluster_var = nn.Linear(200,self.mu_hidden_dim*self.num_components).to(device) # var's per cluster
        
        gnn_dim2 = int((self.mu_hidden_dim - self.gnn_dim) / 2 + self.mu_hidden_dim)
        self.gnn_dropout1 = nn.Dropout(dropout).to(device)
        self.gnn_decoder1 = nn.Linear(self.mu_hidden_dim, gnn_dim2).to(device)  #layer 1 gnn_decoder
        self.gnn_dropout2 = nn.Dropout(dropout).to(device)
        self.gnn_decoder2 = nn.Linear(gnn_dim2, self.gnn_dim).to(device)  # layer 2 gnn_decoder
        self.gnn_dropout3 = nn.Dropout(dropout).to(device)
        self.gnn_decoder3 = nn.Linear(self.gnn_dim, self.gnn_dim).to(device)  # layer 3 gnn_decoder
        
        edge_dim2 = int((self.mu_hidden_dim - self.edge_attr_dim) / 2 + self.mu_hidden_dim)
        self.edge_dropout1 = nn.Dropout(dropout).to(device)
        self.edge_decoder1 = nn.Linear(self.mu_hidden_dim, edge_dim2).to(device)  #layer 1 gnn_decoder
        self.edge_dropout2 = nn.Dropout(dropout).to(device)
        self.edge_decoder2 = nn.Linear(edge_dim2, self.edge_attr_dim).to(device)  # layer 2 gnn_decoder
        self.edge_dropout3 = nn.Dropout(dropout).to(device)
        self.edge_decoder3 = nn.Linear(self.edge_attr_dim, self.edge_attr_dim).to(device)  # layer 3 gnn_decoder
        
        node_dim2 = int((self.mu_hidden_dim - self.node_attr_dim) / 2 + self.mu_hidden_dim)
        self.feat_dropout1 = nn.Dropout(dropout).to(device)
        self.feat_decoder1 = nn.Linear(self.mu_hidden_dim, node_dim2).to(device)  #layer 1 gnn_decoder
        self.feat_dropout2 = nn.Dropout(dropout).to(device)
        self.feat_decoder2 = nn.Linear(node_dim2, self.node_attr_dim).to(device)  # layer 2 gnn_decoder
        self.feat_dropout3 = nn.Dropout(dropout).to(device)
        self.feat_decoder3 = nn.Linear(self.node_attr_dim, self.node_attr_dim).to(device)  # layer 3 gnn_decoder
                
        self.mse_los_gnn = nn.MSELoss(reduction='none').to(device)
        self.mse_loss_edge = nn.MSELoss(reduction='none').to(device)
        self.mse_loss_feat = nn.MSELoss(reduction='none').to(device)
        self.celoss_cluster = nn.CrossEntropyLoss(ignore_index=0).to(device)

        self.relu_cluster = nn.LeakyReLU().to(device)  # activation forhidden to determin cluster id
        
        self.device = device
        
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale/2)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz #.sum(dim=-1)    
    
    def event_mse(self,x_hat,x):
        a =  (x-x_hat)*(x-x_hat)
        return a.sum(-1)
    
    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
    
    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.zeros(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units).to(self.device)
        hidden_b = torch.zeros(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units).to(self.device)
        self.hidden = (hidden_a, hidden_b)
        return (hidden_a, hidden_b)

    def forward(self, node_embed,
                node_attr,  
                x_length, 
                cluster_id,
                y_cluster_id=None):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        if self.training:
            self.init_hidden()           
        batch_size, seq_len, _ = node_embed.size()
        
        # ---------------------
        # 1. embed the input
        # --------------------
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        CID_embedding = self.cluster_embeddings(cluster_id)  # retrieve embedding for cluster id
        X = torch.cat((node_attr, node_embed, CID_embedding), -1)        
        
        # ---------------------
        # 2. Run through RNN
        # ---------------------
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        # X = torch.nn.utils.rnn.pack_padded_sequence(X, x_length, batch_first=True,enforce_sorted=False)
        X, self.hidden = self.lstm(X, self.hidden)  # now run through LSTM
        # X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True, total_length=seq_len)  # unpacking operation
        X = X.contiguous()
        
        # ---------------------
        # 3. Project to tag space
        # ---------------------
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # 2 FC MLPs to predict cluster id props
        Y_hat = nnf.leaky_relu(self.hidden_to_ne_hidden(X))  # Z_cluster
        cluster_logits = self.clusterid_hidden(Y_hat)  # prop logits distrubution over clusters.
    

        # Sample next cluster id (cluster id hat)
        if self.training:  # select true cluster during training?
            cluster_id_hat = y_cluster_id
        else:
            clust_dist = nnf.softmax(cluster_logits, dim=-1)
            clust_dist = clust_dist.view(-1,cluster_logits.shape[-1])
            cluster_id_hat = torch.multinomial(clust_dist, 1, replacement=True)
            cluster_id_hat = cluster_id_hat.view(batch_size,seq_len)
            
        Y_clusterid_sampled = cluster_id_hat.unsqueeze(-1).repeat(1,1,self.mu_hidden_dim).unsqueeze(2)  
                  
        # retrieve mu and log_var for y_cluster
        mu = self.cluster_mu(Y_hat)
        mu = mu.view(batch_size,seq_len,self.num_components,self.mu_hidden_dim)
        mu = torch.gather(mu,2,Y_clusterid_sampled).squeeze(2)
        
        log_var = self.cluster_var(Y_hat)
        log_var = log_var.view((batch_size,seq_len,self.num_components,self.mu_hidden_dim))
        log_var = torch.gather(log_var,2,Y_clusterid_sampled).squeeze(2)
            
        
        std = torch.exp(log_var / 2)  # determine std for hidden layer z to gnn
        q = torch.distributions.Normal(mu, std)  # create distribution layer
        z = q.rsample()  # sample z for reconstruction of gnn embedding + edge atributes
        
        # Reconstruct gnn embedding
        ne_hat = self.gnn_dropout1(z)
        ne_hat = nnf.leaky_relu(self.gnn_decoder1(ne_hat))  # reconstruct  GNN embeding
        ne_hat = self.gnn_dropout2(ne_hat)
        ne_hat = nnf.leaky_relu(self.gnn_decoder2(ne_hat))
        ne_hat = self.gnn_dropout3(ne_hat)
        ne_hat = self.gnn_decoder3(ne_hat)  #3de layer decoder gnn embedding
        
        # Reconstruct edge features
        edge_attr_hat = self.edge_dropout1(z)
        edge_attr_hat = nnf.leaky_relu(self.edge_decoder1(z))  # reconstruct  edge
        edge_attr_hat = self.edge_dropout2(edge_attr_hat)
        edge_attr_hat = nnf.leaky_relu(self.edge_decoder2(edge_attr_hat))
        edge_attr_hat = self.edge_dropout3(edge_attr_hat)
        edge_attr_hat = nnf.leaky_relu(self.edge_decoder3(edge_attr_hat))  #3de layer decoder
        
        # Reconstruct node features
        node_attr_hat = self.feat_dropout1(z)
        node_attr_hat = nnf.leaky_relu(self.feat_decoder1(z))  # reconstruct  edge
        node_attr_hat = self.feat_dropout2(node_attr_hat)
        node_attr_hat = nnf.leaky_relu(self.feat_decoder2(node_attr_hat))
        node_attr_hat = self.feat_dropout3(node_attr_hat)
        node_attr_hat = nnf.leaky_relu(self.feat_decoder3(node_attr_hat))  #3de layer decoder
        
        mask = cluster_id!=0  #TODO check if mask is same as in loss
        self.kl = self.kl_divergence(z, mu, std)*mask   # used for regularisation
        
        return {'node_embed_hat': ne_hat,
                'edge_attr_hat': edge_attr_hat,
                'cluster_id_hat_vector': cluster_logits,
                'cluster_id_hat': cluster_id_hat,
                'node_attr_hat': node_attr_hat,
                }
        
    def train_los(self, node_embed_hat, edge_attr_hat, cluster_id_hat_vector, node_attr_hat,
            cluster_id_hat, node_embed, edge_attr, node_attr, x_length, cluster_id
        ):        
        # reconstruction loss
        mask = cluster_id!=0
        recon_loss_ne = self.mse_los_gnn(node_embed_hat,node_embed)
        recon_loss_ne = recon_loss_ne.sum(-1)*mask
        recon_loss_edge_attr = self.mse_loss_edge(edge_attr_hat, edge_attr)
        recon_loss_edge_attr = recon_loss_edge_attr.sum(-1)*mask
        recon_loss_node_attr = self.mse_loss_feat(node_attr_hat, node_attr)
        recon_loss_node_attr = recon_loss_node_attr.sum(-1)*mask
        elbo = self.kl_weight*self.kl + recon_loss_ne + recon_loss_edge_attr + recon_loss_node_attr  ### recon_loss 
        num_events = x_length.sum()
        elbo = elbo.sum() / num_events
        
        #cluster loss
        cluster_id_hat_vector = cluster_id_hat_vector.view(-1, cluster_id_hat_vector.shape[-1])
        # returns mean value!!!
        loss_cluster = self.celoss_cluster(cluster_id_hat_vector, cluster_id.reshape(-1))
        
        loss = elbo + loss_cluster
        
        log_dict = {
            'loss': loss.item(),
            'elbo_loss': elbo.item(),
            'kl_loss': (self.kl.sum()/ num_events*self.kl_weight).item(),
            'reconstruction_ne': (recon_loss_ne.sum()/num_events).item(),
            'reconstruction_edge': (recon_loss_edge_attr.sum()/num_events).item(),
            'cross_entropy_cluster': loss_cluster.item(),
            'reconstruction_feat': (recon_loss_node_attr.sum()/num_events).item(),
        }      
        
        return loss, log_dict
