import tensorflow as tf
from GAE.graph_case_controller import GraphAutoEncoder
from GAE.data_feeder_graphframes import DataFeederGraphFrames

class LabelTransferrer:
    def __init__(self, nodes, edges, synth_nodes, synth_edges, config_dict, spark=None):
        self.nodes = nodes
        self.edges = edges
        self.synth_nodes = synth_nodes
        self.synth__edges = synth_edges
        for key, val in config_dict.items():
            setattr(self, key, val)
        self.spark = spark
        
    
    def transfer(self):
        
        #train and retrieve graphcase embedding
        if self.spark is not None:
            gea = self.train_and_calculate_graphcase_embedding_spark()

        return gea
        #select embedding for positive labels + add noise
        
        # for each embedding
        
        # find closest embedding in synthetic graph
        
        # update 0th layer node and edgee atributes. Add missing neighbors / remove unused nieghbors
        
        # update 1st layer degree node and edge attributes. 
        # add missing edge / neighbors or remove unused edges / neighbors.
        
        # update 2nd layer node 
         
    def train_and_calculate_graphcase_embedding_spark(self):
        spark_nodes = self.spark.createDataFrame(self.nodes.reset_index())  
        spark_egdes = (self.spark.createDataFrame(self.edges)
            .withColumnRenamed("start", 'src')
            .withColumnRenamed("end", "dst")
            .withColumnRenamed(self.weight_col, "weight")
        )
        graph = (spark_nodes, spark_egdes)
        
        #init graphcase
        encoder_labels=[c for c in spark_nodes.columns if c not in ['id', self.label_col]]
        gae = GraphAutoEncoder(
            graph, support_size=self.support_size, 
            dims=self.dims, 
            batch_size=self.batch_size, 
            hub0_feature_with_neighb_dim=self.hub0_dim,
            useBN=self.usebn, 
            verbose=self.verbose, 
            seed=self.seed, 
            learning_rate=self.learning_rate, 
            act=tf.nn.relu, 
            encoder_labels=encoder_labels,
            data_feeder_cls=DataFeederGraphFrames
        )
        return gae
        