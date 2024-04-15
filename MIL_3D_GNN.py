import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp,global_add_pool as gadp
from torch_geometric.nn import SAGPooling, TransformerConv  
from torch_geometric.data import Batch


torch.manual_seed(42)
class GNN(torch.nn.Module):
    def __init__(self, model_params, isinstance = False):
        super(GNN, self).__init__()
        self.isinstance = isinstance
        node_dim_2d = model_params["node_dim_2d"]
        node_dim_3d = model_params["node_dim_3d"]
        edge_dim_2d = model_params["edge_dim_2d"]
        edge_dim_3d = model_params["edge_dim_3d"]
        hidden_nodes = model_params["hidden_nodes"]
        self.n_block = model_params["n_block"]
        self.pooling_every_n = model_params["pooling_every_n"]
        self.dropout_mlp = model_params["dropout_mlp"]
        self.pooling_rate = model_params["pooling_rate"]
        self.mlp_layers_instance = model_params["mlp_layers_instance"]
        self.mlp_hidden_instance = model_params["mlp_hidden_instance"]  
        #Block 1
        self.conv_2d_1 = TransformerConv(in_channels = node_dim_2d, out_channels = hidden_nodes, heads = 3, edge_dim = edge_dim_2d, beta = True)
        self.conv_3d_1 = TransformerConv(in_channels = node_dim_3d, out_channels = hidden_nodes, heads = 3, edge_dim = edge_dim_3d, beta = True)
        self.embedding_2d_1 = Linear(hidden_nodes*3, hidden_nodes)
        self.embedding_3d_1 = Linear(hidden_nodes*3, hidden_nodes)
        self.bn_2d_1 = BatchNorm1d(hidden_nodes)
        self.bn_3d_1 = BatchNorm1d(hidden_nodes)

        #Block convolutions 2 for 2d and 3d
        self.conv_layers_2d_2 = ModuleList([])
        self.embedding_2d_2 = ModuleList([])
        self.bn_2d_2 = ModuleList([])
        self.conv_layers_3d_2 = ModuleList([])
        self.embedding_3d_2 = ModuleList([])
        self.bn_3d_2 = ModuleList([])

        #Block 3 convolutions 3d and 2d
        self.conv_layers_3 = TransformerConv(in_channels = hidden_nodes*2, out_channels = hidden_nodes, heads = 3, edge_dim = edge_dim_2d+edge_dim_3d, beta = True)
        self.embedding_3 = Linear(hidden_nodes*3, hidden_nodes)
        self.bn_3 = BatchNorm1d(hidden_nodes)

        #Block 4
        self.conv_layers_4 = ModuleList([])
        self.embedding_4 = ModuleList([])
        self.bn_4 = ModuleList([])
        self.pooling_layers = ModuleList([])



        for i in range(self.n_block):
            self.conv_layers_2d_2.append(TransformerConv(in_channels = hidden_nodes, out_channels = hidden_nodes, heads = 3, edge_dim = edge_dim_2d, beta = True))
            self.embedding_2d_2.append(Linear(hidden_nodes*3, hidden_nodes))
            self.bn_2d_2.append(BatchNorm1d(hidden_nodes))
            self.conv_layers_3d_2.append(TransformerConv(in_channels = hidden_nodes, out_channels = hidden_nodes, heads = 3, edge_dim = edge_dim_3d, beta = True))
            self.embedding_3d_2.append(Linear(hidden_nodes*3, hidden_nodes))
            self.bn_3d_2.append(BatchNorm1d(hidden_nodes))

            self.conv_layers_4.append(TransformerConv(in_channels = hidden_nodes, out_channels = hidden_nodes, heads = 3, edge_dim = edge_dim_2d+edge_dim_3d, beta = True))
            self.embedding_4.append(Linear(hidden_nodes*3, hidden_nodes))
            self.bn_4.append(BatchNorm1d(hidden_nodes))
            if i % self.pooling_every_n == 0:
                self.pooling_layers.append(SAGPooling(hidden_nodes, ratio=self.pooling_rate))
            
        #FC block 
        if isinstance: #if instance_based model --> output is a scalar with is probability of each instance in bag
            self.fc_layers = torch.nn.ModuleList([])
            if self.mlp_layers_instance == 1:
                self.fc_layers.append(Linear(hidden_nodes*3, 1))
            else:
                for i in range(self.mlp_layers_instance):
                    if i == 0:
                        intput = hidden_nodes*3
                        output = self.mlp_hidden_instance
                    elif i != self.mlp_layers_instance-1:
                        intput =  self.mlp_hidden_instance
                        output = self.mlp_hidden_instance
                    else:
                        intput = self.mlp_hidden_instance
                        output = 1
                    self.fc_layers.append(Linear(intput, output))
        else:
            self.fc_layers = torch.nn.ModuleList([])
            if self.mlp_layers_instance == 1:
                self.fc_layers.append(Linear(hidden_nodes, 1))
            else:
                for i in range(self.mlp_layers_instance):
                    if i == 0:
                        intput = hidden_nodes*3
                        output = self.mlp_hidden_instance
                    elif i != self.mlp_layers_instance-1:
                        intput =  self.mlp_hidden_instance
                        output = self.mlp_hidden_instance
                    else:
                        intput = self.mlp_hidden_instance
                        output = self.mlp_hidden_instance
                    self.fc_layers.append(Linear(intput, output))

    def forward(self, data):
        node_2d = data.node_features_2d
        #print("node_2d", node_2d) 
        node_3d = data.node_features_3d
        #print("node_3d", node_3d) 
        edge_attr_2d = data.edge_attr_2d
        #print("edge_attr_2d", edge_attr_2d) 
        edge_attr_3d = data.edge_attr_3d
        #print("edge_attr_3d", edge_attr_3d)
        edge_index = data.edge_index
        batch_index = data.batch
        #Block 1 for 2d
        x_2d = self.conv_2d_1(node_2d, edge_index, edge_attr_2d)
        #print("x_2d_block 1", x_2d) 
        x_2d = torch.relu(self.embedding_2d_1(x_2d))
        #x_2d = self.embedding_2d_1(x_2d)
        #x_2d = torch.nn.functional.prelu(self.embedding_2d_1(x_2d))
        x_2d = self.bn_2d_1(x_2d)
        #print("x_2d_block 1", x_2d)
        #Block 1 for 3d
        x_3d = self.conv_3d_1(node_3d, edge_index, edge_attr_3d)
        x_3d = torch.relu(self.embedding_3d_1(x_3d))
        #x_3d = self.embedding_3d_1(x_3d)
        x_3d = self.bn_3d_1(x_3d)
        #print("x_3d_block 1", x_3d)
        #Block 2 for 2d 
        global_2d = []
        for i in range(self.n_block):
            x_2d = self.conv_layers_2d_2[i](x_2d, edge_index, edge_attr_2d)
            x_2d = torch.relu(self.embedding_2d_2[i](x_2d))
            #x_2d = self.embedding_2d_2[i](x_2d)
            #print("x_2d block 2", x_2d)
            x_2d = self.bn_2d_2[i](x_2d)
            global_2d.append(x_2d)

        x_2d = sum(global_2d)
        #print("x_2d_sum", x_2d)
        #Block 2 for 3d
        global_3d = []
        for i in range(self.n_block):
            x_3d = self.conv_layers_3d_2[i](x_3d, edge_index, edge_attr_3d)
            x_3d = torch.relu(self.embedding_3d_2[i](x_3d))
            #x_3d = self.embedding_3d_2[i](x_3d)
            #print("x_3d_n_block", x_3d)
            x_3d = self.bn_3d_2[i](x_3d)
            global_3d.append(x_3d)
        x_3d = sum(global_3d)
        #print("x_3d after block", x_3d)

        x = torch.cat([x_2d, x_3d], dim = 1)
        #print(x)

        ##print(x_2d.shape,x_3d.shape,x.shape)
        edge_attr = torch.cat([edge_attr_2d, edge_attr_3d], dim = 1)
        x = self.conv_layers_3(x, edge_index, edge_attr)
        #x = torch.relu(self.embedding_3(x))
        x = self.embedding_3(x)
        #x = self.bn_3(x)
        #print("after concat", x.shape)
        #Block 3 for 2d and 3d
        global_graph = []
        for i in range(self.n_block):
            x = self.conv_layers_4[i](x, edge_index, edge_attr)
            x = torch.relu(self.embedding_4[i](x))
            #x = self.embedding_4[i](x)
            x = self.bn_4[i](x)
            #print(x)
            if i % self.pooling_every_n == 0 or i == self.n_block:
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i/self.pooling_every_n)](x, edge_index, edge_attr, batch_index)
                global_graph.append(torch.cat([gmp(x, batch_index), gap(x, batch_index),gadp(x, batch_index)], dim=1))
            
        h_g = sum(global_graph)
        #FC block
        if self.isinstance:
            if self.mlp_layers_instance == 1:
                h_g = torch.sigmoid(self.fc_layers[0](h_g))
                return h_g.view(-1)
            else:
                for i in range(self.mlp_layers_instance):
                    if i != self.mlp_layers_instance-1:
                        h_g = torch.relu(self.fc_layers[i](h_g))
                        h_g = F.dropout(h_g, p=self.dropout_mlp, training=self.training)
                    else:

                        h_g =  torch.sigmoid(self.fc_layers[i](h_g))

                return h_g.view(-1)#, weight.view(-1)
        else:
            if self.mlp_layers_instance == 1:
                h_g = torch.sigmoid(self.fc_layers[0](h_g))
                return h_g.view(-1)
            else:   
                for i in range((self.mlp_layers_instance)):
                    if i != self.mlp_layers_instance-1:
                        h_g = torch.relu(self.fc_layers[i](h_g))
                        h_g = F.dropout(h_g, p=self.dropout_mlp, training=self.training)
                    else:
                        h_g = torch.relu(self.fc_layers[i](h_g))

                return h_g#, weight.view(-1)



class instance_GNN(torch.nn.Module):
    def __init__(self, config, aggregation = "mean"):
        super(instance_GNN, self).__init__()
        self.GNN_instance = GNN(model_params = config, isinstance = True)
        self.aggregation = aggregation
        
    def forward(self, data):
        list_bag = data.instance_data #List of bags of instances    
        batch_output = []
        for bag in list_bag:
            batch_instance = Batch.from_data_list(bag)
            output = self.GNN_instance(batch_instance)
            output_bag = torch.mean(output)
        
   
            batch_output.append(output_bag)
        batch_output = torch.stack(batch_output)
        return batch_output

# class bag_GNN(torch.nn.Module):
#     def __init__(self, model_params):
#         super(bag_GNN, self).__init__()
#         self.GNN_bag = GNN(model_params = model_params, isinstance = False)
#         #FC block
#         hidden_nodes = model_params["mlp_hidden_instance"]
#         #self.bag_mlp_layers = model_params["bag_mlp_layers"]
#         self.bag_mlp_hidden = model_params["bag_mlp_hidden"]
        
#         self.fc_layers = torch.nn.ModuleList([])
#         self.mlp_1 = Linear(hidden_nodes, self.bag_mlp_hidden)
#         self.mlp_2 = Linear(self.bag_mlp_hidden, 1)


        
#     def forward(self, data):
#         list_bag = data.instance_data
#         batch_output = []
#         for bag in list_bag:
#             batch_instance = Batch.from_data_list(bag)
#             output = self.GNN_bag(batch_instance)
           
#             output_bag = torch.mean(output, dim = 0)
       
#             output_bag = torch.relu(self.mlp_1(output_bag))
     
#             output_bag = F.dropout(output_bag, p=0.5, training=self.training)
#             output_bag = torch.sigmoid(self.mlp_2(output_bag))
#             # print(output_bag.shape)
#             # for i in range(self.bag_mlp_layers):
#             #     if i != self.bag_mlp_layers-1:
#             #         output_bag = torch.relu(self.fc_layers[i](output_bag))
#             #         output_bag = F.dropout(output_bag, p=0.5, training=self.training)
#             #     else:
#             #         output_bag = torch.sigmoid(self.fc_layers[i](output_bag))
#             batch_output.append(output_bag)
#         batch_output = torch.stack(batch_output)
#         return batch_output.view(-1)









        
        
