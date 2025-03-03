import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import RelGraphConv, GATConv
from node2vec import Node2Vec


class TuckERLayer(nn.Module):
    def __init__(self, dim, r_dim):
        super(TuckERLayer, self).__init__()
        
        self.W = nn.Parameter(torch.rand(r_dim, dim, dim))
        nn.init.xavier_uniform_(self.W.data)
        self.bn0 = nn.BatchNorm1d(dim)
        self.bn1 = nn.BatchNorm1d(dim)
        # original: 0.3, 0.4, 0.5
        self.input_drop = nn.Dropout(0.3)
        self.hidden_drop = nn.Dropout(0.4)
        self.out_drop = nn.Dropout(0.5)

    def forward(self, e_embed, r_embed):
        x = self.bn0(e_embed)
        x = self.input_drop(x)
        x = x.view(-1, 1, x.size(1))
        
        r = torch.mm(r_embed, self.W.view(r_embed.size(1), -1))
        r = r.view(-1, x.size(2), x.size(2))
        r = self.hidden_drop(r)
       
        x = torch.bmm(x, r)
        x = x.view(-1, x.size(2))
        x = self.bn1(x)
        x = self.out_drop(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
    
class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.5):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.sim_func = Similarity(temp=temp)

    def forward(self, emb1, emb2):
        batch_sim = self.sim_func(emb1.unsqueeze(1), emb2.unsqueeze(0))
        labels = torch.arange(batch_sim.size(0)).long().to('cuda')
        return self.loss(batch_sim, labels)


class VISTATucker(nn.Module):
    def __init__(
            self, 
            num_ent, 
            num_rel, 
            rel_vis, 
            dim_vis,
            rel_txt, 
            dim_txt, 
            ent_vis_mask,
            ent_txt_mask,
            rel_vis_mask,
            dim_str,
            num_head,
            dim_hid,
            num_layer_enc_ent,
            num_layer_enc_rel,
            num_layer_dec,
            dropout = 0.1,
            emb_dropout = 0.6, 
            vis_dropout = 0.1, 
            txt_dropout = 0.1,
            visual_token_index = None, 
            text_token_index = None,
            score_function = "tucker"
        ):
        super(VISTATucker, self).__init__()
        self.dim_str = dim_str
        self.num_head = num_head
        self.dim_hid = dim_hid
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.rel_vis = rel_vis
        self.rel_txt = None

        self.visual_token_index = visual_token_index
        self.visual_token_embedding = nn.Embedding(num_embeddings=8193, embedding_dim=self.dim_str)
        self.text_token_index = text_token_index
        self.text_token_embedding = nn.Embedding(num_embeddings=15000, embedding_dim=self.dim_str)
        self.score_function = score_function

        false_ents = torch.full((self.num_ent,1),False).cuda()
        self.ent_mask = torch.cat([false_ents, false_ents, ent_vis_mask, ent_txt_mask], dim = 1)
        # print(self.ent_mask.shape)
        false_rels = torch.full((self.num_rel,1),False).cuda()
        self.rel_mask = torch.cat([false_rels, false_rels], dim = 1)
        
        self.ent_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.ent_embeddings = nn.Parameter(torch.Tensor(num_ent, 1, dim_str))
        self.rel_embeddings = nn.Parameter(torch.Tensor(num_rel, 1 ,dim_str))
        self.lp_token = nn.Parameter(torch.Tensor(1,dim_str))

        self.str_ent_ln = nn.LayerNorm(dim_str)
        self.str_rel_ln = nn.LayerNorm(dim_str)
        self.vis_ln = nn.LayerNorm(dim_str)
        self.txt_ln = nn.LayerNorm(dim_str)

        self.embdr = nn.Dropout(p = emb_dropout)
        self.visdr = nn.Dropout(p = vis_dropout)
        self.txtdr = nn.Dropout(p = txt_dropout)


        self.pos_str_ent = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_vis_ent = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_txt_ent = nn.Parameter(torch.Tensor(1,1,dim_str))

        self.pos_str_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_vis_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_txt_rel = nn.Parameter(torch.Tensor(1,1,dim_str))

        self.pos_head = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_tail = nn.Parameter(torch.Tensor(1,1,dim_str))


        ent_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.ent_encoder = nn.TransformerEncoder(ent_encoder_layer, num_layer_enc_ent)
        rel_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.rel_encoder = nn.TransformerEncoder(rel_encoder_layer, num_layer_enc_rel)
        decoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layer_dec)

        self.contrastive = ContrastiveLoss(temp=0.5)
        self.num_con = 512
        
        if self.score_function == "tucker":
            self.tucker_decoder = TuckERLayer(dim_str, dim_str)
        else:
            pass
        
        self.init_weights()
        

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings)
        nn.init.xavier_uniform_(self.rel_embeddings)
        # nn.init.xavier_uniform_(self.proj_ent_vis.weight)
        # nn.init.xavier_uniform_(self.proj_rel_vis.weight)
        # nn.init.xavier_uniform_(self.proj_txt.weight)
        nn.init.xavier_uniform_(self.ent_token)
        nn.init.xavier_uniform_(self.rel_token)
        nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_vis_ent)
        nn.init.xavier_uniform_(self.pos_txt_ent)
        nn.init.xavier_uniform_(self.pos_str_rel)
        nn.init.xavier_uniform_(self.pos_vis_rel)
        nn.init.xavier_uniform_(self.pos_txt_rel)
        nn.init.xavier_uniform_(self.pos_head)
        nn.init.xavier_uniform_(self.pos_rel)
        nn.init.xavier_uniform_(self.pos_tail)

        nn.init.xavier_uniform_(self.visual_token_embedding.weight)
        nn.init.xavier_uniform_(self.text_token_embedding.weight)

        # self.proj_ent_vis.bias.data.zero_()
        # self.proj_rel_vis.bias.data.zero_()
        # self.proj_txt.bias.data.zero_()

    def forward(self):
        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        
        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)
        rep_ent_vis = self.visdr(self.vis_ln(entity_visual_tokens)) + self.pos_vis_ent
        entity_text_tokens = self.text_token_embedding(self.text_token_index)
        rep_ent_txt = self.txtdr(self.txt_ln(entity_text_tokens)) + self.pos_txt_ent

        ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim = 1)
        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask = self.ent_mask)[:,0]
        # rel_tkn = self.rel_token.tile(self.num_rel, 1, 1)
        rep_rel_str = self.embdr(self.str_rel_ln(self.rel_embeddings)) # + self.pos_str_rel
        # rel_seq = torch.cat([rel_tkn, rep_rel_str], dim = 1)
        # rel_embs = self.rel_encoder(rel_seq, src_key_padding_mask = self.rel_mask)[:,0]
        return torch.cat([ent_embs, self.lp_token], dim = 0), rep_rel_str.squeeze(dim=1)

    def contrastive_loss(self, emb_ent1):
        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)
        rep_ent_vis = self.visdr(self.vis_ln(entity_visual_tokens)) + self.pos_vis_ent
        entity_text_tokens = self.text_token_embedding(self.text_token_index)
        rep_ent_txt = self.txtdr(self.txt_ln(entity_text_tokens)) + self.pos_txt_ent
        ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim = 1)
        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask = self.ent_mask)[:,0]
        emb_ent2 = torch.cat([ent_embs, self.lp_token], dim = 0)
        select_ents = torch.randperm(emb_ent1.shape[0])[: self.num_con]

        contrastive_loss = self.contrastive(emb_ent1[select_ents], emb_ent2[select_ents])
        return contrastive_loss


    def score(self, emb_ent, emb_rel, triplets):
        # args:
        #   emb_ent: [num_ent, emb_dim]
        #   emb_rel: [num_rel, emb_dim]
        #   triples: [batch_size, 3]
        # return:
        #   scores: [batch_size, num_ent]
        h_seq = emb_ent[triplets[:,0] - self.num_rel].unsqueeze(dim = 1) + self.pos_head
        r_seq = emb_rel[triplets[:,1] - self.num_ent].unsqueeze(dim = 1) + self.pos_rel
        t_seq = emb_ent[triplets[:,2] - self.num_rel].unsqueeze(dim = 1) + self.pos_tail
        dec_seq = torch.cat([h_seq, r_seq, t_seq], dim = 1)
        output_dec = self.decoder(dec_seq)
        rel_emb = output_dec[:, 1, :]
        ent_emb = output_dec[triplets != self.num_ent + self.num_rel]
        if self.score_function == "tucker":
            tucker_emb = self.tucker_decoder(ent_emb, rel_emb)
            score = torch.mm(tucker_emb, emb_ent[:-1].transpose(1, 0))
        else:
            output_dec = self.decoder(dec_seq)
            score = torch.inner(ent_emb, emb_ent[:-1])
        return score

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            n_classes = inputs.size(1)
            smooth_targets = (1 - self.label_smoothing) * targets + self.label_smoothing / n_classes
            BCE_loss = F.cross_entropy(inputs, smooth_targets, reduction='none')
        else:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)  # 计算 p_t
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class SupervisedContrastiveLoss(nn.Module):

    def __init__(self,
                 temperature = 0.2):
        super().__init__()
        assert temperature > 0, 'temperature should be a positive number.'
        self.temperature = temperature

    def forward(self,
                features,
                labels):

        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)

        # mask with shape [N, N], mask_{i, j}=1
        # if sample i and sample j have the same label
        label_mask = torch.eq(labels, labels.T).float().to(features.device)

        similarity = torch.div(
            torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        # mask out self-contrastive
        logits_mask = torch.ones_like(similarity)
        logits_mask.fill_diagonal_(0)

        exp_sim = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        per_label_log_prob = (log_prob * logits_mask *
                              label_mask).sum(1) / label_mask.sum(1)

        loss = -per_label_log_prob

        loss = loss.mean()
        return loss

class RegularizedCrossEntropyLoss(nn.Module):
    def __init__(self, model, label_smoothing=0.0, l2_lambda=0.001):
        super(RegularizedCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.l2_lambda = l2_lambda
        self.model = model

    def forward(self, output, target):
        cross_entropy_loss = self.loss_fn(output, target)
        l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
        loss = cross_entropy_loss + self.l2_lambda * l2_norm
        return loss
    
class GraphEncoder_GCN(nn.Module):
    def __init__(self, graph, rels, in_dim, hidden_dim, out_dim, num_node, num_rel, device='cuda'):
        super(GraphEncoder_GCN, self).__init__()
        self.graph = graph.to(device)
        self.relations = torch.tensor(rels).to(device)
        # self.node_emb = nn.Parameter(self.generate_node2vec_embeddings(graph, in_dim).to(device))
        # print(self.node_emb.size())
        self.node_emb = nn.Parameter(torch.Tensor(num_node, in_dim).to(device))
        self.conv1 = RelGraphConv(in_dim, hidden_dim, num_rel, regularizer='basis', num_bases=2, activation=nn.Tanh())
        self.conv2 = RelGraphConv(hidden_dim, out_dim, num_rel, regularizer='basis', num_bases=2, activation=nn.ReLU())
        nn.init.xavier_uniform_(self.node_emb)
    
    def generate_node2vec_embeddings(self, dgl_graph, dimensions, walk_length=5, num_walks=20):
        nx_graph = dgl_graph.to_networkx().to_undirected()
        node2vec = Node2Vec(nx_graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=4)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        embeddings = model.wv
        embeddings_tensor = torch.tensor([embeddings[str(i)] for i in range(len(embeddings))])
        return embeddings_tensor
    
    def forward(self):
        # print(self.graph.device, self.node_emb.device, self.relations.device)
        h = self.conv1(self.graph, self.node_emb, etypes=self.relations)
        out = self.conv2(self.graph, h, etypes=self.relations)
        return out

class GraphEncoder_GAT(nn.Module):
    def __init__(self, graph, rels, in_dim, hidden_dim, out_dim, num_node, num_rel, device='cuda'):
        super(GraphEncoder_GAT, self).__init__()
        self.graph = graph.to(device)
        self.relations = torch.tensor(rels).to(device)
        # self.node_emb = nn.Parameter(torch.Tensor(num_node, in_dim).to(device))
        self.node_emb = nn.Parameter(self.generate_node2vec_embeddings(graph, in_dim).to(device))
        # print(self.node_emb.size())

        # Initialize GAT layers
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads=4, activation=nn.ReLU())
        self.conv2 = GATConv(hidden_dim * 4, out_dim, num_heads=1, activation=None)  # Adjust output based on heads

        nn.init.xavier_uniform_(self.node_emb)
    
    def generate_node2vec_embeddings(self, dgl_graph, dimensions, walk_length=5, num_walks=20):
        nx_graph = dgl_graph.to_networkx().to_undirected()
        node2vec = Node2Vec(nx_graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=4, p=1, q=1) # DeepWalk
        # node2vec = Node2Vec(nx_graph, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=4) #Node2Vec
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        embeddings = model.wv
        embeddings_tensor = torch.tensor([embeddings[str(i)] for i in range(len(embeddings))])
        return embeddings_tensor
    
    def forward(self):
        # print(self.graph.device, self.node_emb.device, self.relations.device)
        h = self.conv1(self.graph, self.node_emb).flatten(1)
        out = self.conv2(self.graph, h).mean(1)
        return out
