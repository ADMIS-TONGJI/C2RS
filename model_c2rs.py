import torch
import torch.nn as nn
import torch.nn.functional as F
from model_tool import *

class C2RS(nn.Module):
    def __init__(
            self, 
            num_ent, 
            num_rel, 
            ent_vis_mask,
            ent_txt_mask,
            dim_str,
            num_head,
            dim_hid,
            num_layer_enc_ent,
            num_layer_enc_rel,
            num_layer_dec,
            dropout = 0.1,
            emb_dropout = 0.9, 
            vis_dropout = 0.4, 
            txt_dropout = 0.1,
            visual_token_index = None, 
            text_token_index = None,
            score_function = "tucker",
            dataset = "MKG-W"
        ):
        super(C2RS, self).__init__()
        self.dim_str = dim_str
        self.num_head = num_head
        self.dim_hid = dim_hid
        self.num_ent = num_ent
        self.num_rel = num_rel

        visual_tokens = torch.load("tokens/visual.pth")
        textual_tokens = torch.load("tokens/{}-textual.pth".format(dataset))
        structure_tokens = torch.load("tokens/{}-node2vec.pth".format(dataset))
        self.visual_token_index = visual_token_index
        self.visual_token_embedding = nn.Embedding.from_pretrained(visual_tokens).requires_grad_(False)
        self.text_token_index = text_token_index
        self.text_token_embedding = nn.Embedding.from_pretrained(textual_tokens).requires_grad_(False)
        self.score_function = score_function

        self.visual_token_embedding.requires_grad_(False)
        self.text_token_embedding.requires_grad_(False)

        false_ents = torch.full((self.num_ent,1),False).cuda()
        self.ent_mask = torch.cat([false_ents, false_ents, ent_vis_mask, ent_txt_mask], dim = 1)
        
        self.ent_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.ent_embeddings = structure_tokens.requires_grad_(False).unsqueeze(1)
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
        
        self.proj_ent_vis = nn.Linear(visual_tokens.size(1), dim_str)
        self.proj_ent_txt = nn.Linear(textual_tokens.size(1), dim_str)
        self.proj_s = nn.Linear(structure_tokens.size(1), dim_str)

        ent_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.ent_encoder = nn.TransformerEncoder(ent_encoder_layer, num_layer_enc_ent)
        rel_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.rel_encoder = nn.TransformerEncoder(rel_encoder_layer, num_layer_enc_rel)
        decoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layer_dec)

        self.num_con = 256
        self.num_vis = ent_vis_mask.shape[1]
        if self.score_function == "tucker":
            self.tucker_decoder = TuckERLayer(dim_str, dim_str)
        else:
            pass
        
        self.init_weights()

        self.register_buffer('head_valid', torch.zeros(self.num_ent, self.num_rel, dtype=torch.bool))
        self.register_buffer('tail_valid', torch.zeros(self.num_ent, self.num_rel, dtype=torch.bool))
        
        self.bceloss = nn.BCEWithLogitsLoss()

        self.head_classifier_r = nn.Sequential(
                                 nn.Linear(dim_str, dim_str),
                                 nn.ReLU(),
                                 nn.Linear(dim_str, self.num_ent)
                                )
        self.tail_classifier_r = nn.Sequential(
                                 nn.Linear(dim_str, dim_str),
                                 nn.ReLU(),
                                 nn.Linear(dim_str, self.num_ent)
                                )
        
    def prefill_valid(self, triplets, label):
        self.eval()
        with torch.no_grad():
            h = triplets[:,0] - self.num_rel
            r = triplets[:,1] - self.num_ent
            t = triplets[:,2] - self.num_rel
            fill_triplets = torch.stack([h, r, t, label]).T
            for i in fill_triplets:
                if i[0] == self.num_ent:
                    self.head_valid[i[3], i[1]] = True
                    self.tail_valid[i[2], i[1]] = True
                else:
                    self.head_valid[i[0], i[1]] = True
                    self.tail_valid[i[3], i[1]] = True

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings)
        nn.init.xavier_uniform_(self.rel_embeddings)
        nn.init.xavier_uniform_(self.proj_ent_vis.weight)
        nn.init.xavier_uniform_(self.proj_ent_txt.weight)
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

    def forward(self):
        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.embdr(self.str_ent_ln(self.proj_s(self.ent_embeddings))) + self.pos_str_ent
        
        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)
        rep_ent_vis = self.visdr(self.vis_ln(self.proj_ent_vis(entity_visual_tokens))) + self.pos_vis_ent
        entity_text_tokens = self.text_token_embedding(self.text_token_index)
        rep_ent_txt = self.txtdr(self.txt_ln(self.proj_ent_txt(entity_text_tokens))) + self.pos_txt_ent

        ent_seq = torch.cat([ent_tkn, rep_ent_str, rep_ent_vis, rep_ent_txt], dim = 1)
        ent_embs = self.ent_encoder(ent_seq, src_key_padding_mask = self.ent_mask)
        rep_rel_str = self.embdr(self.str_rel_ln(self.rel_embeddings))

        ent_v = torch.mean(ent_embs[:, 2: 2 + self.num_vis, :], dim=1)
        ent_t = torch.mean(ent_embs[:, 2 + self.num_vis: , ], dim=1)

        targets_itc = torch.arange(0, self.num_ent).to(ent_t.device)
        temp = 0.5
        sim_itc_tv = ent_v @ ent_t.t() / temp
        sim_itc_vt = ent_t @ ent_v.t() / temp
        itc_tv_loss = F.cross_entropy(sim_itc_tv, targets_itc)
        itc_vt_loss = F.cross_entropy(sim_itc_vt, targets_itc)
        itc_loss = 0.5 * ( itc_tv_loss + itc_vt_loss )

        return torch.cat([ent_embs[:,0], self.lp_token], dim = 0), rep_rel_str.squeeze(dim=1), itc_loss
    
    def contrastive_loss_relation(self, rel_embs, loss_flag = True):
        head_scores = self.head_classifier_r(rel_embs)
        tail_scores = self.tail_classifier_r(rel_embs)
        head_pro = torch.sigmoid(head_scores)
        tail_pro = torch.sigmoid(tail_scores)
        self.head_rel_pro = F.softmax(head_pro, dim=-1)
        self.tail_rel_pro = F.softmax(tail_pro, dim=-1)

        if loss_flag:
            head_loss = self.bceloss(head_scores, self.head_valid.T.float())
            tail_loss = self.bceloss(tail_scores, self.tail_valid.T.float())
            return 0.5 * (head_loss + tail_loss)
        else:
            return
    
    def score(self, emb_ent, emb_rel, triplets):
        h_seq = emb_ent[triplets[:,0] - self.num_rel].unsqueeze(dim = 1) + self.pos_head
        r_seq = emb_rel[triplets[:,1] - self.num_ent].unsqueeze(dim = 1) + self.pos_rel
        t_seq = emb_ent[triplets[:,2] - self.num_rel].unsqueeze(dim = 1) + self.pos_tail
        dec_seq = torch.cat([h_seq, r_seq, t_seq], dim = 1)
        output_dec = self.decoder(dec_seq)
        rel_emb = output_dec[:, 1, :]
        ctx_emb = output_dec[triplets == self.num_ent + self.num_rel]

        if self.score_function == "tucker":
            tucker_emb = self.tucker_decoder(ctx_emb, rel_emb)
            scores = torch.mm(tucker_emb, emb_ent[:-1].transpose(1, 0))
            score = scores
        else:
            score = torch.inner(ctx_emb, emb_ent[:-1])
        return score
