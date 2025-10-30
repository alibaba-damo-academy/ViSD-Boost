from copy import deepcopy
import re
import torch
import torch.distributed as dist
from torch import nn
import random
import os
import json
import numpy as np
import torch.nn.functional as F
from transformers import BertTokenizer
from lavis.utils.vit_vae import ViT_VAE, ViT_VAE_Decoder


train_vqvae = False
class ExponentialMovingAverage(nn.Module):
    """Maintains an exponential moving average for a value.
    
      This module keeps track of a hidden exponential moving average that is
      initialized as a vector of zeros which is then normalized to give the average.
      This gives us a moving average which isn't biased towards either zero or the
      initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)
      
      Initially:
          hidden_0 = 0
      Then iteratively:
          hidden_i = hidden_{i-1} - (hidden_{i-1} - value) * (1 - decay)
          average_i = hidden_i / (1 - decay^i)
    """
    
    def __init__(self, init_value, decay):
        super().__init__()
        
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros_like(init_value))
        
    def forward(self, value):
        self.counter += 1
        self.hidden.sub_((self.hidden - value) * (1 - self.decay))
        average = self.hidden / (1 - self.decay ** self.counter)
        return average
        
    
class VectorQuantizerEMA(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. Use EMA to update embeddings.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
        decay (float): decay for the moving averages.
        epsilon (float): small float constant to avoid numerical instability.
    """
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay,
               epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        
        # initialize embeddings as buffers
        embedding_list = []
        for i in range(4):
            embedding_list.append(torch.empty(100, self.embedding_dim))
        embeddings = torch.cat(embedding_list)
        nn.init.xavier_uniform_(embeddings)
        self.register_buffer("embeddings", embeddings)
        self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)
        
        # also maintain ema_cluster_size， which record the size of each embedding
        self.ema_cluster_size = ExponentialMovingAverage(torch.zeros((self.num_embeddings,)), decay)
        
        
    def forward(self, x):
        # [B, L, C]
        x = x.contiguous()
        # [B, L, C] -> [BL, C]
        flat_x = x.reshape(-1, self.embedding_dim)
        
        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x) # [B, L, C]
        
        if not train_vqvae:
            quantized = quantized.contiguous()
            return quantized
        
        # update embeddings with EMA
        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            updated_ema_cluster_size = self.ema_cluster_size(torch.sum(encodings, dim=0))
            n = torch.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                      (n + self.num_embeddings * self.epsilon) * n)
            dw = torch.matmul(encodings.t(), flat_x) # sum encoding vectors of each cluster
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = (
              updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1))
            self.embeddings.data = normalised_updated_ema_w
        
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        
        quantized = quantized.contiguous()
        return quantized, loss
    
    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embeddings.t())
        ) # [N, M]
        encoding_indices = torch.argmin(distances, dim=1) # [N,]
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.embedding(encoding_indices, self.embeddings) 


class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._layers = nn.ModuleList()
        for i in range(num_residual_layers):
            self._layers.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(num_hiddens, num_residual_hiddens, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(num_residual_hiddens, num_hiddens, 3, padding=1),
            ))
                    
    def forward(self, x):
        h = x
        for layer in self._layers:
            conv = layer(h)
            h = h + conv
        return F.relu(h)


class Encoder(nn.Module):
    def __init__(self, in_dim, embedding_dim):
        super(Encoder, self).__init__()
        self._num_hiddens = in_dim
        self._enc_1 = nn.Conv1d(in_dim, self._num_hiddens, 3, stride=1, padding=1)
        self._enc_2 = nn.Conv1d(self._num_hiddens, self._num_hiddens * 2, 3, stride=1, padding=1)
        self._enc_3 = nn.Conv1d(self._num_hiddens * 2, self._num_hiddens * 2, 3, stride=1, padding=1)
        self._enc_4 = nn.Conv1d(self._num_hiddens * 2, embedding_dim, 3, stride=1, padding=1)
        
    def forward(self, x):
        h = F.relu(self._enc_1(x))
        h = F.relu(self._enc_2(h))
        h = F.relu(self._enc_3(h))
        h = self._enc_4(h)
        return h


class Decoder(nn.Module):
    def __init__(self, out_dim, embedding_dim):
        super(Decoder, self).__init__()
        self._num_hiddens = out_dim
        self._dec_1 = nn.Conv1d(embedding_dim, self._num_hiddens * 2, 3, stride=1, padding=1)
        self._dec_2 = nn.Conv1d(self._num_hiddens * 2, self._num_hiddens * 2, 3, stride=1, padding=1)
        self._dec_3 = nn.Conv1d(self._num_hiddens * 2, self._num_hiddens, 3, stride=1, padding=1)
        self._dec_4 = nn.Conv1d(self._num_hiddens, out_dim, 3, stride=1, padding=1)
        
    def forward(self, x):
        h = self._dec_1(x)
        h = F.relu(self._dec_2(h))
        h = F.relu(self._dec_3(h))
        recon = self._dec_4(h)
        return recon

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        return self.encoding[:seq_len, :]
        

class VISDBOOST(nn.Module):
    def __init__(
        self,
        image_encoder,
        text_encoder,
        alpha=0.4,
        embed_dim=256,
        momentum=0.995,
        tie_enc_dec_weights=True,
        max_txt_len=384,
        in_dim=256,
        embedding_dim=768,
        num_embeddings=4*100, 
        data_variance=0.5,
        commitment_cost=0.25,
        decay=0.99
    ):
        super().__init__()
        
        # define VQ-VAE
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.data_variance = data_variance
        self.vq_layer = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
        self.id_projs = nn.Linear(1, in_dim)
        self.encoder = ViT_VAE(
            indim = in_dim,
            dim = embedding_dim,
            depth = 2,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
        self.decoder = ViT_VAE_Decoder(
            indim = in_dim,
            dim = embedding_dim,
            depth = 2,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        self.tokenizer = BertTokenizer.from_pretrained("BiomedVLP-CXR-BERT-specialized")
        text_encoder.resize_token_embeddings(len(self.tokenizer))
        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder

        # creating projection layers for ITC
        text_width = text_encoder.config.hidden_size
        vision_width = 256

        self.text_proj = nn.Linear(text_width, embed_dim)

        self.momentum = momentum
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.alpha = alpha
        self.max_txt_len = max_txt_len

        self.organs = [
            'lung', 'heart', 'esophagus', 'aorta'
        ]
        
        self.attention = nn.MultiheadAttention(
            embed_dim=vision_width,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        self.vision_projs = nn.ModuleList([nn.Linear(vision_width, embed_dim) for _ in range(len(self.organs))])
        self.query_tokens = nn.Parameter(torch.zeros(len(self.organs), vision_width))
        self.res_projs1 = nn.ModuleList([nn.Linear(vision_width*2, vision_width) for _ in range(len(self.organs))])
        self.res_projs2 = nn.ModuleList([nn.Linear(vision_width, vision_width) for _ in range(len(self.organs))])

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))

    @torch.inference_mode()
    def forward_test_win(
        self, 
        images, 
        masks, 
        organ_logits,
        test_organs,
        text_feat_dict,
        organ_feat_dict,
        whole_organ_sizes,
        skip_organ=None
    ):
        image_embeds1, image_embeds2, image_embeds3, image_embeds4, organ_token_flags1, organ_token_flags2, organ_token_flags3, organ_token_flags4 = self.visual_encoder(images, masks.squeeze(1))

        margin = 2
        masks = masks.squeeze(1)
        
        for i, (embed1, embed2, embed3, embed4, mask) in enumerate(zip(image_embeds1, image_embeds2, image_embeds3, image_embeds4, masks)):
            boundaries = []
            for d in range(mask.dim()):
                start_slice = [slice(None)] * mask.dim()
                end_slice = [slice(None)] * mask.dim()
                
                start_slice[d] = slice(None, margin)
                end_slice[d] = slice(-margin, None)
                
                boundaries.append(mask[tuple(start_slice)][mask[tuple(start_slice)] > 0])
                boundaries.append(mask[tuple(end_slice)][mask[tuple(end_slice)] > 0])
            boundaries = torch.cat(boundaries)
            
            boundary_values = boundaries[boundaries > 0].flatten()
            boundary_organs = torch.unique(boundary_values)

            if skip_organ is not None:
                boundary_organs = boundary_organs[boundary_organs != skip_organ + 1]
            
            organ_ids, organ_counts = torch.unique(mask, return_counts=True)
            organ_ids = organ_ids.long()
            organ_counts = organ_counts[organ_ids != 0]
            organ_ids = organ_ids[organ_ids != 0]

            # organs not touch boundary
            intact_organ_ids = [organ_id for organ_id, organ_count in zip(organ_ids, organ_counts) if organ_id not in boundary_organs]
            intact_organ_ids = torch.tensor(intact_organ_ids, device=masks.device).long()
            intact_organ_ids = intact_organ_ids - 1
            
            if not len(intact_organ_ids):
                continue

            organ_sizes = dict(zip([self.organs[organ_id] for organ_id in intact_organ_ids], [organ_counts[organ_ids == organ_id + 1].item() for organ_id in intact_organ_ids]))

            for k, v in organ_sizes.items():
                if k in whole_organ_sizes and v / whole_organ_sizes[k] != 1:
                    print(f'Mask id: {i}', f'Rank: {dist.get_rank() if dist.is_initialized() else 0}', 'Incomplete', k, v / whole_organ_sizes[k])
                    if v / whole_organ_sizes[k] < 0.9 and self.organs.index(k) != skip_organ:
                        intact_organ_ids = intact_organ_ids[intact_organ_ids != self.organs.index(k)]
            
            for organ_id in intact_organ_ids:
                organ_name = self.organs[organ_id.item()]
                if organ_name not in test_organs:
                    continue
                
                tokens1 = organ_token_flags1[i, organ_id, :]
                tokens2 = organ_token_flags2[i, organ_id, :]
                tokens3 = organ_token_flags3[i, organ_id, :]
                tokens4 = organ_token_flags4[i, organ_id, :]

                key1 = embed1[tokens1].unsqueeze(0)
                key2 = embed2[tokens2].unsqueeze(0)
                key3 = embed3[tokens3].unsqueeze(0)
                key4 = embed4[tokens4].unsqueeze(0)
                
                roi_embeds, roi_embeds_other = key4, torch.cat([key1, key2, key3], dim=1)
                roi_embeds_clone = roi_embeds.clone()
                
                id_embeds = self.id_projs(organ_id.unsqueeze(0).unsqueeze(0).float()).repeat(roi_embeds.shape[0], 1, 1)
                roi_embeds_min = roi_embeds_clone.min()
                roi_embeds_clone = roi_embeds_clone - roi_embeds_min
                roi_embeds_max = roi_embeds_clone.max()
                roi_embeds_clone = (roi_embeds_clone / roi_embeds_max) - 0.5
                roi_embeds_input = torch.cat([id_embeds, roi_embeds_clone], dim=1)
            
                z = self.encoder(roi_embeds_input)
                z = z[:, 1:, :]  # remove the condition
                e = self.vq_layer(z)
                x_recon = self.decoder(e)
                
                recon_feat = (x_recon + 0.5) * roi_embeds_max + roi_embeds_min
                
                origianl_embeds = torch.cat([roi_embeds_other, roi_embeds], dim=1)
                vae_embeds = torch.cat([roi_embeds_other, recon_feat], dim=1)
                hybrid_feat = torch.cat([origianl_embeds, vae_embeds], dim=-1)
                hybrid_feat = self.res_projs1[organ_id](hybrid_feat)
                hybrid_feat = self.res_projs2[organ_id](hybrid_feat)
                image_feat_res = self.get_query_token(hybrid_feat, organ_id)
                image_feat_res = self.vision_projs[organ_id](image_feat_res)
                image_feat = F.normalize(image_feat_res, dim=-1)
            
                organ_feat_dict[organ_name] = image_feat.cpu().tolist()

                for item in organ_logits.keys():
                    if item[0] != organ_name:
                        continue

                    text_feat = text_feat_dict[item]

                    logits = image_feat @ text_feat.t() / self.temp
                    probs = logits.softmax(-1)
                    organ_logits[item].append(probs.cpu().tolist())
    
        return organ_logits

    def get_query_token(self, image_embeds, cl_organ_id):
        query = self.query_tokens[cl_organ_id].unsqueeze(0).unsqueeze(0)
        query = query.repeat(image_embeds.shape[0], 1, 1)

        roi_feats, _ = self.attention(query, image_embeds, image_embeds)

        return roi_feats.squeeze(1)
    
    def prepare_text_feat(self, test_items, length=None):
        if length is None:
            length = self.max_txt_len

        device = self.text_encoder.device
        text_feat_dict = {}
        for prompt, item in zip(*self._get_prompt(test_items)):
            text = self.tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=length,
                return_tensors="pt",
            ).to(device)

            text_output = self.text_encoder.forward_text(text)
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
            text_feat_dict[tuple(item)] = text_feat

        return text_feat_dict

    @staticmethod
    def _get_prompt(
        test_items,
        organ_name: str = None
    ) -> str:
        if organ_name is not None:
            test_items = [item for item in test_items if item[0] == organ_name]

        negative_prompts = [item[2] for item in test_items]
        positive_prompts = [item[3] for item in test_items]

        prompts = list(zip(negative_prompts, positive_prompts))

        return prompts, test_items
