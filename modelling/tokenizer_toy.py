# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modelling.modules import Encoder, Decoder, TimmViTEncoder, TimmViTDecoder
from modelling.quantizers.vq import VectorQuantizer
from modelling.quantizers.kl import DiagonalGaussianDistribution
from modelling.quantizers.softvq import SoftVectorQuantizer

from timm import create_model


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


@dataclass
class ModelArgs:
    image_size: int = 256
    base_image_size: int = 256
    
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    vq_loss_ratio: float = 1.0 # for soft vq
    kl_loss_weight: float = 0.000001
    tau: float = 0.1
    num_codebooks: int = 1
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0

    enc_type: str = 'cnn'
    dec_type: str = 'cnn'
    encoder_model: str = 'llamagen_encoder'
    decoder_model: str = 'llamagen_decoder'
    num_latent_tokens: int = 256
    to_pixel: str = 'linear'
    
    # for pre-trained models
    enc_tuning_method: str = 'full'
    dec_tuning_method: str = 'full'
    enc_pretrained: bool = True
    dec_pretrained: bool = False 
    
    # for vit 
    enc_patch_size: int = 16
    dec_patch_size: int = 16
    enc_drop_path_rate: float = 0.0
    dec_drop_path_rate: float = 0.0
    
    # deocder cls token
    dec_cls_token: bool = True
    
    # rope
    use_ape: bool = True 
    use_rope: bool = True
    rope_mixed: bool = True
    rope_theta: float = 10.0
    
    # repa for vit
    repa: bool = False
    repa_patch_size: int = 16
    repa_model: str = 'vit_base_patch16_224'
    repa_proj_dim: int = 2048
    repa_loss_weight: float = 0.1
    repa_align: str = 'global'
    
    vq_mean: float = 0.0
    vq_std: float = 1.0
    
    # encoder token drop for mask modeling
    enc_token_drop: float = 0.0
    enc_token_drop_max: float = 0.6

    # aux decoder model
    aux_dec_model: str = 'vit_tiny_patch14_dinov2_movq'
    aux_loss_mask: bool = False
    aux_dec_cls_token: bool = True
    aux_hog_dec: bool = True
    aux_dino_dec: bool = True
    aux_supcls_dec: bool = True
    aux_clip_dec: bool = True
    

class VQModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vq_mean = config.vq_mean
        self.vq_std = config.vq_std
        self.num_latent_tokens = config.num_latent_tokens
        self.codebook_embed_dim = config.codebook_embed_dim
        
        self.repa = config.repa
        self.repa_loss_weight = config.repa_loss_weight
        self.repa_align = config.repa_align
        repa_z_dim = None
        
        
        if config.enc_type == 'cnn':
            if config.encoder_model == 'llamagen_encoder':
                self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
            else:
                raise NotImplementedError
            self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        elif config.enc_type == 'vit':
            self.encoder = TimmViTEncoder(
                in_channels=3, num_latent_tokens=config.num_latent_tokens,
                model_name=config.encoder_model,  # 'vit_small_patch14_dinov2.lvd142m', #'vit_base_patch14_dinov2.lvd142m',  #
                model_kwargs={'img_size': config.image_size, 'patch_size': config.enc_patch_size, 'drop_path_rate': config.enc_drop_path_rate},
                pretrained=config.enc_pretrained,
                tuning_method=config.enc_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                token_drop=config.enc_token_drop,
                token_drop_max=config.enc_token_drop_max,
                base_img_size=config.base_image_size
            )
            self.quant_conv = nn.Linear(self.encoder.embed_dim, config.codebook_embed_dim)
            
        
        if config.dec_type == 'cnn':
            if config.decoder_model == 'llamagen_decoder':
                self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
            else:
                raise NotImplementedError
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)
        elif config.dec_type == 'vit':
            self.decoder = TimmViTDecoder(
                in_channels=3, num_latent_tokens=config.num_latent_tokens,
                model_name=config.decoder_model,  # 'vit_small_patch14_dinov2.lvd142m', #'vit_base_patch14_dinov2.lvd142m',  #
                model_kwargs={'img_size': config.image_size, 'patch_size': config.dec_patch_size, 'drop_path_rate': config.dec_drop_path_rate, 'latent_dim': config.codebook_embed_dim},
                pretrained=config.dec_pretrained,
                tuning_method=config.dec_tuning_method,
                tuning_kwargs={'r': 8},
                use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
                cls_token=config.dec_cls_token,
                codebook_embed_dim=config.codebook_embed_dim,
                to_pixel=config.to_pixel,
                base_img_size=config.base_image_size
            )
            self.post_quant_conv = nn.Linear(config.codebook_embed_dim, self.decoder.embed_dim)
        # check movq
        if 'movq' in config.decoder_model:
            self.use_movq = True 
        else:
            self.use_movq = False
        
        
        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        
        if self.repa and self.training:
            # get z from repa_encoder
            rescale_x = self.scale(self.de_scale(x))
            z = self.repa_model.forward_features(rescale_x)[:, self.repa_model.num_prefix_tokens:]

            # taking average over spatial dimension
            if self.repa_align == 'global':
                z = z.mean(dim=1)
                z_hat = quant.mean(dim=1)
                # calculate repa loss
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'avg_1d':
                z = F.adaptive_avg_pool1d(z.permute(0, 2, 1), quant.shape[1]).permute(0, 2, 1)
                z_hat = quant
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'avg_1d_shuffle':
                # shuffle the length dimension of z and avg
                indices = torch.randperm(z.shape[1])
                z = F.adaptive_avg_pool1d(z[:, indices, :].permute(0, 2, 1) , quant.shape[1]).permute(0, 2, 1)
                z_hat = quant
                z_hat = self.projection(z_hat)
            elif self.repa_align == 'repeat':
                z_hat = self.projection(quant)
                b, l, d = z_hat.shape
                z_hat = z_hat.unsqueeze(2).expand(-1, -1, z.size(1) // l, -1).reshape(b, -1, d)
            

            z = F.normalize(z, dim=-1)
            z_hat = F.normalize(z_hat, dim=-1)
            proj_loss = mean_flat(-(z * z_hat).sum(dim=-1))
            proj_loss = proj_loss.mean()
            proj_loss *= self.repa_loss_weight
            
            emb_loss += (proj_loss,)
        
        return quant, emb_loss, info

    def decode(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv(quant)
        if self.use_movq:
            dec = self.decoder(quant, tmp_quant, h, w)
        else:
            dec = self.decoder(quant, None, h, w)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        b, _, h, w = input.size()
        quant, diff, info = self.encode(input)
        self.quant = quant
        dec = self.decode(quant, x=input, h=h, w=w)
        return dec, diff, info


class DynamicAEModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs):
        # config.repa = True # for dinov2 decoder
        super().__init__()
        self.config = config
        self.vq_mean = config.vq_mean
        self.vq_std = config.vq_std
        self.num_latent_tokens = config.num_latent_tokens
        self.codebook_embed_dim = config.codebook_embed_dim
        
        assert config.enc_type == 'vit'
        self.encoder = TimmViTEncoder(
            in_channels=3, num_latent_tokens=config.num_latent_tokens,
            model_name=config.encoder_model,
            model_kwargs={'img_size': config.image_size, 'patch_size': config.enc_patch_size, 'drop_path_rate': config.enc_drop_path_rate},
            pretrained=config.enc_pretrained,
            tuning_method=config.enc_tuning_method,
            tuning_kwargs={'r': 8},
            use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
            token_drop=config.enc_token_drop,
            token_drop_max=config.enc_token_drop_max,
            base_img_size=config.base_image_size
        )
            
        assert config.dec_type == 'vit'
        self.decoder = TimmViTDecoder(
            in_channels=3, num_latent_tokens=config.num_latent_tokens,
            model_name=config.decoder_model,
            model_kwargs={'img_size': config.image_size, 'patch_size': config.dec_patch_size, 'drop_path_rate': config.dec_drop_path_rate, 'latent_dim': config.codebook_embed_dim},
            pretrained=config.dec_pretrained,
            tuning_method=config.dec_tuning_method,
            tuning_kwargs={'r': 8},
            use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
            cls_token=config.dec_cls_token,
            codebook_embed_dim=config.codebook_embed_dim,
            to_pixel=config.to_pixel,
            base_img_size=config.base_image_size
        )

        self.use_movq = False
        self.quantize = None
        assert config.aux_hog_dec == False
        assert config.aux_dino_dec == False
        assert config.aux_clip_dec == False

    def encode(self, x):
        if self.training:
            h, mask, mask_loss = self.encoder(x, return_mask=True)
        else:
            h = self.encoder(x)
        info = None
        if self.training:
            return h, mask_loss, info, mask
        else:
            return h, mask_loss, info

    def decode(self, h, x=None, h_size=None, w=None):
        dec = self.decoder(h, None, h_size, w)
        return dec

    def forward(self, input):
        b, _, h, w = input.size()
        if self.training:
            latent, diff, info, mask = self.encode(input)
        else:
            latent, diff, info = self.encode(input)
        self.quant = latent
        # print(quant.shape)
        dec = self.decode(latent, x=input, h_size=h, w=w)
        
        return dec, diff, info


def DynamicAE_16(**kwargs):
    return DynamicAEModel(ModelArgs(encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs))

VQ_models = {
    'DynamicAE-16': DynamicAE_16,
    }


if __name__ == '__main__':
    
    # model = DynamicAE_16(
    #     codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_tiny_patch14_dinov2.lvd142m', 
    #     decoder_model='vit_tiny_patch14_dinov2.lvd142m', num_codebooks=4, codebook_size=16384, 
    #     aux_dec_model='vit_tinytiny_patch14_dinov2_movq2', aux_loss_mask=True, aux_dec_cls_token=True, aux_hog_dec=True, 
    #     aux_dino_dec=True, aux_clip_dec=True, enc_token_drop=0.4, enc_token_drop_max=0.6)      
    model = DynamicAE_16(
        codebook_embed_dim=16, enc_type='vit', dec_type='vit', encoder_model='vit_tiny_patch16_224', 
        decoder_model='vit_tiny_patch16_224', num_codebooks=4, codebook_size=16384, 
        enc_token_drop=0.4, enc_token_drop_max=0.6)      
    model.train()
    model = model.cuda()
    x = torch.randn(4, 3, 256, 256).cuda()
    y, _, info = model(x)
    print(_)
