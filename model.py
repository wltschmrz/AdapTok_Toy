import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field, fields
from typing import List
from huggingface_hub import PyTorchModelHubMixin


args = {}





def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)





class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x





class VisionTransformerDiffPruning(nn.Module):
    def __init__(
            self, 
            img_size=128, patch_size=16, in_chans=3, num_classes=1000,
            embed_dim=192, depth=8, num_heads=2, mlp_ratio=4., 
            qkv_bias=True, qk_scale=None, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
            hybrid_backbone=None, norm_layer=None, 
            pruning_loc=None, distill=False
            ):
        """
        Args:
            img_size (int | tuple)             : 입력 image의 크기
            patch_size (int | tuple)           : patch의 크기
            in_chans (int)                     : 입력 image의 channel 수
            num_classes (int)                  : classification head의 출력 class 수
            embed_dim (int)                    : token embedding의 차원
            depth (int)                        : transformer block의 개수
            num_heads (int)                    : multi-head self-attention에서의 head 수
            mlp_ratio (float)                  : MLP hidden dimension이 embedding dimension 대비 얼마나 큰지의 비율
            qkv_bias (bool)                    : QKV projection에 bias term을 추가할지 여부
            qk_scale (float | None)            : QK attention score의 scaling factor (기본값은 head_dim^{-0.5})
            representation_size (int | None)   : classification 이전에 사용하는 hidden representation 차원 (None이면 사용 안 함)
            drop_rate (float)                  : 전체 dropout 비율
            attn_drop_rate (float)             : attention weight에 대한 dropout 비율
            drop_path_rate (float)             : stochastic depth(drop path)의 비율
            hybrid_backbone (nn.Module | None) : PatchEmbed 모듈 대신 사용할 CNN 기반 backbone
            norm_layer (nn.Module | None)      : normalization layer 클래스 (예: nn.LayerNorm)
            pruning_loc (list[int] | None)     : pruning이 적용되는 transformer block index들의 리스트
            distill (bool)                     : knowledge distillation용 설정 여부
        """

        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = embed_dim  # num_features for consistency with other models
        self.num_heads = num_heads
        self.distill = distill
        self.pruning_loc = pruning_loc
        self.token_ratio = 0
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Patch embedding
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Pruning router
        self.score_predictor = nn.ModuleList([
            PrefixRouter(embed_dim) for _ in range(len(pruning_loc))
        ])

        # Routing hyperparameters
        self.router_tau = 1.0       # softmax temperature for routing decisions
        self.use_ste = True
        self.lambda_ent = 0.0       # 엔트로피 패널티를 쓸 경우 외부 loss에서 사용

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        B, _, h, w = x.shape
        x = self.patch_embed(x)

        # cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        p_count = 0
        out_pred_prob = []
        patch_length = H = 8 * 8
        prev_decision = torch.ones(B, H, 1, dtype=x.dtype, device=x.device)
        policy = torch.ones(B, H + 1, 1, dtype=x.dtype, device=x.device)
        for i, blk in enumerate(self.blocks):
            if i in self.pruning_loc:
                spatial_x = x[:, 1:]
                pred_score = self.score_predictor[p_count](spatial_x, prev_decision).reshape(B, -1, 2)
                if self.training:
                    hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
                    out_pred_prob.append(hard_keep_decision.reshape(B, H))
                    cls_policy = torch.ones(B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                    policy = torch.cat([cls_policy, hard_keep_decision], dim=1)
                    x = blk(x, policy=policy)
                    prev_decision = hard_keep_decision
                else:
                    score = pred_score[:,:,0]
                    num_keep_node = int(H * self.token_ratio[p_count])
                    keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_node]
                    cls_policy = torch.zeros(B, 1, dtype=keep_policy.dtype, device=keep_policy.device)
                    now_policy = torch.cat([cls_policy, keep_policy + 1], dim=1)
                    x = batch_index_select(x, now_policy)
                    prev_decision = batch_index_select(prev_decision, keep_policy)
                    x = blk(x)
                p_count += 1
            else:
                if self.training:
                    x = blk(x, policy)
                else:
                    x = blk(x)

        x = self.norm(x)
        features = x[:, 1:]
        x = x[:, 0]
        x = self.pre_logits(x)
        x = self.head(x)
        if self.training:
            if self.distill:
                return x, features, prev_decision.detach(), out_pred_prob
            else:
                return x, out_pred_prob
        else:
            return x





class TimmViTEncoder(nn.Module):
    def __init__(self, in_channels=3, num_latent_tokens=16,
                 model_name='vit_tiny_patch16_224',
                 model_kwargs={'img_size': 128, 'patch_size': 16, 'drop_path_rate': 0.0,},
                 pretrained=False, tuning_method='full', tuning_kwargs={'r': 8},
                 rope_theta=100.0, rope_mixed=True, use_rope=True, use_ape=False,
                 token_drop=0.4,
                 token_drop_max=0.6,
                 base_img_size=128
                 ):
        super().__init__()

        self.model_name = model_name

        # parameters
        self.num_latent_tokens = num_latent_tokens

        print(f"model_kwargs:")
        print(model_kwargs)
        
        ######
        # load model
        base_rate = 0.9
        PRUNING_LOC = [2, 4, 6]
        model = VisionTransformerDiffPruning(
            img_size=128,
            patch_size=16, embed_dim=48, depth=8, num_heads=4, mlp_ratio=4, qkv_bias=True, 
            pruning_loc=PRUNING_LOC
            )
        model.num_prefix_tokens=0

        self.img_size = model_kwargs['img_size']
        self.patch_size = model_kwargs['patch_size']
        self.embed_dim = model.embed_dim
        # get num of img tokens
        self.num_img_tokens = model.patch_embed.num_patches
        self.num_prefix_tokens = model.num_prefix_tokens
        
        # tuning method
        assert tuning_method == 'full'
        self.model = model

        assert self.num_latent_tokens
        # latent tokens
        self.latent_tokens = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
        nn.init.normal_(self.latent_tokens, std=.02)
        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
        trunc_normal_(self.latent_pos_embed, std=.02)

        # token drop
        self.token_drop = token_drop > 0.0
        if self.token_drop:
            # self.mask_ratio_generator = stats.truncnorm((1.0 - token_drop) / 0.25, 1.0 / 0.25, loc=1.0, scale=0.25)
            self.mask_ratio_generator = stats.truncnorm((token_drop - token_drop_max) / 0.25, 0, loc=token_drop_max, scale=0.25)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, model.embed_dim))
            nn.init.normal_(self.mask_token, std=.02)

        # rope
        self.use_ape = use_ape
        self.use_rope = use_rope
        if self.use_rope:
            self.use_ape = False
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        
        assert self.rope_mixed and self.use_rope, f"{self.rope_mixed}, {self.use_rope}"
        self.compute_cis = partial(compute_mixed_cis, num_heads=model.num_heads)
        freqs = []
        for i, _ in enumerate(model.blocks):
            freqs.append(init_random_2d_freqs(dim=model.embed_dim // model.num_heads, num_heads=model.num_heads, theta=self.rope_theta))
        freqs = torch.stack(freqs, dim=1).view(2, len(model.blocks), -1)
        self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
        
        if base_img_size != model_kwargs['img_size']:
            t_x, t_y = init_t_xy(end_x = base_img_size // model_kwargs['patch_size'] , end_y =  base_img_size //  model_kwargs['patch_size'] )
        else:
            t_x, t_y = init_t_xy(end_x = model_kwargs['img_size'] // model_kwargs['patch_size'] , end_y =  model_kwargs['img_size'] //  model_kwargs['patch_size'] )
        self.register_buffer('freqs_t_x', t_x)
        self.register_buffer('freqs_t_y', t_y)

        if not self.use_ape:
            for b in self.model.blocks:
                b.attn.flash_attn = False
        
    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'latent_tokens', 'latent_pos_embed', 'freqs']

    def sample_orders(self, bsz, seq_len):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward(self, x, return_mask=False, gumbel_tau=1.0, ):

        # get tokens
        B, _, H, W = x.shape
        x = self.model.patch_embed(x)

        if self.token_drop and self.training:
            orders = self.sample_orders(bsz=x.size(0), seq_len=x.size(1)).to(x.device)
            mask = self.random_masking(x, orders).unsqueeze(-1)
            x = torch.where(mask.bool(), self.mask_token, x)
        else:
            mask = None
        
        assert not 'eva02' in self.model_name
        # x = self.model._pos_embed(x)
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)

        patch_length = x.size(1)

        # insert latent tokens
        z = self.latent_tokens.expand(B, -1, -1)
        x = torch.cat([x, z + self.latent_pos_embed], dim=1)
            
        # pre layer norm
        x = self.model.norm(x)
        
        p_count = 0
        losses = []
        prev_decision = torch.ones(B, patch_length + self.num_latent_tokens, 1, dtype=x.dtype, device=x.device)
        patch_policy = torch.ones(B, patch_length, 1, dtype=x.dtype, device=x.device)
        # if self.use_ape: 
        #     for i, blk in enumerate(self.model.blocks):
        #         x = blk(x)
        assert self.rope_mixed and self.use_rope
        if self.freqs_t_x.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
            t_x, t_y = init_t_xy(end_x = W // self.patch_size, end_y = H // self.patch_size)
            t_x, t_y = t_x.to(x.device), t_y.to(x.device)
        else:
            t_x, t_y = self.freqs_t_x, self.freqs_t_y
        freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
        
        for i , blk in enumerate(self.model.blocks):
            if i in self.model.pruning_loc:
                # latent_x = x[:, -self.num_latent_tokens:]
                logits = self.model.score_predictor[p_count](  # [128, 16]
                    x, 
                    patch_policy=prev_decision[:, :-self.num_latent_tokens, :],
                    latent_policy=prev_decision[:, -self.num_latent_tokens:, :],
                    )  # (BL) <- (BN2)
                if self.training:
                    p_soft = F.gumbel_softmax(logits, tau=gumbel_tau, hard=False, dim=1)  # (B, L) [128, 16]
                    large_pos = torch.argmax(p_soft, dim=1, keepdim=True)  # (B, 1)
                    cumsum_p = torch.cumsum(p_soft, dim=1)
                    keep_soft = 1.0 - cumsum_p  # (B, L) [128, 16]
                    mask_loss = keep_soft.sum(dim=1, keepdim=True) / self.num_latent_tokens  # [128, 1]
                    pos = torch.gather(keep_soft, 1, large_pos)  # [128, 128]?
                    keep_hard = (keep_soft >= pos).float()
                    keep_mask = (keep_hard - keep_soft).detach() + keep_soft   # (B, L)
                    hard_keep_decision = keep_mask.reshape(B, self.num_latent_tokens, 1) * prev_decision[:, -self.num_latent_tokens:, :]
                    # mask_loss  (B, 1)

                    # losses.append(mask_loss.reshape(B, 1))
                    losses.append(mask_loss.mean())
                    policy = torch.cat([patch_policy, hard_keep_decision], dim=1)
                    x = blk(x, policy=policy, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                    prev_decision = policy
                else:
                    with torch.no_grad():
                        p_soft = F.softmax(logits, dim=1)
                        large_pos = torch.argmax(p_soft, dim=1)

                        cumsum_p = torch.cumsum(p_soft, dim=1)
                        keep_soft = 1.0 - cumsum_p

                        pos = keep_soft[:, large_pos].item()
                        keep_hard = (keep_soft >= pos).float()
                        keep_mask = (keep_hard - keep_soft).detach() + keep_soft   # (B, L)
                        hard_keep_decision = keep_mask.reshape(B, self.num_latent_tokens, 1) * prev_decision[:, -self.num_latent_tokens:, :]
                        
                        now_policy = torch.cat([patch_policy, hard_keep_decision], dim=1)
                        x = batch_index_select(x, now_policy)
                        prev_decision = batch_index_select(prev_decision, now_policy)
                        x = blk(x, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                p_count += 1

        # x = self.model.blocks(x)
        x = self.model.norm(x)

        # get z tokens as out
        out = x[:, -self.num_latent_tokens:]
        
        if return_mask:
            return out, mask, losses
        else:
            return out



class TimmViTDecoder(nn.Module):
    def __init__(self, in_channels=3,
                 model_name='vit_tiny_patch16_224',
                 model_kwargs={'img_size': 128, 'patch_size': 16, 'drop_path_rate': 0.0, 'embed_dim': 48}, pretrained=True,
                 tuning_method='lora', tuning_kwargs={'r': 8},
                 num_latent_tokens=16, to_pixel='linear',
                 codebook_embed_dim=32,
                 rope_theta=100.0, rope_mixed=False, use_rope=False, use_ape=True,
                 cls_token=True,
                 base_img_size=128,
                 ):
        super().__init__()

        # model_kwargs['num_latent_tokens'] = num_latent_tokens
        # model_kwargs['class_token'] = cls_token
        model = create_model(
            model_name,
            pretrained=pretrained,
            **model_kwargs
        )
        
        self.patch_size = model_kwargs['patch_size']
        self.embed_dim = model.embed_dim = 48
        # get num of img tokens
        self.num_img_tokens = model.patch_embed.num_patches
        self.num_prefix_tokens = model.num_prefix_tokens
        self.num_latent_tokens = num_latent_tokens
        
        # tuning method
        if tuning_method == 'full':
            # doing nothing
            self.model = model
        elif tuning_method == 'lora':
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d",
                                     modules_to_save=['patch_embed.proj', 'patch_embed.norm', 'norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(model, config)
            # self.model.base_model.model.pos_embed.requires_grad = True
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in model.parameters():
                param.requires_grad = False
            self.model = model

        # latent tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model.embed_dim))
        nn.init.normal_(self.mask_token, std=.02)

        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
        trunc_normal_(self.latent_pos_embed, std=.02)

        # to pixel
        self.to_pixel = ToPixel(to_pixel=to_pixel, img_size=model_kwargs['img_size'], in_channels=in_channels,
                                in_dim=model.embed_dim, patch_size=model_kwargs['patch_size'])

        
        self.use_ape = use_ape
        self.use_rope = use_rope
        if self.use_rope:
            self.use_ape = False
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        
        if self.rope_mixed and self.use_rope:
            self.compute_cis = partial(compute_mixed_cis, num_heads=model.num_heads)
            
            freqs = []
            for i, _ in enumerate(model.blocks):
                freqs.append(
                    init_random_2d_freqs(dim=model.embed_dim // model.num_heads, num_heads=model.num_heads, theta=self.rope_theta)
                )
            freqs = torch.stack(freqs, dim=1).view(2, len(model.blocks), -1)
            self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)
            
            if base_img_size != model_kwargs['img_size']:
                t_x, t_y = init_t_xy(end_x = base_img_size // model_kwargs['patch_size'] , end_y =  base_img_size //  model_kwargs['patch_size'] )
            else:
                t_x, t_y = init_t_xy(end_x = model_kwargs['img_size'] // model_kwargs['patch_size'] , end_y =  model_kwargs['img_size'] //  model_kwargs['patch_size'] )
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
        elif not self.rope_mixed and self.use_rope:
            self.compute_cis = partial(compute_axial_cis, dim=model.embed_dim//model.num_heads, theta=rope_theta)
            
            freqs_cis = self.compute_cis(end_x = model_kwargs['img_size'] // model_kwargs['patch_size'] , end_y = model_kwargs['img_size'] //  model_kwargs['patch_size'] )
            self.freqs_cis = freqs_cis
            
        if not self.use_ape:
            for b in self.model.blocks:
                b.attn.flash_attn = False


        if 'movq' in model_name:
            self.use_movq = True 
            self.model.norm = MoVQNorm(codebook_embed_dim, model.embed_dim)

            # Zero-out adaLN modulation layers in DiT blocks:
            for block in self.model.blocks:
                if isinstance(block, MoVQBlockv2):
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            # Zero-out output layers:
            if isinstance(self.model.norm, MoVQNorm):
                nn.init.constant_(self.model.norm.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(self.model.norm.adaLN_modulation[-1].bias, 0)
        else:
            self.use_movq = False 
            

        self.cls_token = cls_token
        if not cls_token:
            self.model.cls_token = None
            self.num_prefix_tokens -= 1
            self.model.num_prefix_tokens -= 1
            
    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'mask_token', 'latent_pos_embed', 'freqs']

    @property
    def last_layer(self):
        return self.to_pixel.get_last_layer()


    def forward(self, z, interpolate_zq=None, H=None, W=None):

        if H is None:
            num_img_tokens = self.num_img_tokens
            H = W = int(math.sqrt(num_img_tokens)) * self.patch_size
        else:
            num_img_tokens = H * W // self.patch_size ** 2

        # mask tokens
        if self.num_latent_tokens:
            if H is None:
                x = self.mask_token.expand(z.size(0), num_img_tokens, -1)
            else:
                x = self.mask_token.expand(z.size(0), H * W // self.patch_size ** 2, -1)
        else:
            x = z 
            
        x = self.model._pos_embed(x, use_ape=self.use_ape)
        x = self.model.patch_drop(x)
        
        z = z + self.latent_pos_embed
        x = torch.cat([x, z], dim=1)

        x = self.model.norm_pre(x)
        
        
        if self.use_ape: 
            for i, blk in enumerate(self.model.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq=interpolate_zq, num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x)
                
        elif self.rope_mixed and self.use_rope:
            if self.freqs_t_x.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
                t_x, t_y = init_t_xy(end_x = W // self.patch_size, end_y = H // self.patch_size)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            else:
                t_x, t_y = self.freqs_t_x, self.freqs_t_y
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
            
            for i , blk in enumerate(self.model.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)

        else:
            if self.freqs_cis.shape[0] != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens:
                freqs_cis = self.compute_cis(end_x = W // self.patch_size, end_y = H // self.patch_size)
            else:
                freqs_cis = self.freqs_cis
            freqs_cis = freqs_cis.to(x.device)
            
            for i , blk in enumerate(self.model.blocks):
                if self.use_movq:
                    x = blk(x, interpolate_zq,  freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
                else:
                    x = blk(x, freqs_cis=freqs_cis[i], num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)      

        if self.use_movq:
            x = self.model.norm(x, interpolate_zq,  num_prefix_tokens=self.num_prefix_tokens, num_latent_tokens=self.num_latent_tokens)
        else:
            x = self.model.norm(x)

        x = x[:, self.num_prefix_tokens:self.num_img_tokens + self.num_prefix_tokens]

        out = self.to_pixel(x)

        return out

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
    
    # encoder token drop for mask modeling
    enc_token_drop: float = 0.0
    enc_token_drop_max: float = 0.6

class AdapTok(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.num_latent_tokens = config.num_latent_tokens
        self.codebook_embed_dim = config.codebook_embed_dim
        
        self.encoder = TimmViTEncoder(
            in_channels=3, num_latent_tokens=config.num_latent_tokens, model_name=config.encoder_model,
            model_kwargs={'img_size': config.image_size, 'patch_size': config.enc_patch_size, 'drop_path_rate': config.enc_drop_path_rate},
            pretrained=config.enc_pretrained,
            tuning_method=config.enc_tuning_method,
            tuning_kwargs={'r': 8},
            use_ape=config.use_ape, use_rope=config.use_rope, rope_mixed=config.rope_mixed, rope_theta=config.rope_theta,
            token_drop=config.enc_token_drop,
            token_drop_max=config.enc_token_drop_max,
            base_img_size=config.base_image_size
        )
            
        self.decoder = TimmViTDecoder(
            in_channels=3, num_latent_tokens=config.num_latent_tokens, model_name=config.decoder_model,
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

    def encode(self, x):
        info = None
        if self.training:
            h, mask, mask_loss = self.encoder(x, return_mask=True)
            return h, mask_loss, info, mask
        else:
            h = self.encoder(x)
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


# ---






if __name__ == "__main__":
    import argparse

    def extract_model_args(args: argparse.Namespace, cls=ModelArgs) -> ModelArgs:
        arg_dict = vars(args)                                               # argparse.Namespace → dict
        valid_keys = {f.name for f in fields(cls)}                          # ModelArgs가 받는 key 목록 추출
        filtered = {k: v for k, v in arg_dict.items() if k in valid_keys}   # 불필요한 키 제거
        return cls(**filtered)

    model_args = extract_model_args(args)  # 안전하게 생성

    # create and load model
    vq_model = AdapTok(model_args)  # 모델 생성








