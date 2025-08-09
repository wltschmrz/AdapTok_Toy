import os, sys, warnings
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from modelling.tokenizer_toy import VQ_models
from utils.misc import load_model_state_dict
import argparse
import ruamel.yaml as yaml
from torchvision.datasets import ImageFolder
from utils.data import random_crop_arr, center_crop_arr
import time
import logging

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
warnings.filterwarnings('ignore')

def create_logger(log_dir=None, log_name="default"):
    os.makedirs(log_dir, exist_ok=True) if log_dir else None

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console 출력 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 저장 핸들러
    if log_dir:
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{log_name}.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = create_logger(None)

# ---------------------------
# Argument Configuration
# ---------------------------
class InferenceArgs:
    vq_model = "DynamicAE-16"
    vq_ckpt = "experiments/tokenizer/exp004-adaptok-t-16/checkpoints/0010000.pt"  # <- Replace with actual path
    data_path = "data/imagenet10p"     # <- Replace with actual evaluation data path
    config = "experiments/tokenizer/exp004-adaptok-t-16/config.yaml"
    config2 = "configs/adaptok-t-16.yaml"

    image_size = 128
    batch_size = 8

    grid = "imagenet10p_real"
    save_dir = "inference_results/" + grid
    info = grid

    codebook_size = 16384
    codebook_embed_dim = 8
    codebook_l2_norm = True
    commit_loss_beta = 0.25
    entropy_loss_ratio = 0.0
    vq_loss_ratio = 1.0
    kl_loss_weight = 0.000001
    dropout_p = 0.0

args = InferenceArgs()

if args.config is not None:
    with open(args.config, 'r', encoding='utf-8') as f:
        file_yaml = yaml.YAML()
        config_args = file_yaml.load(f)
        for k, v in config_args.items():
            if not hasattr(args, k):
                setattr(args, k, v)


assert torch.cuda.is_available(), "GPU is required for inference."
device = torch.device("cuda")


# create and load model
vq_model = VQ_models[args.vq_model](
        image_size=args.image_size,                             # 256
        codebook_size=args.codebook_size,                       #
        codebook_embed_dim=args.codebook_embed_dim,             #
        codebook_l2_norm=args.codebook_l2_norm,                 #
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        vq_loss_ratio=args.vq_loss_ratio,
        kl_loss_weight=args.kl_loss_weight,
        dropout_p=args.dropout_p,
        encoder_model=args.encoder_model,
        decoder_model=args.decoder_model,
        num_latent_tokens=args.num_latent_tokens,
        enc_tuning_method=args.encoder_tuning_method,
        dec_tuning_method=args.decoder_tuning_method,
        enc_pretrained=args.encoder_pretrained,
        dec_pretrained=args.decoder_pretrained,
        enc_patch_size=args.encoder_patch_size,
        dec_patch_size=args.decoder_patch_size,
        tau=args.tau,
        repa=args.repa,
        repa_model=args.repa_model,
        repa_patch_size=args.repa_patch_size,
        repa_proj_dim=args.repa_proj_dim,
        repa_loss_weight=args.repa_loss_weight,
        repa_align=args.repa_align,
        num_codebooks=args.num_codebooks,
        # encoder mask modeling
        enc_token_drop=args.enc_token_drop,
        enc_token_drop_max=args.enc_token_drop_max,
        # to pixel
        to_pixel=args.to_pixel,
    )

vq_model = vq_model.to(device)
checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
vq_model.load_state_dict(load_model_state_dict(checkpoint["model"]))
vq_model.train()

# Setup data:
transform = transforms.Compose([
    transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

dataset = ImageFolder(args.data_path, transform=transform)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")


ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

# Variables for monitoring/logging purposes:
log_steps = 0
running_loss = 0
start_time = time.time()

save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)


# ---------------------------
# Evaluation Loop
# ---------------------------
total_info = 0
total_count = 0

with torch.no_grad():
    for idx, (x, y) in enumerate(loader):
        imgs = x.to(device, non_blocking=True)
        recons, mask_loss, info = vq_model(imgs, epoch=3000)

        # 누적 info 저장
        total_info += int(info)
        total_count += 1

        # Visualization grid (original + reconstruction)
        vis = torch.cat([imgs[:8], recons[:8]], dim=0)
        vis = torch.clamp(vis, -1, 1)
        vis = make_grid((vis + 1) / 2, nrow=8, padding=2)
        vis = vis.permute(1, 2, 0).cpu().numpy()
        vis = (vis * 255).astype(np.uint8)

        Image.fromarray(vis).save(os.path.join(args.save_dir, f"batch_{idx:04d}.png"))

print("Done saving inference results.")

# 평균 계산 및 저장
mean_info = total_info / total_count if total_count > 0 else 0

with open(os.path.join("./inference_results", args.info+".txt"), "w") as f:
    f.write(f"{mean_info:.6f}\n")

print("Done saving inference results and mean_info.")




















