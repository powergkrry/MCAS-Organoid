# mask-aware pooling for ViT-L/14@336
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from transformers import CLIPVisionModelWithProjection
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorsys
import xarray as xr
from sklearn.preprocessing import normalize
from umap import UMAP
import pandas as pd

# ---- model (ViT-L/14@336) ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "openai/clip-vit-large-patch14-336"
vision = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID).to(DEVICE).eval()

# ---- CLIP normalization (same as your code, but applied with TF) ----
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMG_SIZE = 336
PATCH = 14  # ViT-L/14 -> 24 x 24 tokens


@torch.no_grad()
def preprocess_images_and_masks_batch(
    np_images: np.ndarray,
    np_masks: np.ndarray,
    batch_size: int = 32,
    device: str = DEVICE,
    hard_mask: bool = False,
):
    """
    Yields:
      pixel_values: (B, 3, 336, 336) float tensor normalized to CLIP stats
      token_weights: (B, 576) float tensor in [0,1] (per-patch weights)
    """
    assert np_images.ndim == 4 and np_images.shape[-1] in (1, 3), "images: (N,H,W,C)"
    assert np_masks.ndim in (3, 4), "masks: (N,H,W[,1])"
    N = np_images.shape[0]

    for s in range(0, N, batch_size):
        imgs = np_images[s : s + batch_size]  # (b,H,W,C)
        msks = np_masks[s : s + batch_size]  # (b,H,W[,1])

        # ensure RGB
        if imgs.shape[-1] == 1:
            imgs = np.repeat(imgs, 3, axis=-1)  # grayscale -> RGB

        b = imgs.shape[0]
        px_list, w_list = [], []

        for i in range(b):
            # ---- to torch ----
            img = (
                torch.from_numpy(imgs[i]).permute(2, 0, 1).float() / 255.0
            )  # (3,H,W) in [0,1]
            msk = msks[i]
            if msk.ndim == 3:
                msk = msk[..., 0]
            msk = torch.from_numpy((msk > 0).astype(np.float32))[
                None, ...
            ]  # (1,H,W) binary

            # ---- resize & center-crop (same geometry for both) ----
            # Note: TF.resize expects (C,H,W); specify interpolation per tensor type
            img_r = TF.resize(
                img,
                IMG_SIZE,
                interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True,
            )
            msk_r = TF.resize(msk, IMG_SIZE, interpolation=TF.InterpolationMode.NEAREST)

            img_c = TF.center_crop(img_r, IMG_SIZE)
            msk_c = TF.center_crop(msk_r, IMG_SIZE)  # (1,336,336), values {0,1}

            # ---- normalize image to CLIP stats ----
            img_c = TF.normalize(img_c, CLIP_MEAN, CLIP_STD)

            # ---- per-token weights from mask (avg_pool over 14x14 patches) ----
            weights = F.avg_pool2d(
                msk_c[None, ...], kernel_size=PATCH, stride=PATCH
            )  # (1,1,24,24)
            if hard_mask:
                weights = (weights > 0).float()
            w_flat = weights.view(-1)  # (576,)

            px_list.append(img_c)
            w_list.append(w_flat)

        pixel_values = torch.stack(px_list, dim=0).to(device)  # (B,3,336,336)
        token_weights = torch.stack(w_list, dim=0).to(device)  # (B,576)
        yield pixel_values, token_weights


@torch.no_grad()
def encode_mask_aware(
    np_images: np.ndarray,
    np_masks: np.ndarray,
    batch_size: int = 32,
    hard_mask: bool = False,
    min_tokens: int = 1,
) -> np.ndarray:
    """
    Returns: (N, D) L2-normalized embeddings in CLIP space, mask-aware pooled.
    """
    embs = []
    for pixel_values, token_weights in preprocess_images_and_masks_batch(
        np_images, np_masks, batch_size=batch_size, hard_mask=hard_mask
    ):
        # forward
        out = vision(pixel_values=pixel_values)  # last_hidden_state (B, 577, Dm)
        tokens = out.last_hidden_state[:, 1:, :]  # (B, 576, Dm), drop CLS
        cls_emb_proj = F.normalize(out.image_embeds, dim=-1)  # (B, D) projected CLS

        # normalize weights per example; detect empty selections
        w = token_weights  # (B,576)
        sel_counts = (w > 0).sum(dim=1)  # (B,)
        denom = w.sum(dim=1, keepdim=True).clamp_min(1e-6)  # (B,1)
        w_norm = (w / denom).unsqueeze(1)  # (B,1,576)

        # weighted mean in token space -> project -> normalize
        feat = torch.bmm(w_norm, tokens).squeeze(1)  # (B, Dm)
        proj = vision.visual_projection(feat)  # (B, D)
        proj = F.normalize(proj, dim=-1)

        # fallback to CLS where mask selects too few tokens
        use_cls = sel_counts < min_tokens
        if use_cls.any():
            proj[use_cls] = cls_emb_proj[use_cls]

        embs.append(proj.detach().cpu())

    return torch.cat(embs, dim=0).numpy()


#
root_path = "./"
data = xr.load_dataset(
    os.path.join(root_path, "mcam_max_image_dataset.nc"),
    engine="netcdf4",
    mask_and_scale=False,
)
mask = xr.load_dataset(
    os.path.join(root_path, "mcam_mask_dataset.nc"),
    engine="netcdf4",
    mask_and_scale=False,
)

#
chunk_size = 4
all_features = []
all_labels = []
label_index = 0
for i in range(0, data.images.shape[1], chunk_size):
    image = data.images[:, i : i + chunk_size, ...].values.reshape(-1, 3072, 3072, 1)
    image = image[..., 0, np.newaxis]
    image_mask = mask.images[:, i : i + chunk_size, ...].values.reshape(
        -1, 3072, 3072, 1
    )
    image_mask = image_mask[..., 0] > 0

    with torch.inference_mode():
        feats = encode_mask_aware(image, image_mask, batch_size=32, hard_mask=False)
        all_features.append(feats)

    all_labels += [label_index] * (image.shape[0])
    label_index += 1
all_features = np.concatenate(all_features)
all_dates = np.tile(np.repeat(list(range(data.images.shape[0])), 32), 3)

# 2D per day UMAP
chunk_size = 32
for day in range(17):
    if day != 8:
        continue

    all_feature = []
    all_label = []
    all_date = []

    for i in range(0 + day * chunk_size, all_features.shape[0], chunk_size * 17):
        all_feature.append(all_features[i : i + chunk_size])
        all_label.append(all_labels[i : i + chunk_size])
        all_date.append(all_dates[i : i + chunk_size])
    all_feature = np.concatenate(all_feature)
    all_label = np.concatenate(all_label)
    all_date = np.concatenate(all_date)

    mapping = {0: "0 Cond. 1", 1: "1 Cond. 2", 2: "2 Cond. 3"}

    # Vectorized mapping
    all_label = np.vectorize(mapping.get)(all_label)

    # Normalize (CLIP usually compared via cosine)
    Xn = normalize(all_feature)  # L2-normalize rows

    umap2d_cos = UMAP(
        n_neighbors=8, min_dist=0.01, n_components=2, metric="cosine", random_state=0
    )
    Z = umap2d_cos.fit_transform(Xn)

    # Plot
    y = np.asarray(all_label)
    dates = np.asarray(all_date, dtype=float)
    classes, y_idx = np.unique(y, return_inverse=True)

    colorblind_friendly = ["#E69F00", "#0072B2", "#009E73", "#CC79A7", "#F0E442"]
    # map each class to one of the three colors (cycles if >3)
    base_colors = {
        cls: mpl.colors.to_rgba(colorblind_friendly[i % 5])
        for i, cls in enumerate(classes)
    }

    # --- config ---
    PER_CLASS_NORM = False
    L_MIN, L_MAX = 0.35, 0.85  # lightness band for gradient; tweak to taste

    def vary_lightness_rgba(rgba, t):
        """Keep hue/sat of base color, vary lightness by t in [0,1]."""
        r, g, b, a = rgba
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l_new = L_MIN + t * (L_MAX - L_MIN)
        r2, g2, b2 = colorsys.hls_to_rgb(h, l_new, s)
        return (r2, g2, b2, a)

    def class_lightness_cmap(rgba_base):
        ts = np.linspace(0, 1, 256)
        cols = [vary_lightness_rgba(rgba_base, t) for t in ts]
        return mpl.colors.ListedColormap(cols)

    # --- figure with a narrow right column for per-class colorbars ---
    fig = plt.figure(figsize=(8, 7))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[18, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])

    # scatter with per-point colors (lightness encodes date)
    global_min, global_max = dates.min(), dates.max()
    global_ptp = max(global_max - global_min, 1.0)

    for i, cls in enumerate(classes):
        idx = y_idx == i
        base_color = base_colors[cls]

        if PER_CLASS_NORM:
            d = dates[idx]
            if d.size == 0:
                continue
            dmin, dmax = d.min(), d.max()
            ptp = max(dmax - dmin, 1.0)
            t_vals = (d - dmin) / ptp
        else:
            t_vals = (dates[idx] - global_min) / global_ptp

        colors = [vary_lightness_rgba(base_color, float(t)) for t in t_vals]
        ax.scatter(Z[idx, 0], Z[idx, 1], s=20, alpha=0.85, label=str(cls), color=colors)

    ax.set_title("UMAP by class")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(markerscale=2, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    plt.show()
