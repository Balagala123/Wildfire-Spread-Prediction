# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 22:46:18 2025

@author: olivi
"""

# PyTorch training pipeline using the HadamardUnet
import os, sys, json, re

sys.path += ["D:/wildfire/wildfire_detection/codetfAE1",
             "D:/wildfire/wildfire_detection/codetfAE1/models"]
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import colors
from hadamard_unet_BN_dropout import HadamardUnet
import tensorflow as tf  # Still needed if TFRecord dataset pipeline is reused
from dataset import make_dataset, ModeKeys
#from dataset_adjust_pre import make_dataset, ModeKeys
#from dataset_adjust_pre_post import make_dataset, ModeKeys
from constants import INPUT_FEATURES

from losses1 import combo_loss 
from metrics import AUCWithMaskedClass, PrecisionWithMaskedClass, RecallWithMaskedClass, masked_iou, compute_best_threshold

class HParams:
    train_path = "/archive/next_day_wildfire_spread_train_*.tfrecord"
    eval_path  = "archive/next_day_wildfire_spread_eval_*.tfrecord"
    test_path = "archive/next_day_wildfire_spread_test_*.tfrecord"
    input_features = list(INPUT_FEATURES)
    output_features = ["FireMask"]
    data_sample_size = 64
    sample_size = 64
    output_sample_size = 64
    batch_size = 32
    shuffle_buffer_size = 500
    compression_type = ""
    random_flip = True
    random_rotate = False
    random_crop = False
    input_sequence_length = 1
    output_sequence_length = 1
    binarize_output = True
    downsample_threshold = 0.3
    azimuth_in_channel = "th"
    azimuth_out_channel = None
    learning_rate = 1e-4
    epochs = 5
    steps_per_epoch = 1000
    pos_weight = 3.0
    model_dir = ""
    run_threshold_optimization = False
    
hp = HParams()
os.makedirs(hp.model_dir, exist_ok=True)

train_dataset = make_dataset(hp, mode=ModeKeys.TRAIN)
val_dataset = make_dataset(hp, mode=ModeKeys.EVAL)
test_dataset = make_dataset(hp, mode=ModeKeys.PREDICT)

def evaluate_model(model, dataset, threshold=0.5):
    auc, prec, rec = AUCWithMaskedClass(), PrecisionWithMaskedClass(), RecallWithMaskedClass()
    ious = []
    model.eval()
    with torch.no_grad():
        for i, (feats_tf, labs_tf) in enumerate(dataset):
            if i >= hp.steps_per_epoch:
                break
            x = torch.from_numpy(feats_tf.numpy()).permute(0, 3, 1, 2).float().to(device)
            y_pred = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
            y_true = labs_tf.numpy()[..., 0]
            auc.update_state(torch.tensor(y_true), torch.tensor(y_pred))
            pred_bin = (y_pred > threshold).astype(np.float32)
            prec.update_state(torch.tensor(y_true), torch.tensor(pred_bin))
            rec.update_state(torch.tensor(y_true), torch.tensor(pred_bin))
            for j in range(y_pred.shape[0]):
                ious.append(masked_iou(y_true[j], pred_bin[j]))

    return (
        np.mean(ious),
        auc.result().numpy().item(),
        prec.result().numpy().item(),
        rec.result().numpy().item(),
    )


def plot_train_val_losses(train_losses, val_losses, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(train_losses); ax[0].set_title("Train Loss")
    ax[1].plot(val_losses); ax[1].set_title("Val Loss")
    for a in ax: a.set_xlabel("Epoch"); a.set_ylabel("Loss"); a.grid(True)
    plt.tight_layout()
    if save_path: plt.savefig(save_path); print(f"📉 Saved plot to {save_path}")
    plt.show()

def show_inference(n_rows, feats_tf, labs_tf, model, thr=0.5, save_path=None):
    seg_cmap = colors.ListedColormap(["black", "silver", "orangered"])
    seg_norm = colors.BoundaryNorm([-1, -.1, .001, 1], seg_cmap.N)
    x = torch.from_numpy(feats_tf.numpy()).permute(0, 3, 1, 2).float().to(device)
    with torch.no_grad():
        preds = torch.sigmoid(model(x)).squeeze(1).cpu().numpy() > thr
    gts = labs_tf.numpy()[..., 0]
    bases = feats_tf.numpy()
    prefire_binary = (bases[:, :, :, -1] > 0.5).astype(np.int32)
    plt.figure(figsize=(15, 4 * n_rows))
    for i in range(n_rows):
        plt.subplot(n_rows, 3, 3 * i + 1); plt.imshow(prefire_binary[i], cmap=seg_cmap, norm=seg_norm); plt.title("Prev-day Fire"); plt.axis("off")
        plt.subplot(n_rows, 3, 3 * i + 2); plt.imshow(gts[i], cmap=seg_cmap, norm=seg_norm); plt.title("Ground Truth"); plt.axis("off")
        plt.subplot(n_rows, 3, 3 * i + 3); plt.imshow(preds[i], cmap=seg_cmap, norm=seg_norm); plt.title("Prediction"); plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"🖼️ Saved inference to {save_path}")
    plt.show()

def get_numbered_save_dir(base="output_model_exp"):
    existing = [d for d in os.listdir() if os.path.isdir(d) and re.match(f"{base}[0-9]+", d)]
    ids = [int(re.search(r"\d+", d).group()) for d in existing] if existing else []
    name = f"{base}{max(ids) + 1:02d}" if ids else f"{base}01"
    os.makedirs(name); return name

device = torch.device("cpu")
model = HadamardUnet(len(hp.input_features), hp.sample_size, 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
EPOCHS, best_auc = hp.epochs, -1.0
train_losses, val_losses, all_metrics = [], [], []
train_batches = train_dataset
best_metrics = None

for ep in range(1, EPOCHS + 1):
    # ─── Train ──────────────────────────────────────────────
    model.train()
    tloss, n = 0.0, 0
    train_iter = iter(train_dataset)

    pbar = tqdm(range(hp.steps_per_epoch), desc=f"Epoch {ep} [Train]", dynamic_ncols=True)
    for i in pbar:
        feats_tf, labs_tf = next(train_iter)
        x = torch.from_numpy(feats_tf.numpy()).permute(0, 3, 1, 2).float().to(device)
        y = torch.from_numpy(labs_tf.numpy()).permute(0, 3, 1, 2).float().to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = combo_loss(y_pred, y, pos_weight=hp.pos_weight)
        loss.backward()
        optimizer.step()

        tloss += loss.item()
        n += 1
        pbar.set_postfix({
            "batch": f"{i+1}/{hp.steps_per_epoch}",
            "batch_loss": f"{loss.item():.4f}",
            "avg_loss": f"{tloss / n:.4f}"
        })
    
    avg_train_loss = tloss / n
    train_losses.append(avg_train_loss)

    # ─── Eval ───────────────────────────────────────────────
    model.eval()
    vloss, m = 0.0, 0
    val_iter = iter(val_dataset)

    with torch.no_grad():
        pbar = tqdm(range(hp.steps_per_epoch), desc=f"Epoch {ep} [Val]", dynamic_ncols=False)
        for i in pbar:
            feats_tf, labs_tf = next(val_iter)
            x = torch.from_numpy(feats_tf.numpy()).permute(0, 3, 1, 2).float().to(device)
            y = torch.from_numpy(labs_tf.numpy()).permute(0, 3, 1, 2).float().to(device)
            y_pred = model(x)
            loss = combo_loss(y_pred, y, pos_weight=hp.pos_weight)
            vloss += loss.item()
            m += 1

    avg_val_loss = vloss / m
    val_losses.append(avg_val_loss)

    # ─── Metrics ─────────────────────────────────────────────
    print(f"  🔁 Learning Rate = {optimizer.param_groups[0]['lr']:.2e}")
    iou_val, auc_val, prec_val, rec_val = evaluate_model(model, val_dataset)
    print(f"Epoch {ep:02d} | Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f} | AUC={auc_val:.4f}, Prec={prec_val:.4f}, Rec={rec_val:.4f}, IoU={iou_val:.4f}")

    all_metrics.append({
        "epoch": ep,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "auc": auc_val,
        "precision": prec_val,
        "recall": rec_val,
        "iou": iou_val,
    })

    if auc_val > best_auc:
        best_auc = auc_val
        best_metrics = all_metrics[-1]
        torch.save(model.state_dict(), "best_hadamard_unet.pth")
        print(f"   🔏 Saved new best model (AUC = {best_auc:.4f})")


if hp.run_threshold_optimization:
    best_thr, best_f1, best_prec, best_rec = compute_best_threshold(model, val_dataset)
    best_metrics["best_threshold"] = best_thr
    print("\n🔍 Best Threshold Evaluation:")
    print(f"  🔢 Threshold : {best_thr:.2f}")
    print(f"  🎯 Precision: {best_prec:.4f}")
    print(f"  🔥 Recall   : {best_rec:.4f}")
    print(f"  🏆 F1-Score : {best_f1:.4f}")
else:
    best_thr = 0.5

SAVE_DIR = get_numbered_save_dir()
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_hadamard_unet.pth"))
with open(os.path.join(SAVE_DIR, "best_metrics.json"), "w") as f:
    json.dump(best_metrics, f, indent=2)
pd.DataFrame(all_metrics).to_csv(os.path.join(SAVE_DIR, "epoch_metrics.csv"), index=False)
plot_train_val_losses(train_losses, val_losses, save_path=os.path.join(SAVE_DIR, "loss_curve.png"))
feat_batch, lab_batch = next(iter(test_dataset))
show_inference(20, feat_batch, lab_batch, model, thr=best_thr, save_path=os.path.join(SAVE_DIR, "inference_example.png"))

print("\n🌝 Best Model Summary:")
for k, v in best_metrics.items():
    print(f"  {k:<15}: {v:.4f}" if isinstance(v, float) else f"  {k:<15}: {v}")

print("\n🧪 Final Evaluation on Test Set using Best Threshold")
iou_test, auc_test, prec_test, rec_test = evaluate_model(model, test_dataset, threshold=best_thr)
print(f"  ✅ AUC       : {auc_test:.4f}")
print(f"  🎯 Precision : {prec_test:.4f}")
print(f"  🔥 Recall    : {rec_test:.4f}")
print(f"  📊 IoU       : {iou_test:.4f}")
f1_test = 2 * (prec_test * rec_test) / (prec_test + rec_test + 1e-8)
print(f"  🏆 F1-Score  : {f1_test:.4f}")

