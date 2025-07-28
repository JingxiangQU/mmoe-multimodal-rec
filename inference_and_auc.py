import os
import argparse
import glob
import torch
import webdataset as wds
from torch.cuda.amp import autocast
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from model import (
    preprocess_batch,
    decode_sample,
    build_text_user_expert,
    build_text_item_expert,
    build_img_expert,
    build_cross_expert,
    build_concat_ui_expert,
    build_concat_ti_expert,
    TwoTaskMMoE,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

def make_eval_loader(file_list: list, batch_size: int, num_workers: int):
    """用于评估的WebDataset读取器（单卡）。"""
    dataset = (
        wds.WebDataset(
            file_list,
            shardshuffle=False,
        )
        .map(decode_sample)
        .select(lambda x: x is not None)
    )
    loader = DataLoader(
        dataset.batched(batch_size, collation_fn=lambda b: b),
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )
    print(f"[Loader] files_found={len(file_list)}  batch={batch_size}  num_workers={num_workers}")
    return loader

def plot_roc_curve(y_true, y_pred, auc_score, task_name, output_dir):
    """
    计算并绘制ROC曲线，并将其保存到文件。
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {task_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    save_path = os.path.join(output_dir, f"roc_curve_{task_name.lower().replace(' ', '_')}.png")
    plt.savefig(save_path)
    print(f"ROC curve plot for '{task_name}' saved to: {save_path}")
    plt.close()

def main():
    # 1. args
    parser = argparse.ArgumentParser(description="Inference script for MMoE model")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5", help="Pretrained text model name.")
    parser.add_argument("--img_model", type=str, default="google/vit-base-patch16-224-in21k", help="Pretrained image model name.")
    parser.add_argument("--data_pattern", type=str, required=True, help="Glob pattern for evaluation data shards.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument("--output_dir", type=str, default="./outputs_inference", help="Directory to save plots and results.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes for data loading.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(["<SENT>"])
    file_list = sorted(glob.glob(args.data_pattern))
    if not file_list:
        raise FileNotFoundError(f"No files found matching the pattern: {args.data_pattern}")

    loader = make_eval_loader(
        file_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    user_expert = build_text_user_expert(args.model_name, args.lora_r, 384, tokenizer, device)
    item_expert = build_text_item_expert(args.model_name, args.lora_r, 384, tokenizer, device)
    img_expert = build_img_expert(args.img_model, pool_type="mean", device=device)
    cross_ui = build_cross_expert(device=device)
    concat_ui = build_concat_ui_expert(device=device)
    concat_ti = build_concat_ti_expert(device=device)
    head = TwoTaskMMoE().to(device)

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    # Set weights_only=True for safer model loading
    ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    print("Checkpoint loaded successfully.")

    user_expert.load_state_dict(ckpt['user'])
    item_expert.load_state_dict(ckpt['item'])
    img_expert.load_state_dict(ckpt['img'])
    cross_ui.load_state_dict(ckpt['cross_ui'])
    concat_ui.load_state_dict(ckpt['concat_ui'])
    concat_ti.load_state_dict(ckpt['concat_ti'])
    head.load_state_dict(ckpt['head'])
    print("Model weights loaded into structures.")

    user_expert.eval()
    item_expert.eval()
    img_expert.eval()
    cross_ui.eval()
    concat_ui.eval()
    concat_ti.eval()
    head.eval()

    all_preds_g, all_labels_g = [], []
    all_preds_b, all_labels_b = [], []

 
    for batch in tqdm(loader, desc="Evaluating"):
        with torch.no_grad():
            texts_u = [b['user_text'] for b in batch]
            texts_i = [b['item_text'] for b in batch]
            patches = torch.stack([b['patch'] for b in batch]).to(device)
            y_good = torch.tensor([b['label_good'] for b in batch], dtype=torch.float32, device=device)
            y_best = torch.tensor([b['label_best'] for b in batch], dtype=torch.float32, device=device)

            with autocast():
                in_u, c2s_u, pos_u, max_s_u = preprocess_batch(texts_u, tokenizer, max_tok=384)
                in_i, c2s_i, pos_i, max_s_i = preprocess_batch(texts_i, tokenizer, max_tok=384)
                u_sent, u_mask, u_doc = user_expert(in_u, c2s_u, pos_u, max_s_u, trainable=False)
                i_sent, i_mask, i_doc = item_expert(in_i, c2s_i, pos_i, max_s_i, trainable=False)
                img_vec = img_expert(patches, trainable=False)
                ui_vec = cross_ui(u_sent, u_mask, i_sent, i_mask)
                xui = concat_ui(u_doc, img_vec)
                xti = concat_ti(i_doc, img_vec)
                expert_vecs = torch.stack([u_doc, i_doc, img_vec, ui_vec, xui, xti], dim=1)
                logit_g, logit_b = head(expert_vecs)

            preds_g = torch.sigmoid(logit_g).cpu().numpy()
            preds_b = torch.sigmoid(logit_b).cpu().numpy()
            
            all_preds_g.append(preds_g)
            all_labels_g.append(y_good.cpu().numpy())
            all_preds_b.append(preds_b)
            all_labels_b.append(y_best.cpu().numpy())

    if not all_labels_g:
        print("No samples were processed. Cannot calculate AUC or plot ROC.")
        return

    final_preds_g = np.concatenate(all_preds_g)
    final_labels_g = np.concatenate(all_labels_g)
    final_preds_b = np.concatenate(all_preds_b)
    final_labels_b = np.concatenate(all_labels_b)
    
    print("\n" + "="*30)
    print("Inference finished. Calculating results...")
    
    try:
        auc_g = roc_auc_score(final_labels_g, final_preds_g)
        print(f"AUC for 'good' task: {auc_g:.6f}")
        plot_roc_curve(final_labels_g, final_preds_g, auc_g, "Good Task", args.output_dir)
    except ValueError as e:
        print(f"Could not process 'good' task: {e}")

    try:
        auc_b = roc_auc_score(final_labels_b, final_preds_b)
        print(f"AUC for 'best' task: {auc_b:.6f}")
        plot_roc_curve(final_labels_b, final_preds_b, auc_b, "Best Task", args.output_dir)
    except ValueError as e:
        print(f"Could not process 'best' task: {e}")
    
    print("="*30)

if __name__ == "__main__":
    main()
