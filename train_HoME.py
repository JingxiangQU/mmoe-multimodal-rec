import os
import argparse
import math
import torch
import glob
import time
import torch.distributed as dist
import webdataset as wds
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, ViTModel, ViTConfig
import glob
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from model_HoME import (
    preprocess_batch,
    decode_sample,
    build_text_user_expert,
    build_text_item_expert,
    build_img_expert,
    build_cross_expert,
    build_concat_ui_expert,
    build_concat_ti_expert,
    HOME_MMoE_Complete,
)
from torch.nn import BCEWithLogitsLoss
import itertools
from torch.utils.data import DataLoader
import nltk
def get_state_dict(mod):
    # 如果是 DDP，取 .module.state_dict()，否则直接 .state_dict()
    return mod.module.state_dict() if hasattr(mod, "module") else mod.state_dict()

            # gather 多卡 preds/labels

def calculate_contrastive_loss(anchor, positive, temperature=0.07):
    # 统一进行L2归一化
    anchor_norm = F.normalize(anchor, p=2, dim=1)
    positive_norm = F.normalize(positive, p=2, dim=1)
    # 计算相似度矩阵
    sim_matrix = torch.matmul(anchor_norm, positive_norm.t()) / temperature
    # 对角线是正样本，其余是负样本
    labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
    return F.cross_entropy(sim_matrix, labels)
    
def split_by_worker_fn(sample):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return True  # 单进程模式，不切分
    worker_id = worker_info.id
    num_workers = worker_info.num_workers
    # __key__ 是 WebDataset 每条样本的唯一 key
    key = sample["__key__"]
    return hash(key) % num_workers == worker_id
    
def make_loader(file_list: list, batch_size: int, num_workers: int): # 参数名file_list
    """单/多卡通用 WebDataset 读取器。"""
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank  = dist.get_rank()       if dist.is_initialized() else 0
    dataset = (
        wds.WebDataset(
            file_list,
            shardshuffle=True,
            nodesplitter=wds.split_by_node
        )
        .select(split_by_worker_fn)
        .shuffle(512)
        .map(decode_sample)
        .select(lambda x: x is not None)
        .repeat()
    )
    loader = DataLoader(
        dataset.batched(batch_size, collation_fn=lambda b: b),
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )
    if rank == 0:
        
        print(f"[Loader] files_found={len(file_list)}  batch={batch_size}  "
              f"num_workers={num_workers}  world={world}")
    return loader

    # ---------------- DataLoader ----------------
def all_gather_tensor(t: torch.Tensor):
    """跨进程 gather 张量并 concat（支持单卡）"""
    if dist.is_initialized() and dist.get_world_size() > 1:
        out = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(out, t)
        t = torch.cat(out, dim=0)
    return t

class HomeExpertWrapper(nn.Module):
    """负责BN和silu"""
    def __init__(self, d_model: int, dropout_p: float = 0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = F.silu
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # 兼容 (B, D) 和 (B, L, D) 两种输入
        if x.dim() == 3: # (B, L, D)
            b, l, d = x.shape
            x_reshaped = x.reshape(b * l, d)
            processed = self.dropout(self.activation(self.norm(x_reshaped))).reshape(b, l, d)
        else: # (B, D)
            processed = self.dropout(self.activation(self.norm(x)))
        return processed

def main():
    # 0. DDP init
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # 1. args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--img_model", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--data_pattern", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./outputs", help="保存模型、图表等的目录")
    parser.add_argument("--lambda_cross", type=float, default=0.1, help="Weight for cross-attention contrastive loss.")
    parser.add_argument("--lambda_user_img", type=float, default=0.1, help="Weight for user-image contrastive loss.")
    parser.add_argument("--lambda_item_img", type=float, default=0.1, help="Weight for item-image contrastive loss.")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for all contrastive losses.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # 2. tokenizer & nlp
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(["<SENT>"])
    data_pattern_str = args.data_pattern
    file_list = glob.glob(data_pattern_str)

    # 3. 检查是否找到了文件，如果没有就报错退出
    if not file_list:
        raise FileNotFoundError(f"未找到任何匹配此模式的文件: {data_pattern_str}")
    # 3. data pipeline
    loader = make_loader(
        file_list,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 4. build experts
    user_expert   = build_text_user_expert(
        args.model_name, args.lora_r, 384, tokenizer, device
    )
    item_expert   = build_text_item_expert(
        args.model_name, args.lora_r, 384, tokenizer, device
    )
    img_expert    = build_img_expert(
        args.img_model, device=device
    )
    cross_ui      = build_cross_expert(device=device)
    concat_ui     = build_concat_ui_expert(device=device)
    concat_ti     = build_concat_ti_expert(device=device)
    head_module = HOME_MMoE_Complete(
    expert_dim=768,             # 专家的输入/输出维度，与的特征向量维度一致
    n_shared_experts=4,         # 共享专家的数量
    n_task_experts=2,           # 每个任务的专属专家数量
    tower_hidden=512            # 任务塔中间层的维度
    ).to(device)
    u_doc_wrapper = HomeExpertWrapper(d_model=768).to(device)
    i_doc_wrapper = HomeExpertWrapper(d_model=768).to(device)
    img_vec_wrapper = HomeExpertWrapper(d_model=768).to(device)
    ui_vec_wrapper = HomeExpertWrapper(d_model=768).to(device)
    xui_wrapper = HomeExpertWrapper(d_model=768).to(device)
    xti_wrapper = HomeExpertWrapper(d_model=768).to(device)  
    
    # wrap DDP
    user_expert = DDP(user_expert, device_ids=[rank], find_unused_parameters=True)
    item_expert = DDP(item_expert, device_ids=[rank], find_unused_parameters=True)
    img_expert  = DDP(img_expert,  device_ids=[rank], find_unused_parameters=True) #冻结的专家不加入ddp
    cross_ui    = DDP(cross_ui,    device_ids=[rank], find_unused_parameters=True)
    concat_ui   = DDP(concat_ui,   device_ids=[rank], find_unused_parameters=True)
    concat_ti   = DDP(concat_ti,   device_ids=[rank], find_unused_parameters=True)
    head        = DDP(head_module, device_ids=[rank], find_unused_parameters=True)
    u_doc_wrapper = DDP(u_doc_wrapper, device_ids=[rank], find_unused_parameters=True)
    i_doc_wrapper = DDP(i_doc_wrapper, device_ids=[rank], find_unused_parameters=True)
    img_vec_wrapper = DDP(img_vec_wrapper, device_ids=[rank], find_unused_parameters=True)
    ui_vec_wrapper = DDP(ui_vec_wrapper, device_ids=[rank], find_unused_parameters=True)
    xui_wrapper = DDP(xui_wrapper, device_ids=[rank], find_unused_parameters=True)
    xti_wrapper = DDP(xti_wrapper, device_ids=[rank], find_unused_parameters=True)
    # 5. freeze schedule
    STEPS_PER_EPOCH = 7_200        # 每张卡一个 epoch 跑多少个更新 step
    total_steps   = args.epochs * STEPS_PER_EPOCH
    freeze_steps  = 2_400
    unfreeze_steps  = 1_600
    unfreeze_start  = freeze_steps
    unfreeze_end    = freeze_steps + unfreeze_steps
    ACCUM           = args.grad_accum
   
    lora_params = []
    for expert in (user_expert, item_expert):
        for name, p in expert.module.named_parameters():
            if "lora_" in name:
                lora_params.append(p)

    # —— 收集要全程更新的其他参数（gate/tower/head） —— 
    other_modules = [
        cross_ui, concat_ui, concat_ti, head,
        u_doc_wrapper, i_doc_wrapper, img_vec_wrapper,
        ui_vec_wrapper, xui_wrapper, xti_wrapper
    ]

    other_params = []
    for mod in other_modules:
        # DDP包装后用 .module 访问真实模型
        other_params.extend(p for p in mod.module.parameters() if p.requires_grad)
    # ——  收集图像专家的可训练参数    —— 
    num_layers_to_unfreeze = 2 
    for param in img_expert.module.vit_model.parameters(): # DDP包装后用 .module
        param.requires_grad = False
    layer_names = [name for name, _ in img_expert.module.vit_model.named_parameters() if 'encoder.layer' in name]
    if layer_names:
        layer_indices = sorted(list(set([int(name.split('.')[2]) for name in layer_names])))
        unfreeze_start_index = layer_indices[-num_layers_to_unfreeze]
        for name, param in img_expert.module.vit_model.named_parameters():
            if 'encoder.layer' in name:
                layer_index = int(name.split('.')[2])
                if layer_index >= unfreeze_start_index:
                    param.requires_grad = True
                    if rank == 0:
                        print(f"Optimizer setup: Unfreezing ViT layer: {name}")

    # 从img_expert中提取所有可训练的参数
    image_expert_trainable_params = [
        p for p in img_expert.parameters() if p.requires_grad
    ]
    optimizer = AdamW([
        { "params": other_params },                      # 组0: 其他模块
        { "params": lora_params },                       # 组1: LoRA参数
        { "params": image_expert_trainable_params }      # 组2: 图像专家可训练参数
    ], lr=args.lr, weight_decay=args.weight_decay) # 设置一个基础lr

    # 7. 自定义 Scheduler：Group0 恒为 1；Group1 冻结→warm-up→1
# 调度函数1: 用于 other_params (恒定学习率)
    def lr_other(step):
        return 1.0  # 始终使用基础学习率

    # 调度函数2: 用于 lora_params 和 image_expert_trainable_params (冻结 -> 预热)
    def lr_finetune(step):
        """
        这个函数现在同时用于LoRA和ViT的微调。
        """
        if step <= unfreeze_start:
            return 0.0  # 冻结阶段，学习率为0
        elif step <= unfreeze_end:
            # 预热阶段，学习率从0线性增长到1
            return (step - unfreeze_start) / float(max(1, unfreeze_steps))
        else:
            # 预热结束，使用基础学习率
            return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda=[
        lr_other,       # 应用于第0组参数 (other_params)
        lr_finetune,    # 应用于第1组参数 (lora_params)
        lr_finetune     # 应用于第2组参数 (image_expert_trainable_params)
    ])

    scaler = GradScaler()

    # 7. loss functions with pos_weight
    pw_g = torch.tensor(858627/990303, dtype=torch.float32,device=device)
    loss_fn_good = BCEWithLogitsLoss(pos_weight=pw_g)
    pw_b = torch.tensor(1328721/520209,dtype=torch.float32, device=device)
    loss_fn_best = BCEWithLogitsLoss(pos_weight=pw_b)
    #
    def has_nan(x):
        if isinstance(x, torch.Tensor):
            return torch.isnan(x).any().item() or torch.isinf(x).any().item()
        elif isinstance(x, np.ndarray):
            return np.isnan(x).any() or np.isinf(x).any()
        else:
            return False
   
    # —— 为绘图准备 —— 
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

        # Separate lists for each loss type
        main_losses_per_step = []
        contrastive_losses_per_step = []
        total_losses_per_step = []
        global_steps_for_plot = []
        epoch_avg_main_losses = []
        epoch_avg_contrastive_losses = []
    global_step = 0
    for epoch in range(args.epochs):
        running_main_loss, running_contrastive_loss = 0.0, 0.0
        tot_samples = 0
        #all_preds_g_local, all_labels_g_local = [], [] #  本地的NumPy列表
        #all_preds_b_local, all_labels_b_local = [], [] 

        data_iter = iter(loader)     # 无限流，手动截断
        for step in range(STEPS_PER_EPOCH):
            batch = next(data_iter)      # 若数据集不足，会自动 wrap-around
            global_step += 1
            
            # —— 1. 决定 LoRA 专家是否可 train —— 
            #finetune_active = (global_step > freeze_steps)
            # batch data → device
            texts_u = [b['user_text'] for b in batch]
            texts_i = [b['item_text'] for b in batch]
            patches  = torch.stack([b['patch'] for b in batch]).to(device)
            y_good   = torch.tensor([b['label_good'] for b in batch],
                                     dtype=torch.float32, device=device)
            y_best   = torch.tensor([b['label_best'] for b in batch],
                                     dtype=torch.float32, device=device) 
            skip = torch.tensor(int(has_nan(patches) or has_nan(y_good) or has_nan(y_best)),
                                device=device)
            if dist.is_initialized():
                dist.all_reduce(skip, op=dist.ReduceOp.MAX)
            if skip.item() > 0:
                if rank == 0:
                    print(f"[NaN/Inf样本] batch={step}, 全卡跳过")
                continue
            # preprocess on CPU
            in_u, c2s_u, pos_u, max_s_u = preprocess_batch(
                texts_u, tokenizer, max_tok=384)
            in_i, c2s_i, pos_i, max_s_i = preprocess_batch(
                texts_i, tokenizer, max_tok=384)

            with autocast():
                u_sent, u_mask, u_doc = user_expert(in_u, c2s_u, pos_u, max_s_u )
                i_sent, i_mask, i_doc = item_expert(in_i, c2s_i, pos_i, max_s_i)
                img_vec, projected_img_vec = img_expert(patches)
                ui_vec  = cross_ui(u_sent, u_mask, i_sent, i_mask)
                xui     = concat_ui(u_doc, img_vec)
                xti     = concat_ti(i_doc, img_vec)
                u_doc_normed = u_doc_wrapper(u_doc)
                i_doc_normed = i_doc_wrapper(i_doc)
                img_vec_normed = img_vec_wrapper(img_vec)
                ui_vec_normed = ui_vec_wrapper(ui_vec)
                xui_normed = xui_wrapper(xui)
                xti_normed = xti_wrapper(xti)
                expert_vecs = torch.stack([u_doc_normed, i_doc_normed, img_vec_normed, ui_vec_normed, xui_normed, xti_normed], dim=1)
                logit_g, logit_b = head(expert_vecs)
                main_loss = loss_fn_good(logit_g, y_good) + loss_fn_best(logit_b, y_best)

                # --- 始终计算对比学习Loss ---
                # Use RAW outputs for contrastive learning
                loss_cl_cross = calculate_contrastive_loss(ui_vec, i_doc)
                loss_cl_user_img = calculate_contrastive_loss(u_doc, projected_img_vec)
                loss_cl_item_img = calculate_contrastive_loss(i_doc, projected_img_vec)
                
                contrastive_loss = (args.lambda_cross * loss_cl_cross + 
                                    args.lambda_user_img * loss_cl_user_img + 
                                    args.lambda_item_img * loss_cl_item_img)

                # =================================================================
                # 使用“静态图”技巧合并Loss
                # 保证了计算图的结构始终不变
                total_loss = main_loss + contrastive_loss 
                # =================================================================
                
                # 为梯度累积进行缩放
                loss_to_backward = total_loss / ACCUM
                
                
            running_main_loss += main_loss.item() * len(batch)
            
            running_contrastive_loss += contrastive_loss.item() * len(batch)
            tot_samples += len(batch)
            
            # —— backward & DDP 通信（no_sync 控制） —— 
            if (step + 1) % ACCUM != 0:
                # 非更新步：对所有 DDP 模块跳过同步
                with user_expert.no_sync(), \
                    item_expert.no_sync(), \
                    img_expert.no_sync(), \
                    cross_ui.no_sync(), \
                    concat_ui.no_sync(), \
                    concat_ti.no_sync(), \
                    head.no_sync(), \
                    u_doc_wrapper.no_sync(), \
                    i_doc_wrapper.no_sync(), \
                    img_vec_wrapper.no_sync(), \
                    ui_vec_wrapper.no_sync(), \
                    xui_wrapper.no_sync(), \
                    xti_wrapper.no_sync():
                    scaler.scale(loss_to_backward).backward()
            else:
                # 更新步：正常同步
                scaler.scale(loss_to_backward).backward()
                scaler.unscale_(optimizer)
                all_trainable_params = other_params + lora_params + image_expert_trainable_params
                total_norm = torch.nn.utils.clip_grad_norm_(
                    all_trainable_params,
                    args.max_norm
                )              
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            if rank == 0 and (step + 1) % ACCUM == 0:
                
                main_losses_per_step.append(main_loss.item())
                contrastive_losses_per_step.append(contrastive_loss.item() )
                total_losses_per_step.append(total_loss.item())
                global_steps_for_plot.append(global_step)
                print(f"E{epoch} S{global_step} | Total Loss: {total_loss.item():.4f} | Main: {main_loss.item():.4f} | CL: {contrastive_loss.item():.4f}", flush=True)
                print(f"Epoch{epoch} Step{global_step} total_norm={total_norm:.4f}", flush=True)
        # ——计算 & 保存 (只在 rank0) ——
        if rank == 0:
            epoch_avg_main = running_main_loss / tot_samples
            epoch_avg_contrastive = running_contrastive_loss / tot_samples if running_contrastive_loss > 0 else 0
            epoch_avg_main_losses.append(epoch_avg_main)
            epoch_avg_contrastive_losses.append(epoch_avg_contrastive)
            print(f"End of Epoch {epoch} | Avg Main Loss: {epoch_avg_main:.4f} | Avg CL: {epoch_avg_contrastive:.4f}")
  
            ckpt = {
                'epoch': epoch,
                'user': get_state_dict(user_expert),
                'item': get_state_dict(item_expert),
                'img': get_state_dict(img_expert),
                'cross_ui': get_state_dict(cross_ui),
                'concat_ui': get_state_dict(concat_ui),
                'concat_ti': get_state_dict(concat_ti),
                'head': get_state_dict(head_module),
                'u_doc_wrapper': get_state_dict(u_doc_wrapper),
                'i_doc_wrapper': get_state_dict(i_doc_wrapper),
                'img_vec_wrapper': get_state_dict(img_vec_wrapper),
                'ui_vec_wrapper': get_state_dict(ui_vec_wrapper),
                'xui_wrapper': get_state_dict(xui_wrapper),
                'xti_wrapper': get_state_dict(xti_wrapper),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            model_version = "HoME_CL"
            torch.save(ckpt, os.path.join(args.output_dir, f"ckpt_{model_version}_epoch{epoch}.pt"))


    if rank == 0:
        output_dir_path = args.output_dir
        model_version = "HoME_CL"

        # --- 图 1: 每个更新步的Loss变化曲线 (一张图，三个子图) ---
        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
        fig1.suptitle(f'{model_version} Training Losses (Per Update Step)', fontsize=16)

        # 子图1: Total Loss
        ax1.plot(global_steps_for_plot, total_losses_per_step, label='Total Loss', color='blue')
        ax1.set_title("Total Loss")
        ax1.set_xlabel("Global Step")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # 子图2: Main Loss (BCE)
        ax2.plot(global_steps_for_plot, main_losses_per_step, label='Main Loss (BCE)', color='green')
        ax2.set_title("Main Task (BCE) Loss")
        ax2.set_xlabel("Global Step")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

        # 子图3: Contrastive Loss
        ax3.plot(global_steps_for_plot, contrastive_losses_per_step, label='Contrastive Loss', color='orange')
        ax3.set_title("Contrastive Loss")
        ax3.set_xlabel("Global Step")
        ax3.set_ylabel("Loss")
        ax3.legend()
        ax3.grid(True)

        fig1.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(os.path.join(output_dir_path, f"{model_version}_loss_curves_per_step.png"))
        print(f"DEBUG: Per-step loss curves saved successfully.")
        plt.close(fig1)

        # --- 图 2: 每个Epoch的平均Loss变化曲线  ---
        if epoch_avg_main_losses: # 确保列表不为空
            fig2, (ax4, ax5, ax6) = plt.subplots(3, 1, figsize=(12, 18))
            fig2.suptitle(f'{model_version} Average Training Losses (Per Epoch)', fontsize=16)
            epochs_ran = list(range(len(epoch_avg_main_losses)))

            # 计算每个epoch的平均总损失
            epoch_avg_total_losses = [m + c for m, c in zip(epoch_avg_main_losses, epoch_avg_contrastive_losses)]

            # 子图4: Average Total Loss
            ax4.plot(epochs_ran, epoch_avg_total_losses, marker='o', label='Avg Total Loss', color='blue')
            ax4.set_title("Average Total Loss")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Average Loss")
            ax4.set_xticks(epochs_ran)
            ax4.legend()
            ax4.grid(True)

            # 子图5: Average Main Loss
            ax5.plot(epochs_ran, epoch_avg_main_losses, marker='o', label='Avg Main Loss', color='green')
            ax5.set_title("Average Main Task (BCE) Loss")
            ax5.set_xlabel("Epoch")
            ax5.set_ylabel("Average Loss")
            ax5.set_xticks(epochs_ran)
            ax5.legend()
            ax5.grid(True)

            # 子图6: Average Contrastive Loss
            ax6.plot(epochs_ran, epoch_avg_contrastive_losses, marker='o', label='Avg Contrastive Loss', color='orange')
            ax6.set_title("Average Contrastive Loss")
            ax6.set_xlabel("Epoch")
            ax6.set_ylabel("Average Loss")
            ax6.set_xticks(epochs_ran)
            ax6.legend()
            ax6.grid(True)

            fig2.tight_layout(rect=[0, 0.03, 1, 0.96])
            plt.savefig(os.path.join(output_dir_path, f"{model_version}_loss_curves_per_epoch.png"))
            print(f"DEBUG: Per-epoch average loss curves saved successfully.")
            plt.close(fig2)

        plt.close('all') # 关闭所有可能存在的图形窗口

    dist.destroy_process_group()
if __name__ == "__main__":
    main()
