import os
import argparse
import math
import torch
import time
import torch.distributed as dist
import webdataset as wds
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, ViTModel, ViTConfig
import glob
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg') 
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
from torch.nn import BCEWithLogitsLoss
import itertools
from torch.utils.data import DataLoader
import nltk
def get_state_dict(mod):
    # 如果是 DDP，取 .module.state_dict()，否则直接 .state_dict()
    return mod.module.state_dict() if hasattr(mod, "module") else mod.state_dict()


                
def make_loader(file_list: list, batch_size: int, num_workers: int): # 参数名file_list
    """单/多卡通用 WebDataset 读取器。"""
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank  = dist.get_rank()       if dist.is_initialized() else 0

    dataset = (
        wds.WebDataset(
            file_list,  # 使用传入的文件列表
            shardshuffle=True,
            nodesplitter=wds.split_by_node
        )
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
    """跨进程 gather 张量并 concat """
    if dist.is_initialized() and dist.get_world_size() > 1:
        out = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(out, t)
        t = torch.cat(out, dim=0)
    return t
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
        args.img_model, pool_type="mean", device=device
    )
    cross_ui      = build_cross_expert(device=device)
    concat_ui     = build_concat_ui_expert(device=device)
    concat_ti     = build_concat_ti_expert(device=device)
    head_module   = TwoTaskMMoE().to(device)

    # wrap DDP
    user_expert = DDP(user_expert, device_ids=[rank], find_unused_parameters=True)
    item_expert = DDP(item_expert, device_ids=[rank], find_unused_parameters=True)
    #img_expert  = DDP(img_expert,    device_ids=[rank]) #冻结的专家不加入ddp，也可以加入并冻结参数但find_unused_parameters=True
    cross_ui    = DDP(cross_ui,      device_ids=[rank])
    concat_ui   = DDP(concat_ui,     device_ids=[rank])
    concat_ti   = DDP(concat_ti,     device_ids=[rank])
    head        = DDP(head_module,   device_ids=[rank])
   
    
    # 5. freeze schedule
    STEPS_PER_EPOCH = 5_600         # 每张卡一个 epoch 跑多少个更新 step
    total_steps   = args.epochs * STEPS_PER_EPOCH
    freeze_steps  = 2_000
    unfreeze_steps  = 1_200
    unfreeze_start  = freeze_steps
    unfreeze_end    = freeze_steps + unfreeze_steps
    ACCUM           = args.grad_accum
    # 6. optimizer + scheduler + scaler
    lora_params = []
    for expert in (user_expert, item_expert):
        # DDP 包装后，真实模型在 .module
        for name, p in expert.module.named_parameters():
            if "lora_" in name:
                lora_params.append(p)

    # —— 收集要全程更新的其他参数（gate/tower/head） —— 
    other_params = []
    for mod in (cross_ui, concat_ui, concat_ti, head):
        for p in mod.module.parameters():
            other_params.append(p)

    optimizer = AdamW([
        { "params": other_params, "lr": args.lr           },  # gate/tower/head 恒定 lr
        { "params": lora_params,  "lr": args.lr           },  # LoRA 的 lr 由下面 scheduler 控制
    ], weight_decay=args.weight_decay)

    # 7. 自定义 Scheduler：Group0 恒为 1；Group1 冻结→warm-up→1
    def lr_other(step): 
        return 1.0

    def lr_lora(step):
    # 0 ≤ step ≤ freeze_steps        -> lr=0
    # freeze_steps < step ≤ unfreeze_end -> lr 从 0 线性升到 1
    # step > unfreeze_end              -> lr=1
        if step <= unfreeze_start:
            return 0.0
        elif step <= unfreeze_end:
            return (step - unfreeze_start) / float(max(1, unfreeze_steps))
        else:
            return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda=[lr_other, lr_lora])

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
        epoch_losses = [] 
        # 存储每次梯度更新的 loss 值
        global_step_losses = [] 
        global_steps_for_plot = [] # 存储对应的 global_step
        #epoch_auc_g    = []
        #epoch_auc_b    = []
    global_step = 0
    for epoch in range(args.epochs):
        running_loss, tot_samples = 0.0, 0
        #all_preds_g_local, all_labels_g_local = [], [] #  本地的NumPy列表
        #all_preds_b_local, all_labels_b_local = [], [] 

        data_iter = iter(loader)     # 无限流，手动截断
        for step in range(STEPS_PER_EPOCH):
            batch = next(data_iter)      # 若数据集不足，会自动 wrap-around
            global_step += 1
            
            # —— 1. 决定 LoRA 专家是否可 train —— 
            lora_trainable = (global_step > freeze_steps)
            # batch data → device
            texts_u = [b['user_text'] for b in batch]
            texts_i = [b['item_text'] for b in batch]
            patches  = torch.stack([b['patch'] for b in batch]).to(device)
            y_good   = torch.tensor([b['label_good'] for b in batch],
                                     dtype=torch.float32, device=device)
            y_best   = torch.tensor([b['label_best'] for b in batch],
                                     dtype=torch.float32, device=device)
            if has_nan(patches) or has_nan(y_good) or has_nan(y_best):
                print(f"[NaN/Inf样本] batch={step}, 跳过")
                continue           
            # preprocess on CPU
            in_u, c2s_u, pos_u, max_s_u = preprocess_batch(
                texts_u, tokenizer, max_tok=384)
            in_i, c2s_i, pos_i, max_s_i = preprocess_batch(
                texts_i, tokenizer, max_tok=384)

            with autocast():
                u_sent, u_mask, u_doc = user_expert(in_u, c2s_u, pos_u, max_s_u, trainable=lora_trainable)
                i_sent, i_mask, i_doc = item_expert(in_i, c2s_i, pos_i, max_s_i, trainable=lora_trainable)
                img_vec = img_expert(patches, trainable=False)
                ui_vec  = cross_ui(u_sent, u_mask, i_sent, i_mask)
                xui     = concat_ui(u_doc, img_vec)
                xti     = concat_ti(i_doc, img_vec)
                expert_vecs = torch.stack([
                    u_doc, i_doc, img_vec,
                    ui_vec, xui, xti], dim=1)
                logit_g, logit_b = head.module(expert_vecs)

                loss_g = loss_fn_good(logit_g, y_good)
                loss_b = loss_fn_best(logit_b, y_best)
                if has_nan(logit_g) or has_nan(logit_b):
                    print(f"[NaN/Inf] logit_g/logit_b, step={step}")
                    continue
                loss   = (loss_g + loss_b) / ACCUM
                if has_nan(loss):
                    print(f"[NaN/Inf] loss, step={step}")
                    continue
                tot_samples  += len(batch)
                running_loss += loss.item() * ACCUM * len(batch)   # 还原真实 loss
            
            # —— backward & DDP 通信（no_sync 控制） —— 
            if (step + 1) % ACCUM != 0:
                # 非更新步：对所有 DDP 模块跳过同步
                with user_expert.no_sync(), \
                    item_expert.no_sync(),  \
                    cross_ui.no_sync(),     \
                    concat_ui.no_sync(),    \
                    concat_ti.no_sync(),    \
                    head.no_sync():
                    scaler.scale(loss).backward()
            else:
                # 更新步：正常同步
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                total_norm = torch.nn.utils.clip_grad_norm_(
                    other_params + lora_params,
                    args.max_norm
                )
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                if rank == 0: # 只有 rank 0 收集 loss
                    global_step_losses.append(loss.item() * ACCUM) # 记录的是真实平均loss
                    global_steps_for_plot.append(global_step)            
            # 在每个 step 结束时就将预测和标签转到 CPU 并转换为 NumPy
           # with torch.no_grad():
            #    preds_g_np = torch.sigmoid(logit_g).detach().cpu().numpy()
             #   preds_b_np = torch.sigmoid(logit_b).detach().cpu().numpy()
              #  y_good_np = y_good.detach().cpu().numpy()
               # y_best_np = y_best.detach().cpu().numpy()

                #if has_nan(preds_g_np) or has_nan(y_good_np) or has_nan(preds_b_np) or has_nan(y_best_np):
                 #   print(f"[NaN/Inf预测] step={step}，跳过AUC统计")
                  #  continue
                #all_preds_g_local.append(preds_g_np) #  append NumPy 数组
                #all_labels_g_local.append(y_good_np) # 
                #all_preds_b_local.append(preds_b_np) # 
                #all_labels_b_local.append(y_best_np) # 
            
            if rank == 0 and global_step % ACCUM == 0:
               
                print(f"Epoch{epoch} Step{global_step} Loss={loss.item() * ACCUM:.4f}", flush=True)
                print(f"Epoch{epoch} Step{global_step} total_norm={total_norm:.4f}", flush=True)
        # —— epoch 末 计算 & 保存 (只在 rank0) ——
        if rank == 0 : 
            epoch_loss = running_loss / tot_samples
            epoch_losses.append(epoch_loss)

            # 使用 all_gather_object 收集 NumPy 数组
            # 1. 在每个rank上拼接本地的NumPy数组
            #pg_local_concatenated = np.concatenate(all_preds_g_local)
            #lg_local_concatenated = np.concatenate(all_labels_g_local)
            #pb_local_concatenated = np.concatenate(all_preds_b_local)
            #lb_local_concatenated = np.concatenate(all_labels_b_local)

            # 2. 准备列表来接收所有rank的数据 (只在rank 0需要非空列表)
            #all_pg_concatenated = [None for _ in range(world)]
            #all_lg_concatenated = [None for _ in range(world)]
            #all_pb_concatenated = [None for _ in range(world)]
            #all_lb_concatenated = [None for _ in range(world)]

            # 3. 使用 all_gather_object 收集NumPy数组
            # all_gather_object要求数据是可序列化的，NumPy数组是可序列化的
            #dist.all_gather_object(all_pg_concatenated, pg_local_concatenated)
            #dist.all_gather_object(all_lg_concatenated, lg_local_concatenated)
            #dist.all_gather_object(all_pb_concatenated, pb_local_concatenated)
            #dist.all_gather_object(all_lb_concatenated, lb_local_concatenated)

            # 4. rank 0 拼接所有rank的数据并计算AUC
            # 这些列表现在只在rank 0上被填充了真实数据
           # pg = np.concatenate(all_pg_concatenated)
           # lg = np.concatenate(all_lg_concatenated)
           # pb = np.concatenate(all_pb_concatenated)
           # lb = np.concatenate(all_lb_concatenated)

            #auc_g = safe_auc(lg, pg) # safe_auc 直接接受 NumPy 数组
            #auc_b = safe_auc(lb, pb) # safe_auc 

            epoch_losses.append(epoch_loss)
            #epoch_auc_g.append(auc_g)
            #epoch_auc_b.append(auc_b)
            print(f"[Epoch {epoch}] samples = {tot_samples}")
            print(f"Epoch {epoch} | Loss {epoch_loss:.4f}")    
            ckpt = {
                'epoch': epoch,
                'user':      get_state_dict(user_expert),
                'item':      get_state_dict(item_expert),
                'img':       get_state_dict(img_expert), 
                'cross_ui':  get_state_dict(cross_ui),
                'concat_ui': get_state_dict(concat_ui),
                'concat_ti': get_state_dict(concat_ti),
                'head':      get_state_dict(head),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(ckpt, os.path.join(args.output_dir, f"ckpt_epoch{epoch}.pt"))
        
     
       # all_preds_g_local.clear()
       # all_labels_g_local.clear()
       # all_preds_b_local.clear()
       # all_labels_b_local.clear()


    if rank == 0:
        # 确保 Matplotlib 后端已设置
        print(f"DEBUG: Matplotlib backend in use: {plt.get_backend()}", flush=True)
        output_dir_path = args.output_dir
        
        # 绘图：每个更新 step 的 Loss
        plt.figure()
        plt.plot(global_steps_for_plot, global_step_losses, marker='.', linestyle='-')
        plt.title("Training Loss (Per Update Step)")
        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        
        # 保存图片
        loss_curve_filename = "global_step_loss_curve.png"
        loss_curve_full_path = os.path.join(output_dir_path, loss_curve_filename)
        
        print(f"DEBUG: Attempting to save global step loss curve to: {loss_curve_full_path}", flush=True)
        try:
            plt.savefig(loss_curve_full_path)
            print(f"DEBUG: Global step loss curve saved successfully to: {loss_curve_full_path}", flush=True)
        except Exception as e:
            print(f"ERROR: Failed to save global step loss curve to {loss_curve_full_path}: {e}", flush=True)
        plt.close('all') # 显式关闭图形，释放内存

        # 绘制每个 eval_interval epoch 的平均 loss
        if epoch_losses: # 只有当 epoch_losses 不为空时才绘制
            plt.figure(); 
            actual_epochs = [e for e in range(args.epochs) ]
            # 确保实际评估的epoch数量与epoch_losses列表长度匹配
            if len(actual_epochs) != len(epoch_losses):
                 print(f"Warning: Number of recorded epochs ({len(epoch_losses)}) does not match eval_interval derived epochs ({len(actual_epochs)}). Adjusting plotting X-axis.")
                 actual_epochs = list(range(len(epoch_losses)))

            plt.plot(actual_epochs, epoch_losses, marker='o')
            plt.title("Training Loss (Per Epoch Interval)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            
            epoch_loss_curve_filename = "epoch_interval_loss_curve.png"
            epoch_loss_curve_full_path = os.path.join(output_dir_path, epoch_loss_curve_filename)
            
            print(f"DEBUG: Attempting to save epoch interval loss curve to: {epoch_loss_curve_full_path}", flush=True)
            try:
                plt.savefig(epoch_loss_curve_full_path)
                print(f"DEBUG: Epoch interval loss curve saved successfully to: {epoch_loss_curve_full_path}", flush=True)
            except Exception as e:
                print(f"ERROR: Failed to save epoch interval loss curve to {epoch_loss_curve_full_path}: {e}", flush=True)
            plt.close('all')

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
