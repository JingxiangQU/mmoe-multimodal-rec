import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import webdataset as wds
from transformers import AutoModel, AutoTokenizer
from transformers import ViTModel
from peft import get_peft_model, LoraConfig, TaskType

from typing import List

import nltk
import numpy as np
# -------------------函数----------------------------

from nltk.tokenize import sent_tokenize

def nltk_sentence_split(text: str) -> list[str]:
    """
    使用NLTK的punkt进行快速且准确的分句
    """
    if not text:
        return []
    return sent_tokenize(text)


def preprocess_batch(
    texts: List[str],
    tokenizer,
    max_tok: int,
    max_chunks_per_sample: int = 4,
    fixed_sent_count: int = 64,  #  直接加默认固定值
):
    pad_limit = max_tok - 2
    sent_token_id = tokenizer.convert_tokens_to_ids("<SENT>")

    all_tokens = []
    chunk2sample = []
    all_positions = []

    max_chunk_len = 0
    max_sents_per_chunk = 0
    sample_sent_counts = []

    # 1) 批量分句 → docs
    
    for si, text in enumerate(texts):
        sents = nltk_sentence_split(text)
        

        cur, sent_pos = [], []
        chunk_count = 0
        actual_sents = 0

        for sent in sents:
            if chunk_count >= max_chunks_per_sample:
                break
            base_ids = tokenizer.encode(
                sent,
                add_special_tokens=False,
                max_length=pad_limit - 1,
                truncation=True
            )
            ids = [sent_token_id] + base_ids

            if len(cur) + len(ids) > pad_limit:
                tokens = [tokenizer.cls_token_id] + cur + [tokenizer.sep_token_id]
                all_tokens.append(tokens)
                chunk2sample.append(si)

                this_pos = [p + 1 for p in sent_pos]
                all_positions.append(this_pos)
                max_sents_per_chunk = max(max_sents_per_chunk, len(this_pos))
                max_chunk_len = max(max_chunk_len, len(tokens))
                chunk_count += 1
                actual_sents += len(sent_pos)
                cur, sent_pos = ids.copy(), [0]
            else:
                sent_pos.append(len(cur))
                cur.extend(ids)

        # 最后一个 chunk
        if chunk_count < max_chunks_per_sample and cur:
            tokens = [tokenizer.cls_token_id] + cur + [tokenizer.sep_token_id]
            all_tokens.append(tokens)
            chunk2sample.append(si)

            this_pos = [p + 1 for p in sent_pos]
            all_positions.append(this_pos)
            max_sents_per_chunk = max(max_sents_per_chunk, len(this_pos))
            max_chunk_len = max(max_chunk_len, len(tokens))
            actual_sents += len(sent_pos)

        sample_sent_counts.append(actual_sents)

    # 2) token-level padding → [N_chunks, max_chunk_len]
    pad_id = tokenizer.pad_token_id
    input_ids = [tok + [pad_id] * (max_chunk_len - len(tok)) for tok in all_tokens]
    vocab_size = tokenizer.vocab_size
    final_input_ids = []
    for token_list in input_ids:
        # 检查并修正越界的token ID，将其替换为 [PAD] token
        corrected_list = [
            token_id if token_id < vocab_size else tokenizer.pad_token_id
            for token_id in token_list
        ]
        final_input_ids.append(corrected_list)
    # 3) 句-level padding → [N_chunks, max_sents_per_chunk]
    sent_pos = [pos + [-1] * (max_sents_per_chunk - len(pos)) for pos in all_positions]

    # 4) 只改这里：固定句子数为 128，多的截断，少的 pad
    max_sent_count = fixed_sent_count
    

    return final_input_ids, chunk2sample, sent_pos, max_sent_count



def safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if not (np.isnan(v) or np.isinf(v)) else default
    except Exception:
        return default
def decode_sample(sample: dict):
    """
    安全地从原始二进制数据解码单个样本。
    能够处理文件缺失、JSON解码错误等问题。
    如果样本无效，则返回 None。
    """
    try:
        user_bytes = sample.get("user.json", b"")
        item_bytes = sample.get("item.json", b"")
        label_bytes = sample.get("label.json", b"")
        misc_bytes = sample.get("misc.json", b"")

        # 必要字段
        if not user_bytes or not item_bytes or not label_bytes:
            return None

        user_raw = user_bytes.decode('utf-8').strip()
        item_raw = item_bytes.decode('utf-8').strip()
        label = json.loads(label_bytes)
        misc = json.loads(misc_bytes) if misc_bytes else {}

        # 检查空文本/异常内容
        if not user_raw or not item_raw:
            return None
        if "label_good" not in label or "label_best" not in label:
            return None

        # label保险
        label_good = safe_float(label["label_good"])
        label_best = safe_float(label["label_best"])
        if not (0 <= label_good <= 1) or not (0 <= label_best <= 1):
            return None

        # 图像流保险
        full_image = torch.zeros(3, 224, 224)
        if misc.get("has_image", 0) and "patch.bin" in sample:
            try:
                shape = misc["shape"]
                patch_np = np.frombuffer(sample["patch.bin"], dtype=np.uint8).copy()
                patch_np = patch_np.reshape(shape)
                patch_t = torch.from_numpy(patch_np).float() / 255.
                full_image = patch_t.permute(1, 0, 2, 3)
                full_image = full_image.reshape(3, 14, 14, 16, 16)
                full_image = full_image.permute(0, 1, 3, 2, 4)
                full_image = full_image.reshape(3, 224, 224)
                mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                full_image = (full_image - mean) / std
                if torch.isnan(full_image).any() or torch.isinf(full_image).any():
                    full_image = torch.zeros(3, 224, 224)
            except Exception as e:
                full_image = torch.zeros(3, 224, 224)

        return {
            "user_text": user_raw,
            "item_text": item_raw,
            "patch": full_image,
            "label_good": label_good,
            "label_best": label_best,
        }
    except Exception as e:
        # 捕捉所有异常，直接丢弃样本
        return None

# -------------------- 通用专家组件 --------------------
class AttnPool1D(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        # 可学习 query, 初始化为小值
        self.query = nn.Parameter(torch.randn(1,1,d) * (d ** -0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):  # x:[B,L,D], mask:[B,L]
        q = self.query.expand(x.size(0), -1, -1)
        attn = (q @ x.transpose(1,2)).squeeze(1) / (x.size(-1) ** 0.5)
        attn = attn.masked_fill(mask, float('-inf'))
        w = attn.softmax(-1)
        w = self.dropout(w)
        out = (w.unsqueeze(-1) * x).sum(1)
        return out  # [B,D]
class RobustTransformerLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

class TextExpert(nn.Module):
    def __init__(self, encoder: nn.Module, tokenizer, max_tok=384, d=768):
        super().__init__()
        self.encoder = encoder
        self.max_tok = max_tok
        self.tokenizer = tokenizer
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(0.1)
    def forward(self, input_ids, chunk2sample, sent_pos, max_sent_count, trainable=False):
        #print(f"[TextExpert] vocab_size={len(self.tokenizer)}, sent_id={self.tokenizer.convert_tokens_to_ids('<SENT>')}")
        
        return self.forward_precomputed(input_ids, chunk2sample, sent_pos, max_sent_count, trainable)

    def forward_precomputed(
        self,
        input_ids: List[List[int]],
        chunk2sample: List[int],
        sent_pos: List[List[int]],
        max_sent_count: int,
        trainable: bool = False,
    ):
        """
        input_ids:    [N_chunks, max_chunk_len]       （CPU 已经 pad 好）
        chunk2sample: [N_chunks]                       指明每个 chunk 属于哪个样本
        sent_pos:     [N_chunks, max_sents_per_chunk]   句内 <SENT> token 的位置索引，pad 用 -1
        max_sent_count: int                             batch 内最长的句子数
        trainable:    bool                              是否对 encoder 开启 grad
        """
        device = next(self.encoder.parameters()).device
        #空输入
        if not input_ids:
            B = len(set(chunk2sample)) if chunk2sample else 1  # 推断batch_size
            D = self.encoder.config.hidden_size
            return (
                torch.zeros(B, max_sent_count, D, device=device),  # sent_vecs_pad
                torch.ones(B, max_sent_count, dtype=torch.bool, device=device),  # sent_mask
                torch.zeros(B, D, device=device)  # doc_vecs
            )
            
        # 1) 转张量、前向
        x = torch.tensor(input_ids, device=device)
        attn_mask = (x != self.tokenizer.pad_token_id).long()
        token_type_ids = torch.zeros_like(x)
        position_ids = torch.arange(x.size(1), dtype=torch.long, device=device).unsqueeze(0).expand_as(x)
        current_seq_len = x.size(1)
        model_max_pos_embeddings = self.encoder.config.max_position_embeddings
        
        #print(f"DEBUG: Current input sequence length (x.size(1)): {current_seq_len}")
        #print(f"DEBUG: Model max_position_embeddings: {model_max_pos_embeddings}")
        
        #if current_seq_len > model_max_pos_embeddings:
            #print("!!!!!!!!!!!!!!!! WARNING: input_ids length exceeds model's max_position_embeddings !!!!!!!!!!!!!!!!")
            #print(f"Offending sequence length: {current_seq_len}, Model max: {model_max_pos_embeddings}")
            
            # raise ValueError("Input sequence length too long for model's position embeddings.")
        if trainable:
            h = self.encoder(
                input_ids=x, 
                attention_mask=attn_mask, 
                token_type_ids=token_type_ids, 
                position_ids=position_ids    #防止cuda底层报错
            ).last_hidden_state
        else:
            with torch.no_grad():
                h = self.encoder(
                    input_ids=x, 
                    attention_mask=attn_mask, 
                    token_type_ids=token_type_ids, 
                    position_ids=position_ids    # 防止cuda底层报错
                ).last_hidden_state 
        # h: [N_chunks, max_chunk_len, D]

        # 2) 取出每个 chunk 里那些 <SENT> token 的向量
   
        pos = torch.tensor(sent_pos, device=device)               # [N_chunks, S_chunk]
        N_chunks, S_chunk = pos.shape
        D = h.size(-1)
        seq_len = h.size(1)

        batch_idx = torch.arange(N_chunks, device=device).unsqueeze(1).expand(N_chunks, S_chunk)

        # 所有索引clamp到合法区间
        pos_clamped = pos.clamp(min=0, max=seq_len - 1)           # <- 关键补丁
        sent_vecs = h[batch_idx, pos_clamped]                     # [N_chunks, S_chunk, D]
        sent_vecs = sent_vecs.masked_fill(pos.unsqueeze(-1) < 0, 0.0)

        # 3) 聚回每个样本：根据 chunk2sample 分桶
        B = int(max(chunk2sample)) + 1
        sample_buckets: List[List[torch.Tensor]] = [[] for _ in range(B)]
        for i, sidx in enumerate(chunk2sample):
            # sent_vecs[i]: [S_chunk, D]
            sample_buckets[sidx].append(sent_vecs[i])

        # 4) 样本级 pad 到 max_sent_count
        padded_sents = []
        for bucket in sample_buckets:
            if bucket:
                # cat 所有 chunk 产生的句向量
                cat = torch.cat(bucket, dim=0)  # [n_sents_i, D]
            else:
                cat = torch.zeros(1, D, device=device)
            n_i = cat.size(0)

            # ★ 裁剪超长
            if n_i > max_sent_count:
                cat = cat[:max_sent_count]
                n_i = max_sent_count

            pad_n = max_sent_count - n_i
            if pad_n > 0:
                # 在 time 维度后面 pad pad_n 行
                cat = F.pad(cat, (0, 0, 0, pad_n))
            padded_sents.append(cat.unsqueeze(0))  # [1, max_sent_count, D]
        sent_vecs_pad = torch.cat(padded_sents, dim=0)  # [B, max_sent_count, D]
        sent_mask    = (sent_vecs_pad.abs().sum(-1) == 0)  # pad 行标 True

        # 5) 文档级池化（对非 pad 行求均值）
        lens = (~sent_mask).sum(dim=1, keepdim=True)      # [B, 1]
        doc_vecs = sent_vecs_pad.sum(dim=1) / lens.clamp(min=1)

        # 6) 最后再归一化 + dropout
        sent_vecs_pad = self.dropout(self.norm(sent_vecs_pad))
        doc_vecs      = self.dropout(self.norm(doc_vecs))

        return sent_vecs_pad, sent_mask, doc_vecs




class ItemImageExpert(nn.Module):
    """
    通用图像专家（完整图像版本）
    ------------------------------------
    • 输入:  images  [B, 3, 224, 224]  (float32，已 (x-mean)/std 归一化)
    • 输出:  img_vec [B, hidden_dim]   (hidden_dim 由 base_model.config.hidden_size 决定)
    """
    def __init__(
        self,
        base_model: nn.Module,    # 由 build_img_expert 传进来，可能带 LoRA
        pool_type: str = "mean",
        dropout_p: float = 0.1,
    ):
        super().__init__()
        assert pool_type in {"mean", "cls"}, "`pool_type` 只能为 'mean' 或 'cls'"

        self.backbone  = base_model          # e.g. DINOv2Model, ViTModel, 等
        self.pool_type = pool_type
        self.dropout   = nn.Dropout(dropout_p)

        hidden_dim     = base_model.config.hidden_size
        self.norm      = nn.LayerNorm(hidden_dim)

    # --------------------------------------------
    def forward(self, images: torch.Tensor, trainable: bool = False):
        #print("ItemImageExpert got:", images.shape)
        #print("ItemImageExpert backbone type:", type(self.backbone))
        #print("ItemImageExpert backbone forward:", self.backbone.forward.__code__.co_varnames)
        # images: [B, 3, 224, 224], float32, 已归一化
        if trainable:
            output = self.backbone(pixel_values=images)
        else:
            with torch.no_grad():
                output = self.backbone(pixel_values=images)
        tokens = output.last_hidden_state      # [B, 197, hidden_dim]
    
        if self.pool_type == "mean":
            img_vec = tokens.mean(dim=1)
        else:
            img_vec = tokens[:, 0]
    
        img_vec = self.dropout(self.norm(img_vec))
        return img_vec
class RobustTextCrossExpert(nn.Module):
    def __init__(self, d=768, n_layer=2, n_head=8, dropout=0.1):
        super().__init__()
        # 使用更稳定的Transformer实现
        self.self_user = nn.ModuleList([
            RobustTransformerLayer(
                d_model=d, nhead=n_head,
                dim_feedforward=4*d, dropout=dropout,
                batch_first=True, norm_first=True)
            for _ in range(n_layer)
        ])
        
        self.self_item = nn.ModuleList([
            RobustTransformerLayer(
                d_model=d, nhead=n_head,
                dim_feedforward=4*d, dropout=dropout,
                batch_first=True, norm_first=True)
            for _ in range(n_layer)
        ])
        
        # 改进的交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            d, n_head, dropout=dropout, batch_first=True)
        
        # 门控残差连接
        self.gate = nn.Parameter(torch.tensor([0.5]))  # 可学习门控
        
        # 池化层
        self.pool = AttnPool1D(d, dropout)
        
        # 输出处理
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4*d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d, d),
            nn.Dropout(dropout)
        )

    def forward(self, user_vecs, user_mask, item_vecs, item_mask):
        # 自注意力编码
        for layer in self.self_user:
            user_vecs = layer(user_vecs, src_key_padding_mask=user_mask)
        
        for layer in self.self_item:
            item_vecs = layer(item_vecs, src_key_padding_mask=item_mask)
        
        # 交叉注意力
        cross_out = self.cross_attn(
            query=user_vecs,
            key=item_vecs,
            value=item_vecs,
            key_padding_mask=item_mask
        )[0]
        
        # 门控残差连接
        alpha = torch.sigmoid(self.gate)
        fused = alpha * user_vecs + (1 - alpha) * cross_out
        
        # 池化
        pooled = self.pool(fused, user_mask)
        
        # 输出处理
        normed = self.norm(pooled)
        return normed + self.mlp(normed)


class EnhancedCrossFuse(nn.Module):
    """改进的跨模态融合专家"""
    def __init__(self, d=768, n_head=8, depth=2, dropout=0.1):
        super().__init__()
        # 跨模态交互层
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d, nhead=n_head,
                dim_feedforward=4*d,
                dropout=dropout, batch_first=True,
                norm_first=True)
            for _ in range(depth)
        ])
        
        # 残差连接路径
        self.res_proj = nn.Sequential(
            nn.Linear(2*d, d),
            nn.LayerNorm(d)
        )
        
        # 稳健门控机制
        self.gate = nn.Sequential(
            nn.Linear(2*d, d//2),
            nn.GELU(),
            nn.Linear(d//2, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate[2].bias, 0.5)  # 中性初始化
        
        # 最终投影
        self.proj = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, v_cls, t_cls):
        # 保留原始特征
        identity = self.res_proj(torch.cat([v_cls, t_cls], -1))
        
        # 跨模态交互
        x = torch.stack([v_cls, t_cls], 1)
        for layer in self.layers:
            x = layer(x)
        v_fused, t_fused = x[:,0], x[:,1]
        
        # 门控融合
        gate_input = torch.cat([v_fused, t_fused], -1)
        g = self.gate(gate_input)
        fused = g * v_fused + (1-g) * t_fused
        
        # 残差连接+投影
        return self.proj(fused + identity)





class DenseGate(nn.Module):
    """
    非稀疏 Gate：直接 softmax 得到每个专家的权重。
    没有 bias / Top-K，参数量极少，只含一个线性层。
    """
    def __init__(self, in_dim: int, n_expert: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_expert)

    def forward(self, x: torch.Tensor):
        # x: [B, D] → weights: [B, n_expert]
        return F.softmax(self.fc(x), dim=-1)


class TwoTaskMMoE(nn.Module):
    """
    • expert_vecs 由上游所有专家拼成 [B, n_expert, D]  
    • 此处只做 Dense Gate 融合 + 两个任务塔
    """
    def __init__(
        self,
        expert_dim: int = 768,
        n_expert: int = 6,
        tower_hidden: int = 256,
        tower_dropout: float = 0,   
    ):
        super().__init__()

        # 两个任务各自的 Dense Gate
        self.gate_good = DenseGate(expert_dim, n_expert)
        self.gate_best = DenseGate(expert_dim, n_expert)

        # -------- 任务塔，共用一套结构 ----------
        def _make_tower():
            return nn.Sequential(
                nn.LayerNorm(expert_dim),
                nn.Linear(expert_dim, tower_hidden),
                nn.GELU(),
                nn.Dropout(tower_dropout),
                # 增加一个中间层
                nn.Linear(tower_hidden, tower_hidden//2),
                nn.GELU(),
                nn.Dropout(tower_dropout),
                nn.Linear(tower_hidden//2, 1),
            )
        self.tower_good = _make_tower()
        self.tower_best = _make_tower()

    # -------------------------------------------------
    def forward(self, expert_vecs: torch.Tensor):          # [B, N, D]
        # 1) Gate 取 query 先对 expert 取均值
        query = expert_vecs.mean(1)                        # [B, D]

        w_good = self.gate_good(query)                    # [B, N]
        w_best = self.gate_best(query)

        # 2) 加权求和得到融合特征
        fused_good = (w_good.unsqueeze(-1) * expert_vecs).sum(1)  # [B, D]
        fused_best = (w_best.unsqueeze(-1) * expert_vecs).sum(1)

        # 3) 进入任务塔
        logit_good = self.tower_good(fused_good).squeeze(-1)      # [B]
        logit_best = self.tower_best(fused_best).squeeze(-1)

        return logit_good, logit_best




# -------------------函数----------------------------

# ================= 更新的构建函数 =================
def build_text_user_expert(
    model_name: str,
    lora_r: int,
    max_tok: int,
    tokenizer,            
    device: torch.device  
) -> TextExpert:
    # 1) 加载基础模型并加 LoRA
    cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r, lora_alpha=32, lora_dropout=0.1
    )
    base = AutoModel.from_pretrained(model_name)
    base.resize_token_embeddings(len(tokenizer))  
    peft_model = get_peft_model(base, cfg).to(device)

    # 2) wrap 并 to(device)
    return TextExpert(peft_model, tokenizer, max_tok=max_tok).to(device)


def build_text_item_expert(
    model_name: str,
    lora_r: int,
    max_tok: int,
    tokenizer,            
    device: torch.device 
) -> TextExpert:
    cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r, lora_alpha=32, lora_dropout=0.1
    )
    base = AutoModel.from_pretrained(model_name)
    base.resize_token_embeddings(len(tokenizer))  
    peft_model = get_peft_model(base, cfg).to(device)

    return TextExpert(peft_model, tokenizer, max_tok=max_tok).to(device)
    

def build_img_expert(model_name: str, pool_type: str, device: torch.device):
    base = ViTModel.from_pretrained(model_name)
    return ItemImageExpert(
        base_model=base,
        pool_type=pool_type,
    ).to(device)

def build_cross_expert(d: int = 768, n_layer: int = 2, n_head: int = 8, 
                      dropout: float = 0.1, device: torch.device = None) -> RobustTextCrossExpert:
    """构建文本交叉专家"""
    return RobustTextCrossExpert(
        d=d,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout
    ).to(device)


def build_concat_ui_expert(d: int = 768, n_head: int = 8, depth: int = 2, 
                                   dropout: float = 0.1, device: torch.device = None)-> EnhancedCrossFuse:
    return EnhancedCrossFuse(
        d=d,
        n_head=n_head,
        depth=depth,
        dropout=dropout
    ).to(device)
# concat_ti: item text + item image

def build_concat_ti_expert(d: int = 768, n_head: int = 8, depth: int = 2, 
                                   dropout: float = 0.1, device: torch.device = None)-> EnhancedCrossFuse:
    return EnhancedCrossFuse(
        d=d,
        n_head=n_head,
        depth=depth,
        dropout=dropout
    ).to(device)
