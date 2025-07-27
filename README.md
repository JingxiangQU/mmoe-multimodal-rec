# End-to-End Multimodal Recommendation System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Apache Beam](https://img.shields.io/badge/Apache%20Beam-2.5x-yellow.svg)](https://beam.apache.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> 本项目是基于 Apache Beam 和 PyTorch DDP 构建的，从原始数据到模型训练的端到端多模态推荐系统。项目包括分布式特征工程、高效数据加载、复杂模型（MMoE）的分布式训练与微调。

---

## 核心特性

- **端到端全流程**: 覆盖从数据拉取、处理到模型训练的完整闭环。
- **可拓展架构**: 特征工程与模型训练完全解耦，提高系统可扩展性。
- **高性能数据管道**: 使用 WebDataset 存储，解决云存储环境下的I/O瓶颈。
- **先进模型架构**: 实现复杂的多任务专家混合模型 (MMoE)。
- **高效分布式训练**: 使用 PyTorch DistributedDataParallel (DDP) 训练。

---

## 系统架构

本项目的核心设计思想是将整个推荐流程拆分为两个独立但衔接的阶段：**离线特征工程** 和 **模型训练**。

```mermaid
graph TD
    %% High-Level Pipeline
    subgraph "Phase 1: Offline Data Engineering (Apache Beam on CPU Cluster)"
        A[Raw Data on Hugging Face] --> B(Data Ingestion Scripts);
        B --> C{GCS Storage: Raw JSONL};
        C --> D[Beam Pipeline 1: data4moe_beam.py];
        D -- Generates --> E[User/Item Features];
        D -- Generates --> F(Image URLs List);
        F --> G[Beam Pipeline 2: newpatch.py];
        G -- Distributed Image Fetching & Patching --> H(Image Patches Data);
        E & H --> I[Beam Pipeline 3: data4model.py];
        I -- Joins Features & Image Patches --> J(Processed Data in WebDataset format on GCS);
    end

    subgraph "Phase 2: Model Training (PyTorch DDP on GPU Cluster)"
        J --> K[WebDataset Loader];
        K --> L(Distributed Training Pipeline: train.py);
        L --> M[Multi-GPU Training with DDP];
        M --> N{MMoE Model};
        N --> O[Multi-Task Prediction];
        O --> P(Saved Checkpoints & Performance Plots);
    end

    %% Detailed MMoE Model Structure
    subgraph "Input Modalities"
        U[User Text] --> P1(preprocess_batch);
        I_text[Item Text] --> P2(preprocess_batch);
        IMG[Item Image Patches] --> IIME(ItemImageExpert);
    end

    subgraph "Text Experts (LoRA Tuned)"
        direction LR
        P1 --> UE(TextExpert: User);
        P2 --> IE(TextExpert: Item);
    end

    subgraph "Expert Feature Extraction"
        UE -- u_doc --> MM1[Expert Vecs Collection];
        IE -- i_doc --> MM1;
        IIME -- img_vec --> MM1;

        UE -- u_sent, u_mask --> RC(RobustTextCrossExpert: Cross-UI Attention);
        IE -- i_sent, i_mask --> RC;
        RC -- ui_vec --> MM1;

        UE -- u_doc --> CUI(EnhancedCrossFuse: User-Image Concat);
        IIME -- img_vec --> CUI;
        CUI -- xui --> MM1;

        IE -- i_doc --> CTI(EnhancedCrossFuse: Item-Image Concat);
        IIME -- img_vec --> CTI;
        CTI -- xti --> MM1;
    end

    subgraph "MMoE Head (TwoTaskMMoE)"
        MM1 --> QH("Query Generation: mean(expert_vecs)");

        QH -- Query --> G1(DenseGate: Task Good);
        QH -- Query --> G2(DenseGate: Task Best);

        G1 -- Weights --> F1(Weighted Sum of Expert Vecs);
        G2 -- Weights --> F2(Weighted Sum of Expert Vecs);

        F1 --> T1(Task Tower: Good);
        F2 --> T2(Task Tower: Best);
    end

    T1 -- Logit --> L_GOOD(Label Good Prediction);
    T2 -- Logit --> L_BEST(Label Best Prediction);

    %% Arrow connecting the detailed view to the high-level block
    T1 --> N;
    T2 --> N;

    %% Styling
    style J fill:#f9f,stroke:#333,stroke-width:2px
    style K fill:#ccf,stroke:#333,stroke-width:2px
    style UE fill:#e0f7fa,stroke:#0097a7,stroke-width:2px
    style IE fill:#e0f7fa,stroke:#0097a7,stroke-width:2px
    style IIME fill:#fffde7,stroke:#ffc107,stroke-width:2px
    style RC fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    style CUI fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    style CTI fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    style MM1 fill:#fce4ec,stroke:#d81b60,stroke-width:2px
    style QH fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    style G1 fill:#ffebee,stroke:#f44336,stroke-width:2px
    style G2 fill:#ffebee,stroke:#f44336,stroke-width:2px
    style T1 fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px
    style T2 fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px
```
---

## 特征工程效果展示

以下是一个从Beam流水线中生成的真实用户画像样本：

```json
{
  "user_id": "USER_EXAMPLE_ID",
  "user_feat": {
    "cat_hist": {
      "All Electronics": 0.14,
      "Tools & Home Improvement": 0.43,
      "Sports & Outdoors": 0.14
    },
    "review_cnt": 7,
    "price_mean": 25.59,
    "price_std": 25.5,
    "history": [
      {
        "title": "Do not buy...did not work, will not hold a charge.",
        "text": "Purchased in Dec, first charge to full capacity before use mid January in Makita charger that came with drill. Neither battery held a sufficient charge to be useful..."
      },
      {
        "title": "20 amp 250v plug",
        "text": "Sometimes hard to find this 20 amp 250v plug was just what ai needed and I found it on Amazon. Superior Electric had it to me in good time at a fair price."
      }
    ]
  }
}
```

### 亮点解读 (Key Insights)

- **为句级交叉注意力赋能**: 特征工程中有意识地将用户多条历史评论整合成完整文本段落。
- **精准分句作为桥梁**: 模型训练预处理阶段精准切分文本段落成句子，实现细粒度语义理解。
- **实现深度语义交互**: 句向量序列被送入交叉注意力模块，实现用户历史兴趣和物品描述间的深度语义交互。

---
## 项目结构

```text
.
├── data4moe_beam.py     # 主特征工程脚本 (Beam)，生成用户/物品特征和图片URL列表， 按照日期划分train/valid/test，严防特征穿越
├── newpatch.py          # 分布式图片处理脚本 (Beam)，将URL转换为图像patch数据
├── data4model.py        # 合并所有特征并生成WebDataset的脚本 (Beam)
├── meta2gcs.py          # 从Hugging Face下载元数据的辅助脚本
├── review2gcs.py        # 从Hugging Face下载评论数据的辅助脚本
├── model.py             # 定义所有PyTorch模型架构 (MMoE, Experts)
├── train.py             # 核心模型训练脚本 (PyTorch DDP)
├── requirements.txt     # 项目依赖
└── README.md            # 本文档
```
---
## 环境安装

```shell
git clone [你的仓库URL]
cd [你的仓库目录]

conda create -n mmoe_rec python=3.11
conda activate mmoe_rec

pip install -r requirements.txt

python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## 如何运行

### 数据准备

```shell
python meta2gcs.py --bucket [你的GCS桶名称]
python review2gcs.py --bucket [你的GCS桶名称]

python data4moe_beam.py --[相关参数]
python newpatch.py --[相关参数]
python data4model.py --output_dir gs://[你的GCS路径]
```

### 模型训练

```shell
export TOKENIZERS_PARALLELISM=false

torchrun --nproc_per_node=2 train.py \
  --model_name BAAI/bge-base-en-v1.5 \
  --img_model google/vit-base-patch16-224-in21k \
  --data_pattern '/path/to/your/local/wds_shards/data-*-*.tar.gz' \
  --batch_size 128 \
  --grad_accum 8 \
  --epochs 4 \
  --num_workers 32 \
  --lora_r 8 \
  --lr 1e-5 \
  --output_dir /workspace/outputs
```

---

## 性能优化与成果

- **解决I/O瓶颈**: 早期实验发现，直接从GCS流式读取数据导致严重的I/O瓶颈，GPU利用率极低。通过将数据预先下载到本地磁盘，彻底解决了该问题。
- **优化CPU预处理**: spaCy分词器成为新的CPU瓶颈。通过替换为轻量级的NLTK分词器，数据预处理速度提升了超过50倍，显著提高了GPU的平均利用率。
- **精细化显存管理**: 在LoRA微调解冻时遇到了反复的CUDA OOM问题。通过系统性地下调batch_size、增加梯度累积步数、限制max_tok，最终找到了一个可以在80GB显存内稳定运行的最佳配置，实现了高达91%的显存利用率。


---
## 训练结果
<img width="640" height="480" alt="global_step_loss_curve" src="https://github.com/user-attachments/assets/6372e575-8577-4d20-8280-9268df3bdb8b" />
<img width="640" height="480" alt="epoch_interval_loss_curve" src="https://github.com/user-attachments/assets/51c1126e-e913-439f-aec8-05979c3c9cc9" />

- **注**：共4个epoch，图2中每个epoch重复计数了一次，已定位问题并在`train.py`中修改。
  
---

### 现有不足

-   **loss曲线没有稳定下降**: 当前的训练损失曲线并未呈现理想的稳定下降趋势。初步诊断和行动计划如下：
    * **初步分析**:
        * **学习率问题**: 当前学习率（如1e-5）可能相对较高，导致优化过程在最优解附近震荡或直接跳过。
        * **Warmup与Gate稳定性**: 初始化参数的Warmup步数可能不足，导致LoRA微调启动时，MMoE的Gate权重尚未收敛，对专家的选择仍处于不稳定状态，影响模型学习。
        * **脏数据干扰**: 训练数据中未能成功抓取图片的样本被处理为“全零图片”，这部分无效样本的占比较高，可能导致模型学到的是噪声而非有效特征，从而产生梯度扰动。
        * **数据流与收敛性**: 在每个epoch都引入未学习过的新样本的训练模式下，模型可能需要更多的训练步数才能在庞大的数据空间上达到充分收敛。

    * **行动计划与实验设计**: 针对以上四个潜在问题，已规划三组并行的对照实验进行定位和验证
        * **实验A（学习率和Warmup）**: 固定其他条件，测试三组学习率（1e-5, 5e-6, 1e-6）和两组Warmup步数（1000, 3000）的组合。同时，将记录Gate层输出的熵（entropy）或标准差（standard deviation）作为其稳定性的量化监控指标。
        * **实验B（脏数据分析）**: 修改`model.py`中的数据加载逻辑，精确统计并打印每个epoch中“全零图片”样本的占比。若比例显著（如 > 5%），则优先修复`newpatch.py`中的图片获取和处理逻辑；反之则暂时搁置此问题。
        * **实验C（收敛性分析）**: 保持现有训练策略，但将总训练步数延长50%，并以更细的粒度在独立的验证集上记录AUC指标的变化。通过绘制更长的学习曲线，判断模型是否存在持续收敛的潜力，并评估当前训练步数是否充足。

-   **数据清洗逻辑有待加强**: 当前在`data4model.py`中的`normalize_text`函数虽然实现了一些基础的文本规范化，但对于真实世界中更为嘈杂和非结构化的文本（如俚语、拼写错误、特殊符号等）处理能力有限，逻辑有待进一步加强。

-   **特征拼接方式相对简单**: `data4model.py`中的`build_user_text`和`build_item_text`函数将多源特征直接拼接成一个长字符串。没有考虑特征间的结构化信息，增加了模型从纯文本中分离和理解不同语义信息的负担。

-   **训练中缺失AUC评估**: `train.py`脚本中，由于样本量巨大，在每个step后计算AUC耗时过长，因此当前的训练脚本中并未包含在线的AUC计算。 这导致训练过程中无法实时监控模型的泛化能力。下一步计划是在独立的验证集上进行完整的离线AUC评测（优先度最高）。

### 未来工作方向

-   **离线特征预处理**: 将分词和Tokenizer编码等CPU密集型任务完全离线化，将最终的Token ID存为二进制文件，进一步压榨数据加载性能，让GPU利用率达到极致。
-   **流式处理**: 思考如何将Beam流水线从批处理模式切换到流处理模式，对接实时用户行为数据流，实现准实时的特征更新和模型训练。
-   **模型架构探索**: 尝试更复杂的交叉注意力结构，以更好地捕捉多模态特征间的细粒度交互; 在MMoE中引入稀疏门控机制（Top-K Gating）。 受到近期前沿研究（如DeepSeek-V2的LFBias）的启发，未来可以探索更先进的负载均衡策略，以确保稀疏门控下各个专家的均衡利用，防止模型退化。

---

