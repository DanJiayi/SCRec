# 多模态对比学习增强RQ-VAE推荐系统

## 🎯 系统概述

本系统实现了一个创新的推荐模型：**多模态对比学习增强RQ-VAE**。该模型的核心思想是：

1. **训练阶段**：使用文本信息通过对比学习来增强码本的语义表示
2. **推理阶段**：完全不需要文本输入，码本已经内化了语义结构
3. **优势**：既保持了文本语义信息，又实现了高效的推理

## 🏗️ 系统架构

### 核心组件

```
ContrastiveRQVAERec
├── Item Embedding Layer (物品ID嵌入层)
├── Transformer Encoder (序列编码器)
├── RQ-VAE Quantizer (残差向量量化器)
├── Contrastive Projection Heads (对比学习投影头)
├── Output Layer (输出层)
└── Logit Scale (可学习温度参数)
```

### 数据流

```
训练阶段:
物品ID序列 + 文本嵌入 → Transformer → RQ-VAE量化 → 对比学习 → 损失计算

推理阶段:
物品ID序列 → Transformer → RQ-VAE量化 → 推荐预测
```

## 📁 文件结构

```
quantization/
├── train_contrastive_rqvae.py          # 主训练脚本
├── prepare_contrastive_data.py         # 数据预处理脚本
├── contrastive_rqvae_config.yaml       # 配置文件
├── run_contrastive_training.sh         # 训练启动脚本
├── README_Contrastive_RQVAE.md        # 本文档
└── rqvae/                             # 原有RQ-VAE实现
```

## 🚀 快速开始

### 1. 环境准备

确保安装了必要的依赖：

```bash
pip install torch torchvision torchaudio
pip install numpy tqdm pyyaml
pip install tensorboard  # 可选，用于可视化
```

### 2. 数据准备

系统会自动处理数据，但需要确保以下文件存在：

```
cache/AmazonReviews2014/Toys_and_Games/processed/
├── final_pca_embeddings.npy           # PCA降维后的文本嵌入
├── final_pca_embeddings_mapping.json  # 物品ID到文本嵌入的映射
└── 其他必要的数据文件
```

### 3. 开始训练

#### 方法1：使用自动化脚本（推荐）

```bash
cd quantization
./run_contrastive_training.sh
```

#### 方法2：手动执行

```bash
# 第一步：数据预处理
python prepare_contrastive_data.py \
    --data_dir cache/AmazonReviews2014/Toys_and_Games/processed \
    --output_dir cache/AmazonReviews2014/Toys_and_Games/contrastive_data \
    --max_seq_len 128 \
    --min_seq_len 5

# 第二步：开始训练
python train_contrastive_rqvae.py --config contrastive_rqvae_config.yaml
```

## ⚙️ 配置说明

### 主要配置参数

```yaml
dataset:
  name: "Toys_and_Games"        # 数据集名称
  max_items_per_seq: 128        # 最大序列长度
  min_items_per_seq: 5          # 最小序列长度

model:
  embedding_dim: 512            # 物品嵌入维度
  num_layers: 6                 # Transformer层数
  num_quantizers: 8             # 量化器数量
  codebook_size: 256            # 码本大小

training:
  epochs: 1000                  # 训练轮数
  batch_size: 128               # 批次大小
  lr: 0.001                     # 学习率
  
  # 损失权重
  lambda_recon: 1.0             # 重构损失权重
  lambda_commit: 0.25           # Commitment损失权重
  lambda_contrast: 0.1          # 对比损失权重
```

### 损失函数说明

1. **重构损失 (Reconstruction Loss)**
   - 目标：预测序列中的下一个物品
   - 权重：1.0（主要损失）

2. **量化损失 (Commitment Loss)**
   - 目标：确保量化后的向量接近原始向量
   - 权重：0.25（参考RQ-VAE论文）

3. **对比损失 (Contrastive Loss)**
   - 目标：让量化表示和文本表示在投影空间中相似
   - 权重：0.1（需要调试）

## 📊 训练监控

### 日志输出

训练过程中会显示：

```
Epoch 1/1000
  Total Loss: 8.2345
  Reconstruction Loss: 6.1234
  Commitment Loss: 1.2345
  Contrastive Loss: 0.8766
  Learning Rate: 0.001000
```

### 检查点保存

- 每100轮保存一次检查点
- 最终模型保存为 `contrastive_rqvae_final.pt`
- 检查点包含模型状态、优化器状态、配置等

### TensorBoard支持

启用TensorBoard监控训练过程：

```bash
tensorboard --logdir logs/contrastive_rqvae
```

## 🔮 推理使用

训练完成后，模型可以用于推理：

```python
import torch
from train_contrastive_rqvae import ContrastiveRQVAERec

# 加载训练好的模型
checkpoint = torch.load('ckpt/contrastive_rqvae_final.pt')
model = ContrastiveRQVAERec(...)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理（不需要文本输入！）
item_seq = torch.LongTensor([[1, 5, 23, 67, 89]])  # 用户历史序列
with torch.no_grad():
    next_item_logits = model.inference(item_seq)
    next_item_probs = torch.softmax(next_item_logits, dim=-1)
    
    # 获取Top-K推荐
    top_k = 10
    top_probs, top_indices = torch.topk(next_item_probs, top_k)
    
    print(f"Top-{top_k} 推荐物品:")
    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
        print(f"  {i+1}. 物品ID: {idx.item()}, 概率: {prob.item():.4f}")
```

## 🎯 核心优势

### 1. **语义保持**
- 通过对比学习，码本内化了文本语义信息
- 推理时不需要文本输入，但保持了语义理解能力

### 2. **效率提升**
- 训练时使用文本信息增强学习
- 推理时只需要物品ID，计算效率高

### 3. **可扩展性**
- 支持不同的文本嵌入模型
- 可调整的码本大小和层数
- 灵活的损失权重配置

## 🔧 高级功能

### 1. **码本可视化**
```python
# 可视化码本向量
codebooks = model.codebooks.detach().cpu().numpy()
# 使用t-SNE或PCA进行降维可视化
```

### 2. **对比学习分析**
```python
# 分析量化表示和文本表示的相似度
quantized_proj = model.quantized_proj_head(quantized_seq)
text_proj = model.text_proj_head(text_embeddings)
similarity = F.cosine_similarity(quantized_proj, text_proj, dim=-1)
```

### 3. **码本更新策略**
- 支持EMA（指数移动平均）更新
- 可配置的更新频率和衰减率

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 减小max_seq_len
   - 使用梯度累积

2. **训练不收敛**
   - 调整学习率
   - 检查损失权重
   - 增加训练轮数

3. **数据加载错误**
   - 检查数据路径
   - 验证数据格式
   - 检查映射文件

### 调试技巧

1. **启用详细日志**
```python
logging.basicConfig(level=logging.DEBUG)
```

2. **检查中间输出**
```python
# 在forward方法中添加print语句
print(f"Transformer output shape: {transformer_output.shape}")
```

3. **梯度检查**
```python
# 检查梯度是否正常
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm()}")
```

## 📚 参考文献

1. RQ-VAE: Residual Quantizer for Vector Quantization
2. Contrastive Learning for Representation Learning
3. Transformer Architecture for Sequential Recommendation

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个系统！

## �� 许可证

本项目采用MIT许可证。

