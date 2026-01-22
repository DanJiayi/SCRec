# Logs文件夹组织结构

本文件夹包含了所有实验的日志文件，按照数据集和实验类型进行了分类整理。

## 文件夹结构

### 📁 toys/
玩具数据集相关的所有日志文件
- `toys64_nohup.out` - 64维玩具数据集训练日志
- `toys64_without_text_nohup.out` - 64维玩具数据集（无文本）训练日志
- `toys16_nohup_games.out` - 16维玩具数据集游戏相关训练日志
- `toys_nohup_games.out` - 玩具数据集游戏相关训练日志
- `toys_8_codebook.out` - 8维玩具数据集码本训练日志
- `toys_32_codebook.out` - 32维玩具数据集码本训练日志
- `toys_64_codebook.out` - 64维玩具数据集码本训练日志
- `toys8_without_text_nohup.out` - 8维玩具数据集（无文本）训练日志

### 📁 beauty/
美容数据集相关的所有日志文件
- `beauty_64.out` - 64维美容数据集训练日志
- `beauty_64_codebook.out` - 64维美容数据集码本训练日志
- `beauty_nohup_rqvae_training_300epoch.out` - 美容数据集RQVae训练300轮日志
- `nohup_beauty_fixed.out` - 美容数据集修复版本训练日志
- `nohup_beauty_test.out` - 美容数据集测试日志

### 📁 sports/
运动数据集相关的所有日志文件
- `nohup_sports16_main.out` - 16维运动数据集主训练日志
- `nohup_sports64_main.out` - 64维运动数据集主训练日志
- `nohup_sports_rqvae_16d.out` - 16维运动数据集RQVae训练日志
- `nohup_sports_rqvae_64d.out` - 64维运动数据集RQVae训练日志
- `sports_nohup_training_300epoch.out` - 运动数据集300轮训练日志
- `nohup_sports_codebook.out` - 运动数据集码本训练日志
- `sports_nohup_rqvae_training.out` - 运动数据集RQVae训练日志

### 📁 cds/
CD数据集相关的所有日志文件
- `cds_nohup_vinyl.out` - CD数据集黑胶唱片相关训练日志

### 📁 text/
文本相关实验的所有日志文件
- `nohup.text_pca256_out` - 256维文本PCA实验日志
- `nohup.text_pca64_out` - 64维文本PCA实验日志
- `nohup.text_pca_out` - 文本PCA实验日志
- `nohup.text_out` - 文本实验日志

### 📁 general/
通用训练和实验日志文件
- `main_model_training.log` - 主模型训练日志
- `rqvae_training.log` - RQVae训练日志
- `rqvae_training_5000epochs.log` - RQVae 5000轮训练日志
- `training_5000epochs.log` - 5000轮训练日志
- `training_5000epochs_fixed.log` - 5000轮训练修复版本日志
- `training_90epochs.log` - 90轮训练日志
- `nohup_rqvae_rpg.out` - RQVae RPG训练日志
- `nohup_rqvae_rpg_final.out` - RQVae RPG最终训练日志
- `nohup_rqvae_rpg_final_fixed.out` - RQVae RPG最终修复版本训练日志
- `nohup_rqvae_rpg_fixed.out` - RQVae RPG修复版本训练日志

### 📁 rqvae/
RQVae相关的专门日志文件（已存在）
- 包含各种RQVae实验的详细日志

### 📁 tensorboard/
TensorBoard日志文件（已存在）
- 用于可视化训练过程

### 📁 AmazonReviews2014/
Amazon评论数据集相关日志（已存在）

## 文件命名规范

- `nohup_*.out` - 后台运行的训练任务日志
- `*_codebook.out` - 码本训练相关日志
- `*_training*.log` - 训练过程日志
- `*_test.out` - 测试相关日志
- `*_fixed.out` - 修复版本日志

## 注意事项

1. 所有日志文件已按照数据集类型进行分类整理
2. 每个子文件夹包含该数据集的所有相关实验日志
3. 通用训练日志放在general文件夹中
4. 原有的rqvae、tensorboard等文件夹保持不变
