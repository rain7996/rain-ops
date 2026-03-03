import os
import json
import shutil
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

def extract_deepseek_layers(input_path, output_path, num_layers=10, with_mtp=True):
    """
    提取 DeepSeek V3 的前 N 层，并可选地将原第 61 层(MTP)重命名为第 N 层。
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 1. 加载并修改 config.json
    config_path = os.path.join(input_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    original_total_layers = config.get("num_hidden_layers", 61) # DeepSeek V3 默认为 61
    config["num_hidden_layers"] = num_layers
    
    # 显式保留 MTP 配置参数（如果原 config 存在）
    if with_mtp:
        # 确保预测层数配置正确，vLLM 依赖这些字段识别 MTP
        config["num_nextn_predict_layers"] = config.get("num_nextn_predict_layers", 1)

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 2. 解析原始权重索引
    index_path = os.path.join(input_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)

    old_weight_map = index_data["weight_map"]
    new_weight_map = {}
    # 建立 shard 到 (旧Key, 新Key) 的映射，避免重复读取大文件
    shard_to_tasks = {}

    for weight_name, shard_file in old_weight_map.items():
        keep = False
        new_name = weight_name
        
        # A. 基础组件 (Embedding, Norm, Head)
        if any(x in weight_name for x in ["model.embed_tokens", "model.norm", "lm_head"]):
            keep = True
        
        # B. Transformer 层与 MTP 层逻辑
        else:
            parts = weight_name.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_idx = int(parts[2])
                
                # 提取普通层 (0 到 num_layers-1)
                if layer_idx < num_layers:
                    keep = True
                
                # 提取 MTP 层 (原索引为 61)，重命名为 num_layers (即第 10 层)
                elif with_mtp and layer_idx == original_total_layers:
                    parts[2] = str(num_layers)
                    new_name = ".".join(parts)
                    keep = True
        
        if keep:
            new_weight_map[new_name] = shard_file
            shard_to_tasks.setdefault(shard_file, []).append((weight_name, new_name))

    # 3. 物理提取并保存权重
    print(f"Starting extraction: {num_layers} layers + MTP (as layer {num_layers})")
    
    # 追踪哪些 shard 真正被写入了，用于更新最终索引
    final_weight_map = {}

    for shard, tasks in tqdm(shard_to_tasks.items(), desc="Processing Shards"):
        input_shard_path = os.path.join(input_path, shard)
        output_shard_path = os.path.join(output_path, shard)
        
        new_shard_weights = {}
        with safe_open(input_shard_path, framework="pt", device="cpu") as f:
            for old_key, new_key in tasks:
                new_shard_weights[new_key] = f.get_tensor(old_key)
                final_weight_map[new_key] = shard
        
        if new_shard_weights:
            save_file(new_shard_weights, output_shard_path)

    # 4. 保存修正后的索引文件
    new_index_data = {
        "metadata": index_data.get("metadata", {}),
        "weight_map": final_weight_map
    }
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index_data, f, indent=2)

    # 5. 复制辅助文件
    copy_files = ["tokenizer.json", "tokenizer_config.json", "generation_config.json"]
    for file in copy_files:
        src = os.path.join(input_path, file)
        if os.path.exists(src):
            shutil.copy(src, output_path)

    print(f"\nSUCCESS: Model saved to {output_path}")
    print(f"MTP layers now mapped to index: {num_layers}")

if __name__ == "__main__":
    # 配置路径
    SOURCE = "/disc/data1/model/DeepSeek-V3.2" 
    TARGET = "/disc/data1/model/DeepSeek-V3.2-10Layer-mtp"
    
    extract_deepseek_layers(SOURCE, TARGET, num_layers=10, with_mtp=True)