import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from transformers import AutoTokenizer

def main():
    # 使用 config_mamba.py 中的默认配置
    config = MambaConfig()

    # 构建 Mamba 语言模型
    model = MambaLMHeadModel(config=config).to("cuda")  # 将模型移动到 GPU

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # 输入自然语言
    input_text = "What is this?"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")  # 将输入移动到 GPU

    # 推理
    with torch.no_grad():
        output = model(input_ids)

    # 解码输出
    logits = output.logits
    predicted_ids = torch.argmax(logits, dim=-1)
    output_text = tokenizer.decode(predicted_ids[0])

    print("输入:", input_text)
    print("输出:", output_text)

if __name__ == "__main__":
    main()