from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


# model_path = '/mnt/public/open_source_model/Qwen1.5-0.5B'
model_path = '/root/zhangfengzhao/listmodel/Llama-2-7b-hf'
quant_path = '/root/zhangfengzhao/test_algorithm/llama2_7b_awq_lwc_int4_g64_lr1e-2'
quant_config = { "zero_point": True, "q_group_size": 64, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(quant_config)
# Quantize
model.quantize(tokenizer, quant_config=quant_config, disable_lwc=False)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')