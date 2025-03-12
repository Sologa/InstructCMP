import sys
import os
import fire
import torch
from detokenize.detokenizer import detokenize
from transformers import BitsAndBytesConfig
from transformers import(
    AutoTokenizer,
    LlamaForCausalLM,
)

from src.inference_utils import inference
from src.utils import get_template, apply_template, get_dataset
from src.evaluate_utils import generated_output_post_processing, evaluate

def main(
        model_size:str = "13",
        batch_size:int = 10,
        data_name:str = "Google",
        split:str = "test"
):  
    model_name = f"meta-llama/Llama-2-{model_size}b-chat-hf"

    print("Loading Model...")
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = LlamaForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=nf4_config,
            cache_dir="/data/huggingface_models/"
        )
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              use_fast=True,
                                              padding_side="left",
                                              cache_dir="/data/huggingface_models/"
                                              )
    tokenizer.pad_token = tokenizer.eos_token

    template = get_template()

    os.environ["dataset_path"] = f"dataset/{data_name}/{data_name}_{split}.jsonl"
    sources, targets = get_dataset()
    
    if "BNC".lower() in data_name.lower() or "Broadcast".lower() in data_name.lower():
        print("Detokenizing...")
        targets = [detokenize(tgt.split()) for tgt in targets]
        sources = [detokenize(src.split()) for src in sources]

    instances = []
    for src, tgt in zip(sources, targets):
        instances.append(
            {
                "src": src,
                "del_len": len(src.split()) - len(tgt.split())
            }
        )

    prompts = apply_template(instances, template)

    generated_text = inference(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=batch_size,
        max_new_tokens=200,
        do_sample=False,
    )

    post_processed_outputs = generated_output_post_processing(generated_text)
    result = evaluate(targets, sources, post_processed_outputs)

if __name__ == '__main__':
    with torch.no_grad():
        fire.Fire(main)