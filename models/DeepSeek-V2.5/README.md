---
license: other
license_name: deepseek
license_link: https://github.com/deepseek-ai/DeepSeek-V2/blob/main/LICENSE-MODEL
library_name: transformers
---

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V2" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://chat.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20V2-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/deepseek-ai" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank" style="margin: 2px;">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/qr.jpeg?raw=true" target="_blank" style="margin: 2px;">
    <img alt="Wechat" src="https://img.shields.io/badge/WeChat-DeepSeek%20AI-brightgreen?logo=wechat&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/LICENSE-CODE" style="margin: 2px;">
    <img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/LICENSE-MODEL" style="margin: 2px;">
    <img alt="Model License" src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2405.04434"><b>Paper Link</b>👁️</a>
</p>

# DeepSeek-V2.5

## 1. Introduction

DeepSeek-V2.5 is an upgraded version that combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct. The new model integrates the general and coding abilities of the two previous versions.
For model details, please visit [DeepSeek-V2 page](https://github.com/deepseek-ai/DeepSeek-V2) for more information.

DeepSeek-V2.5 better aligns with human preferences and has been optimized in various aspects, including writing and instruction following:

| Metric                 | DeepSeek-V2-0628 | DeepSeek-Coder-V2-0724 | DeepSeek-V2.5 |
|:-----------------------|:-----------------|:-----------------------|:--------------|
| AlpacaEval 2.0          | 46.6             | 44.5                   | 50.5          |
| ArenaHard              | 68.3             | 66.3                   | 76.2          |
| AlignBench             | 7.88             | 7.91                   | 8.04          |
| MT-Bench               | 8.85             | 8.91                   | 9.02          |
| HumanEval python       | 84.5             | 87.2                   | 89            |
| HumanEval Multi        | 73.8             | 74.8                   | 73.8          |
| LiveCodeBench(01-09)   | 36.6             | 39.7                   | 41.8          |
| Aider                  | 69.9             | 72.9                   | 72.2          |
| SWE-verified           | N/A              | 19                     | 16.8          |
| DS-FIM-Eval            | N/A              | 73.2                   | 78.3          |
| DS-Arena-Code          | N/A              | 49.5                   | 63.1          |



## 2. How to run locally

**To utilize DeepSeek-V2.5 in BF16 format for inference, 80GB*8 GPUs are required.**
### Inference with Huggingface's Transformers
You can directly employ [Huggingface's Transformers](https://github.com/huggingface/transformers) for model inference.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "deepseek-ai/DeepSeek-V2.5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# `max_memory` should be set based on your devices
max_memory = {i: "75GB" for i in range(8)}
# `device_map` cannot be set to `auto`
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="sequential", torch_dtype=torch.bfloat16, max_memory=max_memory, attn_implementation="eager")
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

messages = [
    {"role": "user", "content": "Write a piece of quicksort code in C++"}
]
input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
print(result)
```

The complete chat template can be found within `tokenizer_config.json` located in the huggingface model repository.

**Note: The chat template has been updated compared to the previous DeepSeek-V2-Chat version.**

An example of chat template is as belows:

```bash
<｜begin▁of▁sentence｜><｜User｜>{user_message_1}<｜Assistant｜>{assistant_message_1}<｜end▁of▁sentence｜><｜User｜>{user_message_2}<｜Assistant｜>
```

You can also add an optional system message:

```bash
<｜begin▁of▁sentence｜>{system_message}<｜User｜>{user_message_1}<｜Assistant｜>{assistant_message_1}<｜end▁of▁sentence｜><｜User｜>{user_message_2}<｜Assistant｜>
```

### Inference with vLLM (recommended)
To utilize [vLLM](https://github.com/vllm-project/vllm) for model inference, please merge this Pull Request into your vLLM codebase: https://github.com/vllm-project/vllm/pull/4650.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

max_model_len, tp_size = 8192, 8
model_name = "deepseek-ai/DeepSeek-V2.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=tp_size, max_model_len=max_model_len, trust_remote_code=True, enforce_eager=True)
sampling_params = SamplingParams(temperature=0.3, max_tokens=256, stop_token_ids=[tokenizer.eos_token_id])

messages_list = [
    [{"role": "user", "content": "Who are you?"}],
    [{"role": "user", "content": "Translate the following content into Chinese directly: DeepSeek-V2 adopts innovative architectures to guarantee economical training and efficient inference."}],
    [{"role": "user", "content": "Write a piece of quicksort code in C++."}],
]

prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list]

outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

generated_text = [output.outputs[0].text for output in outputs]
print(generated_text)
```

### Function calling

Function calling allows the model to call external tools to enhance its capabilities.

Here is an example:

```python
# Assume that `model` and `tokenizer` are loaded
model.generation_config = GenerationConfig(do_sample=False, max_new_tokens=128, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)

tool_system_prompt = """You are a helpful Assistant.

## Tools

### Function

You have the following functions available:

- `get_current_weather`:
```json
{
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": [
                    "celsius",
                    "fahrenheit"
                ]
            }
        },
        "required": [
            "location"
        ]
    }
}
```"""

tool_call_messages = [{"role": "system", "content": tool_system_prompt}, {"role": "user", "content": "What's the weather like in Tokyo and Paris?"}]
tool_call_inputs = tokenizer.apply_chat_template(tool_call_messages, add_generation_prompt=True, return_tensors="pt")
tool_call_outputs = model.generate(tool_call_inputs.to(model.device))
# Generated text: '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "Tokyo"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n```json\n{"location": "Paris"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>'

# Mock response of calling `get_current_weather`
tool_messages = [{"role": "tool", "content": '{"location": "Tokyo", "temperature": "10", "unit": null}'}, {"role": "tool", "content": '{"location": "Paris", "temperature": "22", "unit": null}'}]
tool_inputs = tokenizer.apply_chat_template(tool_messages, add_generation_prompt=False, return_tensors="pt")[:, 1:]
tool_inputs = torch.cat([tool_call_outputs, tool_inputs.to(model.device)], dim=1)
tool_outputs = model.generate(tool_inputs)
# Generated text: The current weather in Tokyo is 10 degrees, and in Paris, it is 22 degrees.<｜end▁of▁sentence｜>
```

### JSON output

You can use JSON Output Mode to ensure the model generates a valid JSON object. To active this mode, a special instruction should be appended to your system prompt.

```python
# Assume that `model` and `tokenizer` are loaded
model.generation_config = GenerationConfig(do_sample=False, max_new_tokens=128, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)

user_system_prompt = 'The user will provide some exam text. Please parse the "question" and "answer" and output them in JSON format.'
json_system_prompt = f"""{user_system_prompt}

## Response Format

Reply with JSON object ONLY."""

json_messages = [{"role": "system", "content": json_system_prompt}, {"role": "user", "content": "Which is the highest mountain in the world? Mount Everest."}]
json_inputs = tokenizer.apply_chat_template(json_messages, add_generation_prompt=True, return_tensors="pt")
json_outpus = model.generate(json_inputs.to(model.device))
# Generated text: '```json\n{\n  "question": "Which is the highest mountain in the world?",\n  "answer": "Mount Everest."\n}\n```<｜end▁of▁sentence｜>'
```

### FIM completion

In FIM (Fill In the Middle) completion, you can provide a prefix and an optional suffix, and the model will complete the content in between.

```python
# Assume that `model` and `tokenizer` are loaded
model.generation_config = GenerationConfig(do_sample=False, max_new_tokens=128, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)

prefix = """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = []
    right = []
"""

suffix = """
        if arr[i] < pivot:
            left.append(arr[i])
        else:
            right.append(arr[i])
    return quick_sort(left) + [pivot] + quick_sort(right)"""

fim_prompt = f"<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"
fim_inputs = tokenizer(fim_prompt, add_special_tokens=True, return_tensors="pt").input_ids
fim_outputs = model.generate(fim_inputs.to(model.device))
# Generated text: "    for i in range(1, len(arr)):<｜end▁of▁sentence｜>"
```

## 3. License
This code repository is licensed under the MIT License. The use of DeepSeek-V2 Base/Chat models is subject to [the Model License](LICENSE). DeepSeek-V2 series (including Base and Chat) supports commercial use.

## 4. Citation
```
@misc{deepseekv2,
      title={DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model}, 
      author={DeepSeek-AI},
      year={2024},
      eprint={2405.04434},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## 5. Contact
If you have any questions, please raise an issue or contact us at [service@deepseek.com](service@deepseek.com).
