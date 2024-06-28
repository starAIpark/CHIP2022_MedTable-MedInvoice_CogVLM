---
license: mit
---
# CHIP2022 医疗清单发票OCR要素提取任务 解决方案

使用CogVLM-V2在CHIP2022医疗清单发票OCR要素提取任务上进行lora微调的infer代码


## 权重下载
Huggingface：https://huggingface.co/starAIpark/CHIP2022_MedTable-MedInvoice_CogVLM/tree/main


## Infer 代码
将代码中的`LORA_PATH`改为下载的文件

若需自己的图片上做infer则应更改下述代码中的 `img_path` 和 `prompt`

```python
import os
import cv2
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torchvision import transforms 

import math
import numpy as np
from tqdm import tqdm
from easymodel.model import MODEL, build_model
from easymodel.loss import build_loss

from transformers import AutoModelForCausalLM, AutoModel, GPTQConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class LoadCogVLM(object):
    def __init__(
        self, imgpath_key, img_key,
        image_size, patch_size, q_key,
        vision_text
    ):
        self.img_key = img_key
        self.imgpath_key = imgpath_key
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_text = vision_text
        self.q_key = q_key


    def __call__(self, data):
        img_path = data[self.imgpath_key]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        vision_token_num = (self.image_size // self.patch_size // 2) * (self.image_size // self.patch_size // 2) + 2
        question = '<|begin_of_text|>' + self.vision_text * vision_token_num
        data[self.q_key] = question
        data[self.img_key] = image
        return data

class VQA(object):
    def __init__(
        self, tokenizer_type,
        conv_key, max_input_len, eos_text, pad_text,
        inputids_key, lossmask_key, only_question,template_type = None, **kwargs
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type, trust_remote_code = True)
        self.conv_key = conv_key
        self.max_input_len = max_input_len
        self.eos_text = eos_text
        self.pad_text = pad_text
        self.inputids_key = inputids_key
        self.lossmask_key = lossmask_key
        self.only_question = only_question
        self.template_type = template_type
        if self.template_type == 'CogVLM':
            self.pad_id = self.tokenizer(self.pad_text, return_tensors='pt')['input_ids'][0][1]
        else:
            self.pad_id = self.tokenizer(self.pad_text, return_tensors='pt')['input_ids'][0][0]
        self.kwargs = kwargs

    def __call__(self, data):
        conv_list = data[self.conv_key]

        # 这里需要考虑多轮对话
        if self.template_type is None:
            question_text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>'
            answer_text = ''
        elif self.template_type == 'CogVLM':
            question_text = data['question']
            answer_text = ''

            
        conv_cnts = len(conv_list) / 2
        selected_cnts = random.randint(1, conv_cnts)
        for k in range(selected_cnts):
            if self.template_type is None:
                q_str = conv_list[k*2]['value']
                a_str = conv_list[k*2 + 1]['value']
                question_text += f'<|im_start|>user\n{q_str}<|im_end|>'
                if k == selected_cnts-1:
                    question_text += '<|im_start|>assistant\n'
                    answer_text = f'{a_str}<|im_end|>'
                else:
                    question_text += f'<|im_start|>assistant\n{a_str}<|im_end|>'
            elif self.template_type == 'CogVLM':
                q_str = conv_list[k*2]['value']
                a_str = conv_list[k*2 + 1]['value']
                question_text += f'Question: {q_str}'
                if k == selected_cnts-1:
                    question_text += 'Answer: '
                    answer_text = f'{a_str}' + self.eos_text
                else:
                    question_text += f'Answer: {a_str}'

        prompt_ids = self.tokenizer(question_text, return_tensors='pt')['input_ids'][0]
        if self.template_type == 'CogVLM':
            prompt_ids = prompt_ids[1:]
        
        if self.only_question:
            data[self.inputids_key] = prompt_ids
            if self.template_type == 'CogVLM':
                token_type_ids = torch.zeros(len(prompt_ids))
                ind = prompt_ids == self.tokenizer(self.kwargs['vision_text'], return_tensors = 'pt')['input_ids'][0][1]
                token_type_ids[ind] = 1
                data[self.kwargs['token_type_key']] = token_type_ids
            return data
        if self.template_type == 'CogVLM':
            answer_ids = self.tokenizer(answer_text, return_tensors = 'pt')['input_ids'][0][1:]
        else:
            answer_ids = self.tokenizer(answer_text, return_tensors = 'pt')['input_ids'][0]
        input_ids = torch.cat((prompt_ids, answer_ids))
        loss_mask = torch.cat((torch.zeros(len(prompt_ids)), torch.ones(len(answer_ids))))
        
        if self.template_type == 'CogVLM':
            token_type_ids = torch.zeros(len(input_ids))
            ind = input_ids == self.tokenizer(self.kwargs['vision_text'], return_tensors = 'pt')['input_ids'][0][1]
            token_type_ids[ind] = 1

        if len(input_ids) >= self.max_input_len:
            input_ids = input_ids[:self.max_input_len]
            loss_mask = loss_mask[:self.max_input_len]
            if self.template_type == 'CogVLM':
                token_type_ids = token_type_ids[:self.max_input_len]
        else:
            padding_len = self.max_input_len - len(input_ids)
            input_pad_id = torch.ones(padding_len, dtype = input_ids.dtype) * self.pad_id
            input_ids = torch.cat((input_ids, input_pad_id))

            lossmask_pad_id = torch.zeros(padding_len)
            loss_mask = torch.cat((loss_mask, lossmask_pad_id), axis = 0)

            if self.template_type == 'CogVLM':
                token_type_pad_ids = torch.zeros(padding_len)
                token_type_ids = torch.cat((token_type_ids, token_type_pad_ids), axis = 0)
        
        data[self.inputids_key] = input_ids
        data[self.lossmask_key] = loss_mask
        if self.template_type == 'CogVLM':
            data[self.kwargs['token_type_key']] = token_type_ids
        return data        

def load_lora(model, weight_path):
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict, strict=False)
    return model

class CogVLMSFT(nn.Module):
    def __init__(
        self, model_path, lora_cfg, logger = None
    ):
        super().__init__()
        self.GPU_ID = int(os.environ.get('LOCAL_RANK') or 0)
        self.device = f"cuda:{self.GPU_ID}"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype = torch.bfloat16,
            low_cpu_mem_usage = True, trust_remote_code = True,
            device_map = self.device
        )

        if lora_cfg is not None:
            lora_config = LoraConfig(**lora_cfg)
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self.loss = build_loss(dict(type = 'CrossEntropyLoss', loss_weight = 1.0))
        self.logger = logger


    # compute train losses
    def forward(self, data):
        input_ids = data['input_ids']
        loss_mask = data['loss_mask']
        labels = input_ids[:,1:]
        loss_mask = loss_mask[:,1:]

        pred_logits = self.model(
            input_ids = data['input_ids'][:,:-1],
            images = data['imgs'][:, None, ...],
            token_type_ids = data['token_type_ids'][:,:-1]
        ).logits

        loss = self.loss(pred = pred_logits, target = labels, mask = loss_mask)
        output = dict()
        output['loss'] = loss
        output['pred'] = pred_logits
        output['label'] = labels
        output['loss_mask'] = loss_mask

        return output

    def gen(self, data):
        output = self.model.generate(
            input_ids = data['input_ids'][None, ...].to('cuda'),
            token_type_ids = data['token_type_ids'][None, ...].to('cuda'),
            images = data['imgs'][None, None, ...].to('cuda').to(torch.bfloat16),
            max_new_tokens = 2048,
            do_sample = True
        )
        return output[0][len(data['input_ids']):]

# 加载模型
MODEL_PATH = 'THUDM/cogvlm2-llama3-chinese-chat-19B'
LORA_PATH = 'cogvlm_chip2022_lora.bin'
model_cfg = dict(
    model_path = MODEL_PATH,
    lora_cfg = dict(
        r = 128, target_modules = [
            'attention.query_key_value',
            'self_attn.language_expert_query_key_value',
            'self_attn.vision_expert_query_key_value',
        ],
        lora_alpha = 256, lora_dropout = 0.05,
        modules_to_save = None
    )
)
model = CogVLMSFT(**model_cfg)
model = load_lora(model, LORA_PATH)

# 加载pipeline
load_cfg = dict(imgpath_key = 'img_path', 
         image_size = 1344, patch_size = 14, img_key = 'imgs',
         vision_text = '<|reserved_special_token_0|>', q_key = 'question')
vqa_cfg = dict(type = 'VQA', tokenizer_type = MODEL_PATH,
         conv_key = 'conversations', max_input_len = 2560 + 512,
         eos_text = '<|end_of_text|>', pad_text = '<|reserved_special_token_0|>',
         inputids_key = 'input_ids', lossmask_key = 'loss_mask',
         only_question = True, template_type = 'CogVLM',
         vision_text = '<|reserved_special_token_0|>', token_type_key = 'token_type_ids')
pipeline_load = LoadCogVLM(**load_cfg)
pipeline_vqa = VQA(**vqa_cfg)



img_path = 'info_extra_demo.jpg'
prompt = '请给出图中票据代码的值，若图中不存在票据代码请输出\'无\''

conv =  [
    dict(value = prompt),
    dict(value = '')
]
data = dict(
    img_path = img_path,
    conversations = conv
)
data = pipeline_load(data)
data = pipeline_vqa(data)

print('img_path:\t', img_path)
print('prompt:\t', prompt)
with torch.no_grad():
    answer = model.gen(data)
    answer = pipeline_vqa.tokenizer.decode(answer)
    answer = answer.replace('<|end_of_text|>', '')


print('model output:\t', answer)


```