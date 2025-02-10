import os
## visable gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
from PIL import Image
import os
import json
import pickle
from tqdm import tqdm

from modelscope import AutoModelForCausalLM, AutoTokenizer
# from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

# MODEL_PATH = "/data3/lisibo/.cache/modelscope/hub/ZhipuAI/cogvlm2-llama3-chinese-chat-19B-int4"
MODEL_PATH= "ZhipuAI/cogvlm2-llama3-chinese-chat-19B-int4"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
    0] >= 8 else torch.float16

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).eval()



img_dirs = []
valid_dirs = []
valid_pois = []
dirs = os.listdir("../../datasets/source/")
dirs = [d for d in dirs if d.endswith(".json")]
dirs = [d for d in dirs if d.startswith("yinchuan")]

print("total dirs:", len(dirs))
for p in dirs:
    if p.endswith(".json"):
        js = json.load(open("../../datasets/source/" + p))
        if len(js["labels"]) == 1:
            valid_dirs.append(p.split(".")[0] + ".png")
            valid_pois.append(js["poi_names"])

print("valid dirs:", len(valid_dirs))

res = {}


text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

for img_path in tqdm(valid_dirs):
    print(img_path)
    save_path = "../../datasets/res_Cog2/" + img_path.split(".")[0] + ".pkl"
    if os.path.exists(save_path):
        continue
    history = []
    image_path = "../../datasets/full_dataset/" + img_path
    assert image_path is not None
    names = []
    if len(valid_pois[valid_dirs.index(img_path)]) >= 20:
        names = valid_pois[valid_dirs.index(img_path)][:20]
    else:
        names = valid_pois[valid_dirs.index(img_path)]
    names = "，".join(names)
    image = Image.open(image_path).convert('RGB')

    query = "请描述这幅遥感图片，并推测其用地类型与其中的人类活动。以下是区域中一些地点的名称，" + names
    # "Please describe this remote sensing image and speculate on its land use type and human activities. Here are the names of some places in the area: " + names


    input_by_model = model.build_conversation_input_ids(
        tokenizer,
        query=query,
        history=history,
        images=[image],
        template_version='chat'
    )

    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
    }
    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": 128002,  
        "output_hidden_states": True,
        "return_dict_in_generate": True,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        # print(outputs.shape)
        sequence = outputs.sequences
        # print(sequence.shape)
        hidden = outputs.hidden_states
        seq_len = len(hidden)
        hidden1 = hidden[0][0][:,-1,:]
        hidden2 = hidden[0][-1][:,-1,:]

        hidden1 = hidden1.detach().float().cpu().numpy()
        hidden2 = hidden2.detach().float().cpu().numpy()

        outputs = sequence[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        response = response.split("<|end_of_text|>")[0]

    with open("../../datasets/res_Cog2/result.txt", "a") as f:
        f.write(img_path + "\n")
        f.write(query + "\n")
        f.write(response + "\n")
        f.write("======================\n")

    pickle.dump((response, hidden1, hidden2), open("../../datasets/res_Cog2/" + img_path.split(".")[0] + ".pkl", "wb"))
    # # print("====================================")

    # history.append((query, response))