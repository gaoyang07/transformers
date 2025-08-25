import os
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.moss_ttsd.processing_moss_ttsd import MossTTSDProcessor


device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "/home/gary/ckpts/MOSS-TTSD-v0.5/models--fnlp--MOSS-TTSD-v0.5/snapshots/19aa13feb2d5d63b14d865c3cfd31181b9f0d57a"

processor = MossTTSDProcessor.from_pretrained(
    model_path,
    audio_tokenizer_path="/home/gary/.cache/huggingface/hub/models--fnlp--XY_Tokenizer_TTSD_V0_hf/snapshots/7b9ea0694be4444e9a458eacd12d3589e3cdbf9b",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()

data = [
    {
        "text": "人工智能浪潮正在席卷全球，给我们带来深刻变化",
        "system_prompt": "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text.",
    },
    {
        "text": "人工智能浪潮正在席卷全球，给我们带来深刻变化",
        "system_prompt": "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text.",
        "prompt_text": "周一到周五，每天早晨七点半到九点半的直播片段，言下之意呢就是废话有点儿多，大家也别嫌弃，因为这都是直播间最真实的状态",
        "prompt_audio": "/home/gary/code/transformers/MOSS-TTSD/examples/zh_spk1_moon.wav",
    },
    {
        "text": "[S1]你听说了吗，人工智能现在变得非常厉害！[S2]是啊，我听说现在TTS模型生成的声音已经非常逼真了",
        "system_prompt": "You are a speech synthesizer that generates natural, realistic, and human-like conversational audio from dialogue text.",
        "prompt_audio_speaker1": "/home/gary/code/transformers/MOSS-TTSD/examples/zh_spk1_moon.wav",
        "prompt_text_speaker1": "周一到周五，每天早晨七点半到九点半的直播片段，言下之意呢就是废话有点儿多，大家也别嫌弃，因为这都是直播间最真实的状态",
        "prompt_audio_speaker2": "/home/gary/code/transformers/MOSS-TTSD/examples/zh_spk2_moon.wav", 
        "prompt_text_speaker2": "如果大家想听到更丰富、更及时的直播内容，记得在周一到周五准时进入直播间，和大家一起，畅聊新消费、新科技、新趋势"
    }
]

inputs = processor(data)

token_ids = model.generate(
    input_ids=inputs["input_ids"].to(device), 
    attention_mask=inputs["attention_mask"].to(device), 
    tokenizer=tokenizer,
    do_sample=True,
    temperature=0.7,
    top_p=0.8
)

text, audios = processor.batch_decode(token_ids)

SAVE_DIR = "output3"

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)
for i, data in enumerate(audios):
    for j, fragment in enumerate(data):
        print(f"Saving {i}_{j}.wav")
        torchaudio.save(f"{SAVE_DIR}/new_{i}_{j}.wav", fragment.cpu(), 24000)
