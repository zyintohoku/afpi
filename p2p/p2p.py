from typing import Optional, Union, Tuple, List, Callable, Dict
import os
import torch
torch.backends.cudnn.allow_tf32 = False # For reproducibility across Titan, A6000, and A8000
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
from PIL import Image
import json
import argparse

# ── Constants ──────────────────────────────────────────────────
SEED = 0
NUM_DIFFUSION_STEPS = 50
MAX_NUM_WORDS = 77
LOW_RESOURCE = False
MODEL_ID = "CompVis/stable-diffusion-v1-4"
CROSS_REPLACE_STEPS = 0.8
SELF_REPLACE_STEPS = 0.6

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], tokenizer, threshold=.3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to('cuda')
        self.threshold = threshold

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend], tokenizer=None):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to('cuda')
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend
        self.tokenizer = tokenizer

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, tokenizer=None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer=tokenizer)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, self.tokenizer).to('cuda')

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, tokenizer=None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer=tokenizer)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, self.tokenizer)
        self.mapper, alphas = self.mapper.to('cuda'), alphas.to('cuda')
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


def load_model():
    torch.manual_seed(SEED)
    ldm_stable = StableDiffusionPipeline.from_pretrained(MODEL_ID).to('cuda')
    ldm_stable.scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        clip_sample=False, set_alpha_to_one=False, steps_offset=1,
    )
    return ldm_stable


def load_editing_instructions():
    mapping_path = os.path.join(PROJECT_ROOT, "PIE_bench", "mapping_file.json")
    with open(mapping_path, "r") as f:
        return json.load(f)


def p2p_editing(inv_dir, ldm_stable, editing_instruction):
    tokenizer = ldm_stable.tokenizer
    latents = torch.load(os.path.join(inv_dir, 'inv_latents.pt'), weights_only=True)

    cfg_schedules_path = os.path.join(inv_dir, 'cfg_schedules.pt')
    if os.path.exists(cfg_schedules_path):
        cfg_schedules = torch.load(cfg_schedules_path, weights_only=True)
        print(f"[p2p] Loaded per-step CFG schedules from {cfg_schedules_path}")
    else:
        cfg_schedules = None
        print(f"[p2p] cfg_schedules.pt not found in {inv_dir}, using constant guidance_scale=7.5")

    os.makedirs(inv_dir, exist_ok=True)

    for i, (_, item) in enumerate(editing_instruction.items()):
        latent = latents[i]
        cfg_schedule = cfg_schedules[i] if cfg_schedules is not None else None
        prompt_src = item["original_prompt"].replace("[", "").replace("]", "")
        prompt_tgt = item["editing_prompt"].replace("[", "").replace("]", "")
        blended_word = item["blended_word"]
        prompts = [prompt_src, prompt_tgt]

        if blended_word != '':
            s1, s2 = blended_word.split(" ")
            lb = LocalBlend(prompts, (s1, s2), tokenizer)
        else:
            lb = None

        if len(prompt_src.split()) == len(prompt_tgt.split()):
            controller = AttentionReplace(
                prompts, NUM_DIFFUSION_STEPS,
                cross_replace_steps=CROSS_REPLACE_STEPS,
                self_replace_steps=SELF_REPLACE_STEPS,
                local_blend=lb, tokenizer=tokenizer,
            )
        else:
            controller = AttentionRefine(
                prompts, NUM_DIFFUSION_STEPS,
                cross_replace_steps=CROSS_REPLACE_STEPS,
                self_replace_steps=SELF_REPLACE_STEPS,
                local_blend=lb, tokenizer=tokenizer,
            )

        images, x_t = ptp_utils.text2image_ldm_stable(
            ldm_stable, prompts, controller,
            latent=latent, cfg_schedule=cfg_schedule,
        )
        Image.fromarray(images[0]).save(os.path.join(inv_dir, f'{i}ori.png'))
        Image.fromarray(images[1]).save(os.path.join(inv_dir, f'{i}edi.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inv_dir", type=str, default="outputs/cfg7/afpi",
                        help="Path to inversion output directory (relative to project root)")
    args = parser.parse_args()

    # Resolve relative paths against project root
    inv_dir = args.inv_dir
    if not os.path.isabs(inv_dir):
        inv_dir = os.path.join(PROJECT_ROOT, inv_dir)

    ldm_stable = load_model()
    editing_instruction = load_editing_instructions()
    p2p_editing(inv_dir, ldm_stable, editing_instruction)
