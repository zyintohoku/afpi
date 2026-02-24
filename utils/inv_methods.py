import time

# %%
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import shutil
from torch.optim.adam import Adam
from PIL import Image
import os

import torch.nn.functional as F
from datetime import datetime

import matplotlib.pyplot as plt
import os
from utils.exact_inversion_scheduler import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Inversion:
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def loop(self, latent, guidance_scale):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        latent = latent.clone().detach()
        all_latent = [latent]
        
        timesteps = self.model.scheduler.timesteps
        total_steps = self.num_ddim_steps
        for i in range(total_steps):
            t = timesteps[-i - 1]
            latent_input = torch.cat([latent] * 2)

            noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            latent_ztm1 = latent
            latent = self.next_step(guided_noise, t, latent_ztm1)

            ################ optimization steps #################
            if self.method=='afpi':
                latent = self.afpi_step(latent, latent_ztm1, t, guidance_scale)
            #elif self.method=='afpi2':
            #    latent = self.afpi_step2(latent, latent_ztm1, t, guidance_scale)
            elif self.method=='aidi':
                latent = self.aidi_step(latent, latent_ztm1, t, guidance_scale)
            elif self.method=='fpi':
                latent = self.fpi_step(latent, latent_ztm1, t, guidance_scale)
            elif self.method=='spd':
                latent = self.spd_step(latent, latent_ztm1, t, guidance_scale)
            elif self.method=='exact':
                latent = self.exact_step(latent, latent_ztm1, t, guidance_scale)
            elif self.method=='newton':
                latent = self.newton_step(latent, latent_ztm1, t, guidance_scale)
            elif self.method=='ddim':
                all_latent.append(latent)
                continue

            all_latent.append(latent)
        return all_latent

    def fpi_step(self, init_latent, latent_ztm1, t, guidance_scale):
        optimal_latent = init_latent.clone().detach()
        #prev_latent = init_latent.clone().detach()
        
        loss_prev = 1.0
        #print(t)
        for rid in range(self.opt_round):
            latent_input = torch.cat([optimal_latent] * 2)
            noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            updated_latent = self.next_step(guided_noise, t, latent_ztm1)
            loss = F.mse_loss(updated_latent, optimal_latent).item()
            #print(loss, F.mse_loss(noise_uncond, noise_cond).item())
            if loss < self.threshold:
                break
            #if self.conv_check and loss > loss_prev:
            #    break
            optimal_latent = updated_latent
            loss_prev = loss
        return optimal_latent.detach()

    def aidi_step(self, init_latent, latent_ztm1, t, guidance_scale):
        optimal_latent = init_latent.clone().detach()
        #prev_latent = init_latent.clone().detach()
        
        loss_prev = 1.0
        #print(t)
        for rid in range(self.opt_round):
            latent_input = torch.cat([optimal_latent] * 2)
            noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            updated_latent = self.next_step(guided_noise, t, latent_ztm1)
            loss = F.mse_loss(updated_latent, optimal_latent).item()
            #print(loss, F.mse_loss(noise_uncond, noise_cond).item())
            if loss < self.threshold:
                break
            #if self.conv_check and loss > loss_prev:
            #    break
            optimal_latent = 0.5 * optimal_latent + 0.5 * updated_latent
            loss_prev = loss
        return optimal_latent.detach()

    #def afpi_step(self, init_latent, latent_ztm1, t, guidance_scale):
    #    optimal_latent = init_latent.clone().detach()
    #    #prev_latent = init_latent.clone().detach()
    #    
    #    alpha = 1.0
    #    loss_prev = 1.0

    #    fp_th = self.fp_th
    #    for rid in range(self.opt_round):
    #        latent_input = torch.cat([optimal_latent] * 2)
    #        noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
    #        noise_uncond, noise_cond = noise_pred.chunk(2)
    #        guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    #        updated_latent = self.next_step(guided_noise, t, latent_ztm1)
    #        loss = F.mse_loss(updated_latent, optimal_latent).item()
    #        #print(loss)
    #        if loss < self.threshold:
    #            break
    #        if alpha==0.5 and loss > loss_prev:
    #            break
    #        optimal_latent = (1 - alpha) * optimal_latent + alpha * updated_latent

    #        fp_ratio = loss / loss_prev

    #        if fp_ratio > fp_th:
    #            alpha = max(0.5, round(alpha-0.1, 1))
    #        loss_prev = loss
    #    return optimal_latent.detach()

    def afpi_step(self, init_latent, latent_ztm1, t, guidance_scale):
        optimal_latent = init_latent.clone().detach()
        #prev_latent = init_latent.clone().detach()
        
        alpha = 1.0
        loss_prev = 1.0

        fp_th = self.fp_th
        for rid in range(self.opt_round):
            latent_input = torch.cat([optimal_latent] * 2)
            noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            updated_latent = self.next_step(guided_noise, t, latent_ztm1)
            loss = F.mse_loss(updated_latent, optimal_latent).item()
            #print(loss)
            if loss < self.threshold:
                break
            if loss > loss_prev:
                if alpha == 0.5:
                    break
                alpha = 0.5
            optimal_latent = (1 - alpha) * optimal_latent + alpha * updated_latent
            loss_prev = loss
        return optimal_latent.detach()

    def spd_step(self, init_latent, latent_ztm1, t, guidance_scale):
        optimal_latent = init_latent.clone().detach()
        #prev_latent = init_latent.clone().detach()
        optimal_latent.requires_grad = True
        optimizer = torch.optim.AdamW([optimal_latent], lr=0.001)
        for rid in range(self.opt_round):
            with torch.enable_grad():
                optimizer.zero_grad()
                latent_input = torch.cat([optimal_latent] * 2)
                noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
                noise_uncond, noise_cond = noise_pred.chunk(2)
                guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

                updated_latent = self.next_step(guided_noise, t, latent_ztm1)
                loss = F.mse_loss(updated_latent, optimal_latent)
                loss.backward()
                optimizer.step()
                if loss < self.threshold:
                    break
        optimal_latent.requires_grad = False
        return optimal_latent.detach()

    def exact_step(self, init_latent, latent_ztm1, t, guidance_scale):
        optimal_latent = init_latent.clone().detach()
        original_step_size = 0.5
        step_scheduler = StepScheduler(current_lr=0.5, factor=0.5, patience=20)
        for rid in range(self.opt_round):
            if rid < 20:
                step_size = original_step_size * (rid+1)/20
            latent_input = torch.cat([optimal_latent] * 2)
            noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            pred_latent = self.prev_step(guided_noise, t, optimal_latent)
            loss = F.mse_loss(pred_latent, latent_ztm1, reduction='sum')
            #print(t, loss.item())
            if loss.item() < 1e-3:
                break
            optimal_latent = optimal_latent - step_size * (pred_latent - latent_ztm1)
            step_size = step_scheduler.step(loss)
        return optimal_latent.detach()
    
    def newton_step(self, init_latent, latent_ztm1, t, guidance_scale):
        optimal_latent = init_latent.clone().detach()
        optimal_latent.requires_grad = True
        best_score = torch.inf
        for rid in range(self.opt_round):
            with torch.enable_grad():
                latent_input = torch.cat([optimal_latent] * 2)
                noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
                noise_uncond, noise_cond = noise_pred.chunk(2)
                guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

                updated_latent = self.next_step(guided_noise, t, latent_ztm1)
                f_x = (updated_latent - optimal_latent).abs()
                loss = f_x.sum()
                score = f_x.mean()
                if score < best_score:
                    best_score = score
                    best_latent = updated_latent.detach()
                loss.backward()
                optimal_latent = optimal_latent - (1 / (64 * 64 * 4)) * (loss / optimal_latent.grad)
                optimal_latent.grad = None
                optimal_latent._grad_fn = None
        optimal_latent.requires_grad = False
        return  best_latent
                
    def invert(self, image_latent, prompt: str, guidance_scale):
        self.init_prompt(prompt)
        all_latent = self.loop(image_latent, guidance_scale)
        return all_latent

    def __init__(self, model, K_round=50, num_ddim_steps=50, delta_threshold=5e-12, method='afpi', fp_th=0.7, conv_check=True):
        self.model = model
        self.scheduler = model.scheduler
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(num_ddim_steps)
        self.prompt = None
        self.context = None
        self.opt_round = K_round
        self.num_ddim_steps = num_ddim_steps
        self.threshold = delta_threshold
        self.method = method
        self.fp_th = fp_th
        self.conv_check = conv_check
