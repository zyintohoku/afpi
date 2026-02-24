from diffusers import StableDiffusionPipeline
from typing import Callable, List, Optional, Union
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import matplotlib.pyplot as plt
from utils.inv_methods import Inversion
import torch.nn.functional as F

class MyStableDiffusionPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        cfg_schedule: List[int] = None,
        skip_timesteps: List[str] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        #do_classifier_free_guidance = guidance_scale > 1.0
        do_classifier_free_guidance = True

        # 3. Encode input prompt
        #breakpoint()
        text_embeddings = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        #with self.progress_bar(total=num_inference_steps) as progress_bar:
        dist_list=[]
        inter_latents=[latents]
        cfg_scale = guidance_scale
        if cfg_schedule is not None:
            cfg_schedule.reverse()
        for i, t in enumerate(timesteps):
            if cfg_schedule is not None:
                guidance_scale = cfg_schedule[i]
            #print(guidance_scale, t.item())
    
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                dist_list.append(F.mse_loss(noise_pred_uncond, noise_pred_text).item())
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                inter_latents.append(latents)
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                #progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
        # 8. Post-processing
        #if not output_type == "latent":
        image = self.decode_latents(latents)
        #else:
        #    return latents
        
        # 9. Run safety checker
        #image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings.dtype)
        # 10. Convert to PIL
        #if output_type == "pil":
        image = self.numpy_to_pil(image)

        #if not return_dict:
        #    return (image, has_nsfw_concept)
        #return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
        return image, inter_latents, dist_list

class analysis_Inversion(Inversion):
    @torch.no_grad()
    def invert(self, latent, prompt, guidance_scale, skip_timesteps=None):
        self.init_prompt(prompt)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        latent = latent.clone().detach()
        all_latent = [latent]
        
        timesteps = self.model.scheduler.timesteps
        total_steps = self.num_ddim_steps
        fpi_conv_list, cfg_div_list = [], []
        #cfg_scale = guidance_scale
        fpi_conv_state = True
        cfg_schedule = []
        for i in range(total_steps):
            latent_temp = latent.clone().detach()
            #guidance_scale = cfg_scale
            #guidance_scale = guidance_scale + 1 if guidance_scale < 7 else 7
            t = timesteps[-i - 1]
            if t<60:
                self.threshold = 4e-13
            else:
                self.threshold = 2e-14
            while guidance_scale >= 0:

                #print(guidance_scale, t.item())
                latent_input = torch.cat([latent] * 2)

                noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
                noise_uncond, noise_cond = noise_pred.chunk(2)
                guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

                latent_ztm1 = latent
                latent = self.next_step(guided_noise, t, latent_ztm1)

                ################ optimization steps #################
                if self.method == 'ddim':
                    all_latent.append(latent)
                    continue
                if self.method == 'fpi':
                    latent, dist_loss = self.fpi_step(latent, latent_ztm1, t, guidance_scale)
                elif self.method == 'afpi':
                    latent, fpi_conv, cfg_div, fpi_conv_state = self.afpi_step(latent, latent_ztm1, t, guidance_scale)
                elif self.method == 'aidi':
                    latent, fpi_conv, cfg_div, fpi_conv_state = self.aidi_step(latent, latent_ztm1, t, guidance_scale)

                if fpi_conv_state == False:
                    if guidance_scale == 0:
                        all_latent.append(latent)
                        fpi_conv_list.append(fpi_conv)
                        cfg_div_list.append(cfg_div)
                        cfg_schedule.append(guidance_scale)
                        break
                    guidance_scale -= 1
                    latent = latent_temp.clone().detach()
                else:
                    all_latent.append(latent)
                    fpi_conv_list.append(fpi_conv)
                    cfg_div_list.append(cfg_div)
                    cfg_schedule.append(guidance_scale)
                    break
                #dist_losses.append(dist_loss)
        return all_latent, fpi_conv_list, cfg_div_list, cfg_schedule

    #def invert(self, image_latent, prompt: str, guidance_scale):
    #    self.init_prompt(prompt)
    #    all_latent, fpi_conv_list, cfg_div_list = self.loop(image_latent, guidance_scale)
    #    return all_latent, fpi_conv_list, cfg_div_list

    def fpi_step(self, init_latent, latent_ztm1, t, guidance_scale):
        optimal_latent = init_latent.clone().detach()
        prev_latent = init_latent.clone().detach()
        
        loss_prev = 1.0
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
            if self.conv_check and loss > loss_prev:
                break
            optimal_latent = updated_latent
            loss_prev = loss
        return optimal_latent.detach(), F.mse_loss(noise_uncond, noise_cond).item()

    def aidi_step(self, init_latent, latent_ztm1, t, guidance_scale):
        optimal_latent = init_latent.clone().detach()
        prev_latent = init_latent.clone().detach()
        
        loss_prev = 1.0
        for rid in range(self.opt_round):
            latent_input = torch.cat([optimal_latent] * 2)
            noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            updated_latent = self.next_step(guided_noise, t, latent_ztm1)
            loss = F.mse_loss(updated_latent, optimal_latent).item()
            print(loss)
            #print(loss, F.mse_loss(noise_uncond, noise_cond).item())
            if loss < self.threshold:
                fpi_conv_state = True
                break
            #if self.conv_check and loss > loss_prev:
            #    break
            if loss > loss_prev:
                fpi_conv_state = False
                break
            optimal_latent = 0.5 * optimal_latent + 0.5 * updated_latent
            loss_prev = loss
        #return optimal_latent.detach(), F.mse_loss(noise_uncond, noise_cond).item()
        return optimal_latent.detach(), loss, F.mse_loss(noise_uncond, noise_cond).item(), fpi_conv_state

    def afpi_step(self, init_latent, latent_ztm1, t, guidance_scale):
        optimal_latent = init_latent.clone().detach()
        prev_latent = init_latent.clone().detach()
        
        alpha = 1.0
        loss_prev = 1.0

        fp_th = self.fp_th
        fpi_conv_state = False
        for rid in range(self.opt_round):
            latent_input = torch.cat([optimal_latent] * 2)
            noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            updated_latent = self.next_step(guided_noise, t, latent_ztm1)
            loss = F.mse_loss(updated_latent, optimal_latent).item()
            #print(loss)
            if loss < self.threshold:
                fpi_conv_state = True
                break
            if loss > loss_prev * 0.99:
                if alpha == 0.5:
                    fpi_conv_state = False
                    break
                alpha = 0.5
            optimal_latent = (1 - alpha) * optimal_latent + alpha * updated_latent
            loss_prev = loss
        return optimal_latent.detach(), loss, F.mse_loss(noise_uncond, noise_cond).item(), fpi_conv_state

    def __init__(self, model, K_round=50, num_ddim_steps=50, delta_threshold=5e-12, afpi=True, fp_th=0.7, conv_check=True):
        self.model = model
        self.scheduler = model.scheduler
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(num_ddim_steps)
        self.prompt = None
        self.context = None
        self.opt_round = K_round
        self.num_ddim_steps = num_ddim_steps
        self.threshold = delta_threshold
        self.afpi = afpi
        self.fp_th = fp_th
        self.conv_check = conv_check
        self.method = 'afpi'
