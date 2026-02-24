from diffusers import StableDiffusionPipeline
from typing import Callable, List, Optional, Union
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import matplotlib.pyplot as plt
from utils.inv_methods import Inversion
import torch.nn.functional as F
import numpy as np
import pickle

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
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
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
        dist_losses=[]
        latents_dict={}
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                #losses.append(torch.nn.functional.mse_loss(noise_pred_uncond, noise_pred_text).item())
                #print(t, torch.nn.functional.mse_loss(noise_pred_uncond, noise_pred_text).item())
                #dist_losses.append(F.mse_loss(noise_pred_uncond, noise_pred_text).item())
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                latents_dict[t.item()]={'timestep':t, 'latent':latents, 'dist':F.mse_loss(noise_pred_uncond, noise_pred_text).item()}

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
        #return image, latents, dist_losses
        return image, latents_dict

class analysis_Inversion(Inversion):
    @torch.no_grad()
    def loop(self, latent, gen_dist, t, guidance_scale):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        latent = latent.clone().detach()
        afpi_latent = [latent]
        afpi_conv_list, aidi_conv_list, fpi_conv_list = [], [], []

        latent_input = torch.cat([latent] * 2)

        noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
        noise_uncond, noise_cond = noise_pred.chunk(2)
        guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        latent_ztm1 = latent
        latent = self.next_step(guided_noise, t, latent_ztm1)

        afpi_conv, afpi_dist = self.afpi_step(latent, latent_ztm1, t, guidance_scale)
        aidi_conv, aidi_dist = self.aidi_step(latent, latent_ztm1, t, guidance_scale)
        fpi_conv, fpi_dist = self.fpi_step(latent, latent_ztm1, t, guidance_scale)
        #with open(f'conv_analysis/{t}.pkl', 'wb') as f:
        #    pickle.dump({'afpi_conv': afpi_conv, 'afpi_dist': afpi_dist, 'aidi_conv': aidi_conv, 'aidi_dist': aidi_dist, 'fpi_conv': fpi_conv, 'fpi_dist': fpi_dist, 'gen_dist': gen_dist}, f)
        self.plot_conv(afpi_conv, aidi_conv, fpi_conv, t.item())
        self.plot_dist(afpi_dist, aidi_dist, fpi_dist, gen_dist, t.item())
        return 1

    def plot_conv(self, l1, l2, l3, timestep):
        x1 = np.arange(len(l1))
        x2 = np.arange(len(l2))
        x3 = np.arange(len(l3))
        plt.figure(figsize=(12, 7))
        plt.plot(x1, l1, 'o-', label='afpi', markersize=4)
        plt.plot(x2, l2, 's-', label='aidi', markersize=4)
        plt.plot(x3, l3, '^-', label='fpi', alpha=0.8, lw=1.5, markersize=4)
        plt.yscale('log')
        plt.title('Fixed-point Convergence')
        plt.xlabel('# of Iteration')
        plt.ylabel('L2 Distance')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(f'conv_analysis/21/conv{timestep}.png')
        plt.close()

    def plot_dist(self, l1, l2, l3, gen_dist, timestep):
        x1 = np.arange(len(l1))
        x2 = np.arange(len(l2))
        x3 = np.arange(len(l3))
        plt.figure(figsize=(12, 7))
        plt.plot(x1, l1, 'o-', label='afpi', markersize=4)
        plt.plot(x2, l2, 's-', label='aidi', markersize=4)
        plt.plot(x3, l3, '^-', label='fpi', alpha=0.8, lw=1.5, markersize=4)
        plt.axhline(y=gen_dist, color='r', linestyle='--', linewidth=2, label=f'gen: {gen_dist:.2e}')
        plt.yscale('log')
        plt.title('Uncond-cond Prediction Divergence')
        plt.xlabel('# of Iteration')
        plt.ylabel('L2 Distance')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig(f'conv_analysis/21/dist{timestep}.png')
        plt.close()

    def invert(self, image_latent, gen_dist, t, prompt: str, guidance_scale):
        self.init_prompt(prompt)
        _ = self.loop(image_latent, gen_dist, t, guidance_scale)
        return 1

    def fpi_step(self, init_latent, latent_ztm1, t, guidance_scale):
        optimal_latent = init_latent.clone().detach()
        prev_latent = init_latent.clone().detach()
        
        loss_prev = 1.0
        conv_list, dist_list = [], []
        for rid in range(self.opt_round):
            latent_input = torch.cat([optimal_latent] * 2)
            noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            updated_latent = self.next_step(guided_noise, t, latent_ztm1)
            loss = F.mse_loss(updated_latent, optimal_latent).item()
            conv_list.append(loss)
            #print(loss, F.mse_loss(noise_uncond, noise_cond).item())
            dist_list.append(F.mse_loss(noise_uncond, noise_cond).item())
            if loss < self.threshold:
                break
            #if self.conv_check and loss > loss_prev:
            #    break
            optimal_latent = updated_latent
            loss_prev = loss
        return conv_list, dist_list

    def aidi_step(self, init_latent, latent_ztm1, t, guidance_scale):
        optimal_latent = init_latent.clone().detach()
        prev_latent = init_latent.clone().detach()
        
        loss_prev = 1.0
        conv_list, dist_list = [], []
        for rid in range(self.opt_round):
            latent_input = torch.cat([optimal_latent] * 2)
            noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            updated_latent = self.next_step(guided_noise, t, latent_ztm1)
            loss = F.mse_loss(updated_latent, optimal_latent).item()
            conv_list.append(loss)
            #print(loss, F.mse_loss(noise_uncond, noise_cond).item())
            dist_list.append(F.mse_loss(noise_uncond, noise_cond).item())
            if loss < self.threshold:
                break
            #if self.conv_check and loss > loss_prev:
            #    break
            optimal_latent = 0.5 * optimal_latent + 0.5 * updated_latent
            loss_prev = loss
        return conv_list, dist_list

    def afpi_step(self, init_latent, latent_ztm1, t, guidance_scale):
        optimal_latent = init_latent.clone().detach()
        prev_latent = init_latent.clone().detach()
        
        alpha = 1.0
        loss_prev = 1.0

        fp_th = self.fp_th
        conv_list, dist_list = [], []
        for rid in range(self.opt_round):
            latent_input = torch.cat([optimal_latent] * 2)
            noise_pred = self.get_noise_pred_single(latent_input, t, self.context)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            guided_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            updated_latent = self.next_step(guided_noise, t, latent_ztm1)
            loss = F.mse_loss(updated_latent, optimal_latent).item()
            conv_list.append(loss)
            dist_list.append(F.mse_loss(noise_uncond, noise_cond).item())
            #print(loss)
            if loss < self.threshold:
                break
            if loss > loss_prev:
                #if alpha == 0.5:
                #   break
                #alpha = 0.5 if alpha==1.0 else 1.0
                alpha = 0.5
            optimal_latent = (1 - alpha) * optimal_latent + alpha * updated_latent
            loss_prev = loss
        return conv_list, dist_list
