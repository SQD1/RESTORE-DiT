from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline
import numpy as np
import PIL.Image
import torch
from typing import List, Optional, Tuple, Union
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

class SeqDiffusionPipeline(DiffusionPipeline):
    """
    Args:
        model: to denoise the image sequence.
        scheduler: A Diffusion scheduler to be used to denoise the image sequence.
    """
    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    def __call__(
            self,
            image: PipelineImageInput = None,  # [B, T, C, H, W]
            mask: PipelineImageInput = None,   # [B, T, 1 ,H, W]
            batch_positions: Optional[torch.Tensor] = None,
            cond: Optional[torch.Tensor] = None,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            eta: float = 0.0,
            num_inference_steps: int = 1,
            use_clipped_model_output: Optional[bool] = None,
            output_type: Optional[str] = "numpy",  # "pil" or "numpy"
            return_dict: bool = True,
            seq_length: int = 15
    ):
        # Sample gaussian noise to begin loop
        noise = torch.randn(image.shape, generator=generator).to(image.device)  # [B, T, C, H, W]
        latents = noise

        self.scheduler.set_timesteps(num_inference_steps, device=image.device) # device

        for i, t in self.progress_bar(enumerate(self.scheduler.timesteps)):
            # 1. predict noise model_output
            init_image_proper = image
            # t = torch.tensor([t])
            t = torch.tensor([t], device=image.device).long()
            unet_input = latents * mask + (1. - mask) * image

            with torch.no_grad():
                if cond is not None and batch_positions is not None:
                    model_output = self.model(unet_input, t, batch_positions, cond)
                elif cond is not None and batch_positions is None:
                    model_output = self.model(unet_input, t, date=None, cond=cond)
                else:
                    model_output = self.model(unet_input, t)
            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            # for DDIM
            # latents = self.scheduler.step(
            #     model_output, t[0], latents, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            # ).prev_sample # t.cpu()

            # for DPM-solver
            latents = self.scheduler.step(
                model_output, t[0], latents, generator=generator
            ).prev_sample  # t.cpu()

            if i < len(self.scheduler.timesteps) - 1:
                noise_timestep = self.scheduler.timesteps[i + 1]
                init_image_proper = self.scheduler.add_noise(
                    init_image_proper, noise, torch.tensor([noise_timestep])
                )

            latents = (1 - mask) * init_image_proper + mask * latents


        # output_image = noise * mask + (1. - mask) * image
        output_image = (latents / 2 + 0.5).clamp(0, 1)

        if output_type == "tensor":
            return output_image

        output_image = output_image.cpu().detach().numpy()  # [B, T, C, H, W]
        if output_type == "pil":
            output_image = self.numpy_to_pil(output_image)

        if not return_dict:
            return (output_image,)

        return ImagePipelineOutput(images=output_image)



