import os
from cog import BasePredictor, Input, Path
from typing import List
import sys
sys.path.append('/content/Arc2Face-hf')
os.chdir('/content/Arc2Face-hf')

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)

from arc2face import CLIPTextModelWrapper, project_face_embs

import torch
from insightface.app import FaceAnalysis
from PIL import Image
import numpy as np
import random

MAX_SEED = np.iinfo(np.int32).max

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def generate_image(image_path, num_steps, guidance_scale, seed, num_images, device, pipeline, app, dtype):

    if image_path is None:
        print(f"Cannot find any input face image! Please upload a face image.")
    
    img = np.array(Image.open(image_path))[:,:,::-1]

    # Face detection and ID-embedding extraction
    faces = app.get(img)
    
    if len(faces) == 0:
        print(f"Face detection failed! Please try with another image.")
    
    faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # select largest face (if more than one detected)
    id_emb = torch.tensor(faces['embedding'], dtype=dtype)[None].to(device)
    id_emb = id_emb/torch.norm(id_emb, dim=1, keepdim=True)   # normalize embedding
    id_emb = project_face_embs(pipeline, id_emb)    # pass throught the encoder
                    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print("Start inference...")        
    images = pipeline(
        prompt_embeds=id_emb,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale, 
        num_images_per_prompt=num_images,
        generator=generator
    ).images

    return images

class Predictor(BasePredictor):
    def setup(self) -> None:

        # global variable
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        # Load face detection and recognition package
        self.app = FaceAnalysis(name='antelopev2', root='./', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Load pipeline
        base_model = 'runwayml/stable-diffusion-v1-5'
        encoder = CLIPTextModelWrapper.from_pretrained('models', subfolder="encoder", torch_dtype=self.dtype)
        unet = UNet2DConditionModel.from_pretrained('models', subfolder="arc2face", torch_dtype=self.dtype)
        self.pipeline = StableDiffusionPipeline.from_pretrained(
                base_model,
                text_encoder=encoder,
                unet=unet,
                torch_dtype=self.dtype,
                safety_checker=None
            )
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline = self.pipeline.to(self.device)

    def predict(
        self,
        input_image: Path = Input(description="Input Image"),
        num_steps: int = Input(default=25),
        guidance_scale: float = Input(default=3.0),
        seed: int = Input(default=0),
        num_images: int = Input(default=4),
        randomize_seed: bool = True
    ) -> List[Path]:
        seed = randomize_seed_fn(seed, randomize_seed)
        images = generate_image(input_image, num_steps, guidance_scale, seed, num_images, self.device, self.pipeline, self.app, self.dtype)
        for i, img in enumerate(images):
            img.save(f'/content/{i+1}.png')
        return [Path(f'/content/{i+1}.png') for i in range(num_images)]