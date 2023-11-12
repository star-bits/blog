# Stable Diffusion web UI

- [stable-diffusion-webui-colab](https://github.com/star-bits/blog/blob/main/stable-diffusion-webui-colab.ipynb): Stable Diffusion web UI on Colab

## Model

- [v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) [*v1-5-pruned-emaonly.ckpt*](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt)
- [v1.5 inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting) [*sd-v1-5-inpainting.ckpt*](https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt)
- [Dreamlike Photoreal](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0) [*dreamlike-photoreal-2.0.safetensors*](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0/resolve/main/dreamlike-photoreal-2.0.safetensors)
- [Realistic Vison](https://civitai.com/models/4201/realistic-vision-v20) [*realisticVisionV51_v51VAE.safetensors*](https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor&size=pruned&fp=fp16)
- [Dreamshaper](https://civitai.com/models/4384/dreamshaper) [*dreamshaper_8.safetensors*](https://civitai.com/api/download/models/128713?type=Model&format=SafeTensor&size=pruned&fp=fp16)
- [PerfectDeliberate](https://civitai.com/models/24350/perfectdeliberate) [*perfectdeliberate_v40.safetensors*](https://civitai.com/api/download/models/86698?type=Model&format=SafeTensor&size=full&fp=fp32)
- [Lyriel](https://civitai.com/models/22922/lyriel) [*lyriel_v16.safetensors*](https://civitai.com/api/download/models/72396)
- [Anything](https://civitai.com/models/9409) [*AnythingV5Ink_ink.safetensors*](https://civitai.com/api/download/models/90854?type=Model&format=SafeTensor&size=full&fp=fp16)

Model size ranges from 2 to 7 GB. *Why do they vary? Don't they share the exact same network?*

- `.ckpt`: TensorFlow checkpoint (model).
- `.pt` and `.pth`: PyTorch model. .pt stores the entire model, while .pth stores just the parameters.
- `.safetensors`: secure format for storing and sharing models.

*TensorFlow and PyTorch model files are interchangeable?*

- 'pruned': unnecessary weights that have small impact on model accuracy are removed
- 'ema-only': weights are Exponential Moving Averages of the last few epochs

## Sampler

Each denoising step is called sampling. There are many different ways to schedule this denoising process. Noise schedule controls the noise level at each sampling step. Sampler’s job is to produce an image with a noise level matching the noise schedule.

- Ancestral sampler adds noise to the image at each sampling step. At each step, it subtracts more noise than it should and adds some random noise back to match the noise schedule. They are maked with 'a' in names. (e.g., Euler a, DPM2 a, DPM++ 2S a, DPM++ 2S a Karras) Images generated with ancestral samplers do not converge at high sampling steps. Convergence means reproducibility.
- Samplers with Karras have smaller noise steps near the end. This improves image quality.
- DDIM and PLMS were shipped with original Stable Diffusion v1, and are generally considered as outdated.

Go for:
- DPM++ 2M Karras (20 – 30 steps) if you care about convergence,
- DPM++ SDE Karras (10-15 steps) if you don't.

## VAE

VAE = Encoder/Decoder. One may use different VAEs other than the default one. Changing VAE may improve the quality of rendered images. The effect is usually tiny. I'd say unnoticeable.

- EMA VAE: [*vae-ft-ema-560000-ema-pruned.ckpt*](https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-560000-ema-pruned.ckpt)
- MSE VAE: [*vae-ft-mse-840000-ema-pruned.ckpt*](https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt)

Settings -> User Interface -> Quicksettings list -> add sd_vae

## CFG scale

Classifier Free Guidance scale is a parameter to control how much the model should respect your prompt.
- 1: mostly ignore your prompt
- 3: be more creative
- 7: a good balance between following the prompt and freedom
- 15: adhere more to the prompt
- 30: strictly follow the prompt

## Prompt scheduling and Keyword weight

Prompt scheduling
- `[Ana de Armas:Emma Watson:0.5]`: First half of the samplings will render Ana de Armas, and the last half of the samplings will render Emma Watson.
```
photo of young woman, [Ana de Armas:Emma Watson:0.5], highlight hair, sitting outside restaurant, wearing dress, rim lighting, studio lighting, looking at the camera, dslr, ultra quality, sharp focus, tack sharp, dof, film grain, Fujifilm XT3, crystal clear, 8K UHD, highly detailed glossy eyes, high detailed skin, skin pores
```
- `[woman:Ana de Armas:0.4]`: Certain names will influence the whole image, not just face, through the association effect. To create several images differing only in face, put feature-less word in the initial samplings.
```
photo of young [woman:Ana de Armas:0.4], highlight hair, sitting outside restaurant, wearing dress, rim lighting, studio lighting, looking at the camera, dslr, ultra quality, sharp focus, tack sharp, dof, film grain, Fujifilm XT3, crystal clear, 8K UHD, highly detailed glossy eyes, high detailed skin, skin pores
```

Keyword weight
- `(disfigured:1.5)`
```
(disfigured:1.5), ugly, bad, immature, cartoon, anime, 3d, painting, b&w
```

## Conditioning

### LoRA

LoRA (Low-Rank Adaptation) weights modify the models's cross-attention weights. It reduces the file size by representing (m, n) matrix as a multiplication of (m, x) matrix and (x, n) matrix.

Example: [epi_noiseoffset](https://civitai.com/models/13941/epinoiseoffset)
- Base model: SD 1.5 
- Trigger words: dark studio, rim lighting, two tone lighting, dimly lit, low key

add LoRA file in `root/Lora` -> Lora tab -> click epi_noiseoffset2 -> see `<lora:epiNoiseoffset_v2:1>` added in the positive prompt -> add trigger words in the positive prompt

### Embedding (Textual Inversion)

Embedding for a new, unseen token is created in embedding lookup table through textual inversion. It finds the embedding vector for the new keyword that best represents the new style or object. It can inject new styles or objects without modifying the model, with as few as 3-5 sample images.

add embedding file in `root/embeddings` -> Textual Inversion tab -> click embedding_file_name -> see `embedding_file_name` added in the positive prompt 

### ControlNet Image Generation (Edge, Pose, Depth)

ControlNet conditions the model using detected edge, human pose, depth, etc. The process of extracting those information is called annotation or preprocessing.

Extension -> Install from URL -> https://github.com/Mikubill/sd-webui-controlnet -> Settings -> Reload UI -> add [ControlNet models](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main) in `stable-diffusion-webui/extensions/sd-webui-controlnet/models`

set inputs for txt2img -> check Enable, Pixel Perfect (generated image size is as specified in the Generation tab), and Allow Preview -> choose Preprocessor and Model (e.g., openpose_full and control_v11p_sd15_openpose) -> click Run preprocessor (explosion emoji) -> Generate

Reference preprocessors:
- reference_adain: style transfer via adaptive instance normalization
- reference_only: link the reference image directly to the attention layers
- reference_adain+attn: combination of both

IP-Adapter (Image Prompt Adapter - image as a prompt) models: 
- [ip-adapter_sd15.pth](https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/ip-adapter_sd15.pth)
- [ip-adapter_sd15_plus.pth](https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/ip-adapter_sd15_plus.pth)

uncheck Pixel Perfect -> Preprocessor: ip-adapter_clip_sd15 -> Model: ip-adapter_sd15 -> Control Weight: 0.5 (experiment with 0.1-1.0 range) -> click Run preprocessor (explosion emoji) -> Generate

## Fixing

### ADetailer

Automatic face inpainting.

Extension -> Install from URL -> https://github.com/Bing-su/adetailer -> Settings -> Reload UI

- Apply ADetailer while generating an image (text-to-image): txt2img -> ADetailer -> Enable ADetailer -> ADetailer model: face_yolov8n.pt -> Generate
- Apply ADetailer on a generated image (image-to-image): PNG Info -> Send to img2img -> Denoising strength (of img2img, not ADetailer): 0.1 -> ADetailer -> Enable ADetailer -> ADetailer model: face_yolov8n.pt -> Generate

### ESRGAN Upscaler

Extras -> Single Image -> upload image -> Scale by: 4 -> Upscaler 1: ESRGAN_4x -> Generate

### ControlNet Upscaler

Extension -> Install from URL -> https://github.com/Coyote-A/ultimate-upscale-for-automatic1111 -> Settings -> Reload UI

img2img tab -> upload the image to resize -> Resize mode: Just resize -> Sampling steps: 50 -> Denoising strength 0.5 -> Enable ControlNet -> Upload independent control image: upload the same image to ControlNet's image canvas -> Preprocessor: tile_resample -> Model: control_v11f1e_sd15_tile -> Script: Ultimate SD upscale -> Target size type: Scale from image size -> Scale: 4 -> Upscaler: ESRGAN_4x or R-ESRGAN 4x+ or 4x_UniversalUpscalerV2-Neutral_115000_swaG -> Tile width: 512, Tile height: 0 -> Generate

### Inpainting

PNG Info -> Send to inpaint -> draw mask -> modify the prompt -> Masked content: original -> Inpaint area: Whole picture -> Denoising strength: 1 -> Generate

### ControlNet Inpainting

PNG Info -> Send to inpaint -> draw mask -> modify the prompt -> Masked content: original -> Inpaint area: Whole picture -> Denoising strength: 1 -> Enable ControlNet -> Preprocessor: inpaint_global_harmonious -> Model: control_v11p_sd15_inpaint -> Generate

## Sources

- [Stable Diffusion Colab tutorial](https://stable-diffusion-art.com/automatic1111-colab/), [Colab notebook purchase](https://andrewongai.gumroad.com/l/stable_diffusion_quick_start), [Colab notebook](https://colab.research.google.com/github/sagiodev/stablediffusion_webui/blob/master/StableDiffusionUI_ngrok_sagiodev.ipynb)
