#StableDiffusionControlNetPipeline
!pip install diffusers["torch"] transformers
!pip install accelerate
!pip install git+https://github.com/huggingface/diffusers
!pip install safetensors
!pip install compel
!pip install -q controlnet-aux


import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.utils import load_image, make_image_grid
from controlnet_aux import OpenposeDetector
from PIL import Image
from compel import Compel

controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-openpose", torch_dtype=torch.float16
)

SD_Models = "stablediffusionapi/anything-v5" # @param ["stablediffusionapi/mistoonanime","stablediffusionapi/sakushimix-hentai","stablediffusionapi/anything-v5","Meina/MeinaHentai_V4"] {allow-input: true}
!wget https://civitai.com/api/download/models/172508 -O lora.safetensors

pipe = StableDiffusionControlNetPipeline.from_pretrained(SD_Models, torch_dtype=torch.float16,safety_checker = None, controlnet=controlnet,)
pipe.load_lora_weights(".", weight_name="lora.safetensors")
pipe = pipe.to("cuda")


#Detect Poses
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
original_image = load_image(
    "https://github.com/maczxc/ai-poses/blob/main/poses-1/3_.png?raw=true"
)
openpose_image = openpose(original_image)
make_image_grid([original_image, openpose_image], rows=1, cols=2)


#Prompts and Negative Prompts
prompt = "anime, nobarakugisakinova, naked {{{ultra sharpen}}}. (chromatic aberration), ((masterpiece:1.3)), (((best quality:1.3))), (((high resolution))), {beautiful detailed eyes}, cheerful smile, skirt, {tank top white}, {{thigh}}, bed, {shadowlighting}, mature, 20 years old, {{{plain background}}}, {{{white background}}}, kneeling, seductive look"# @param {type:"string"}
compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
conditioning = compel.build_conditioning_tensor(prompt)

negativePrompt = "verybadimagenegative_v1.3, ng_deepnegative_v1_75t, (ugly face:0.8),cross-eyed,sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, bad anatomy, DeepNegative, facing away, tilted head, {Multiple people}, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed,mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot, ((repeating hair)), cartoon, illustration, realistic" # @param {type:"string"}

#Sampling Methods
sampling_method = "EulerAncestral" # @param ["DPM++ 2M Karras","DPM++ 2M SDE Karras","EulerAncestral"]
if sampling_method == "DPM++ 2M Karras":
  pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True)
if sampling_method == "DPM++ 2M SDE Karras":
  pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config,use_karras_sigmas=True,algorithm_type="sde-dpmsolver++")
elif sampling_method == "EulerAncestral":
  pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

#Settings
height = 800 # @param {type:"integer"}
width = 600 # @param {type:"integer"}
steps = 30 # @param {type:"slider", min:1, max:150, step:1}
guidance = 5 # @param {type:"slider", min:1, max:30, step:1}
clip = 2 # @param {type:"slider", min:1, max:5, step:1}

#Output
results = pipe(prompt_embeds=conditioning, image=openpose_image, num_inference_steps=steps, guidance_scale=guidance, negative_prompt=negativePrompt, clip_skip=clip).images[0]
results
