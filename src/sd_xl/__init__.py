import torch
from diffusers import AutoPipelineForText2Image

print("Cargando el modelo ...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    torch_dtype=torch.float32,
)

modelo = modelo.to("cpu")

prompt = input("Escribe el prompt de la imagen que quieres generar: ")
negative_prompt = "blurry, low quality, worst quality"

print("Generando imagen...")

imagen = modelo(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=20,
    height=768,
    width=768,
).images[0]

imagen.save("imagen.png")
print("Imagen guardada como imagen.png")
