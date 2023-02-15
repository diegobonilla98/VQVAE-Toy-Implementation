import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import VQVAE


image_size = 128

model = VQVAE(256, 8, 128, 32)

model_path = "VQVAE_C_256_N_8_M_128_D_32\\model.ckpt-3500.pt"
model_state_dict = torch.load(model_path)['model']
model.load_state_dict(model_state_dict, strict=False)
model = model.cuda()
model.eval()

transformation = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda l: l - 0.5)
])

image_path = "D:\\FFHQ\\images1024x1024\\images1024x1024\\11000\\11039.png"
image = Image.open(image_path)
image_tensor = transformation(image).unsqueeze(0)
image_tensor = image_tensor.cuda()

with torch.no_grad():
    dist, _, _, embedding = model(image_tensor)

targets = (image_tensor + 0.5) * 255
targets = targets.long()
logp = dist.log_prob(targets).sum((1, 2, 3)).mean()

samples = torch.argmax(dist.logits, dim=-1)
image_rec = (samples.squeeze().permute(1, 2, 0).cpu().numpy() / 255)


plt.figure(0)
plt.title("Original")
plt.imshow(image)

plt.figure(1)
plt.title("Reconstructed")
plt.imshow(image_rec)

plt.show()
