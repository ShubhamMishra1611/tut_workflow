import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load the content and style images
content_image = Image.open("content.jpg")
style_image = Image.open("style.jpg")

# Preprocess the images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
content_tensor = preprocess(content_image).unsqueeze(0)
style_tensor = preprocess(style_image).unsqueeze(0)

# Load the pre-trained VGG19 model
vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)

# Define the content and style layers
content_layers = ['conv4_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# Define the loss functions
content_loss = nn.MSELoss()
style_loss = nn.MSELoss()

# Define the target image as a parameter for optimization
target = content_tensor.clone().requires_grad_(True)

# Set the hyperparameters
style_weight = 1e6
content_weight = 1

# Define the optimizer
optimizer = optim.Adam([target], lr=0.01)

# Run the style transfer
num_steps = 2000
for step in range(num_steps):
    optimizer.zero_grad()
    target_features = vgg(target)
    content_features = vgg(content_tensor)
    style_features = vgg(style_tensor)

    style_loss_value = 0
    content_loss_value = 0

    for layer in style_layers:
        target_feature = target_features[layer]
        style_feature = style_features[layer]
        style_loss_value += style_loss(target_feature, style_feature)

    for layer in content_layers:
        target_feature = target_features[layer]
        content_feature = content_features[layer]
        content_loss_value += content_loss(target_feature, content_feature)

    total_loss = style_weight * style_loss_value + content_weight * content_loss_value
    total_loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step [{step}/{num_steps}], Total loss: {total_loss.item()}")

# Save the stylized image
output_image = target.detach().squeeze(0)
output_image = transforms.ToPILImage()(output_image)
output_image.save("output.jpg")