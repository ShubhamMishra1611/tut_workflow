from model import style_transfer
from PIL import Image
import matplotlib.pyplot as plt

# Load the content and style images
content_image = Image.open("content.jpg")
style_image = Image.open("style.jpg")

# Run the style transfer
target_image = style_transfer(content_image, style_image, num_steps=2000)

# Display the target image
plt.imshow(target_image)
plt.axis('off')

# hi

# nah