from unet import UNet

import torch
import numpy as np
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt

def load_model(model_path, device):
    model = UNet(dim=1,n_filters=100,FL=3,init= "he_normal",drop=0.10,lmbda=0.00001).to("cuda")
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load the saved model weights
    model.to(device)  # Move model to the appropriate device (GPU or CPU)
    model.eval()  # Set the model to evaluation mode
    return model

def predict_image(model, image_path, device):
    # Step 1: Load the grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = image[1:, 1:]
    # Step 2: Preprocess the image
    # Normalize image (this should match your training preprocessing)
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming you normalized with mean=0.5 and std=0.5
    ])

    # Convert the image to a tensor and normalize it
    image_tensor = normalize(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Step 3: Predict the image
    with torch.no_grad():  # No need to track gradients
        output = model(image_tensor)  # Forward pass
        output = output.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU

    # Step 4: Post-process the output (apply a threshold to make it binary if required)
    output = np.squeeze(output)  # Remove singleton dimensions
    # output = (output > 0.5).astype(np.uint8) * 255  # Threshold output to 0 or 255
    output = output * 255  # Threshold output upto 255
    output = output.astype(np.uint8)  # Convert to uint8 for displaying
    # Step 5: Display or save the predicted image
    plt.imshow(output, cmap='gray')
    # plt.title('Predicted Image')
    plt.axis('off')  # Hide axes
    plt.show()

    # Optionally, save the output image
    # cv2.imwrite('output_image.png', output)

# Example usage:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "/content/best_model.pth"  # Path to your saved model
model = load_model(model_path, device)

image_path = "/content/togepi.png"  # Replace with the path to your input image
predict_image(model, image_path, device)
