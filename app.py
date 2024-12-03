from flask import Flask, render_template, request
from train4 import MyDataset, Resnet18_two_stream
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Resnet18_two_stream().to(DEVICE)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to perform prediction
def predict_image(rgb_img_path, ir_img_path):
    # Load and preprocess images
    rgb_img = Image.open(rgb_img_path).convert('RGB')
    ir_img = Image.open(ir_img_path).convert('RGB')
    rgb_img = transform(rgb_img).unsqueeze(0).to(DEVICE)
    ir_img = transform(ir_img).unsqueeze(0).to(DEVICE)

    # Perform prediction
    with torch.no_grad():
        output = model(rgb_img, ir_img, mode='both')
        _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get file paths from form data
        rgb_image = request.files['rgb_image']
        ir_image = request.files['ir_image']

        # Perform prediction
        predicted_class = predict_image(rgb_image, ir_image)

        # Return the predicted class as response
        return f"Predicted class: {predicted_class}"

if __name__ == '__main__':
    app.run(debug=True)