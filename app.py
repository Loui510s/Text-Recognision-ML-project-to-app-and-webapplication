from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch
from io import BytesIO
import base64

# Import your CNN model and prediction logic
class CNN(torch.nn.Module): 
    def __init__(self): 
        super(CNN, self).__init__() 
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3) 
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3) 
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3) 
        self.pool = torch.nn.MaxPool2d(2, 2) 
        self.dropout = torch.nn.Dropout(0.25) 
        self.relu = torch.nn.ReLU()
        
        # Pass a sample input through conv layers to calculate output size
        sample_input = torch.zeros(1, 1, 28, 28)
        conv_output = self._forward_conv(sample_input)
        self.fc1 = torch.nn.Linear(conv_output.numel(), 512)
        self.fc2 = torch.nn.Linear(512, 10)

    def _forward_conv(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)
        return x
    
    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def predict_digit(model, image): 
    image = image.resize((28, 28)).convert('L') 
    image = np.array(image).reshape(1, 1, 28, 28) / 255.0 
    image = torch.tensor(image, dtype=torch.float32) 
    image = (image - 0.5) / 0.5  # Normalization for the model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = model.to(device) 
    image = image.to(device) 

    with torch.no_grad(): 
        outputs = model(image) 
    _, predicted = torch.max(outputs.data, 1) 
    confidence = torch.softmax(outputs, dim=1) * 100  # Get confidence as a percentage
    return predicted.item(), confidence[0][predicted].item()

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = CNN()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.json.get('image')
        image_data = base64.b64decode(data.split(",")[1])  # Remove base64 header
        image = Image.open(BytesIO(image_data))

        # Predict digit
        digit, confidence = predict_digit(model, image)
        return jsonify({'digit': digit, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
