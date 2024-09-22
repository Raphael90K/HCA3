import numpy as np
import hailo_platform
from PIL import Image
from hailo_platform import HailoModel

# Load your Hailo model
hef_file = "../model/mobilenetv2_cifar10.hef"
model = HailoModel(hef_file)


# Prepare the input preprocessing
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))  # Resize to the model's input size
    image_array = np.array(image).astype(np.float32)  # Convert to numpy array
    image_array = image_array / 255.0  # Normalize the image if required
    image_array = np.transpose(image_array, (2, 0, 1))  # Change to CxHxW format
    return image_array


# Run inference on an image
def run_inference(image_path):
    input_data = preprocess_image(image_path)

    # Prepare input shape as needed (1, C, H, W)
    input_data = np.expand_dims(input_data, axis=0)

    # Run inference
    output = model.predict(input_data)

    return output


# Test the function
image_path = "../img/1.jpg"
output = run_inference(image_path)

# Process the output (depends on your model)
print("Output:", output)
