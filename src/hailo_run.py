import numpy as np
from PIL import Image
import importlib

try:
    # Dynamischer Import des gesamten 'hailo_platform'-Moduls
    hailo_platform = importlib.import_module("hailo_platform")
    print("hailo_platform Modul wurde erfolgreich importiert.")

    # Zugriff auf die benötigten Klassen und Funktionen
    HEF = hailo_platform.HEF
    VDevice = hailo_platform.VDevice
    HailoStreamInterface = hailo_platform.HailoStreamInterface
    InferVStreams = hailo_platform.InferVStreams
    ConfigureParams = hailo_platform.ConfigureParams
    InputVStreamParams = hailo_platform.InputVStreamParams
    OutputVStreamParams = hailo_platform.OutputVStreamParams
    InputVStreams = hailo_platform.InputVStreams
    OutputVStreams = hailo_platform.OutputVStreams
    FormatType = hailo_platform.FormatType
except ImportError:
    pass

from profiler import Profiler

# Map the output (usually an index) to human-readable labels
class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
profiler = Profiler()


# Load and preprocess the image to match the input shape
def preprocess_image(image_path, image_height, image_width):
    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Resize the image to the model's input size
    image = image.resize((image_width, image_height))

    # Convert the image to a numpy array
    image_array = np.array(image).astype(np.float32)  # Hailo models usually expect float32 data

    # Normalize the image (this is standard for models trained on ImageNet or CIFAR)
    image_array = image_array / 255.0  # Normalize pixel values to [0, 1]

    # Add batch dimension (1, height, width, channels)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# Main function for loading model and performing inference
def classify_image(image_path):
    # Initialize the Hailo VDevice
    target = VDevice()

    # Load the HEF file for the model
    model_name = 'mobilenetv2_cifar10'
    hef_path = '../model/{}.hef'.format(model_name)
    hef = HEF(hef_path)

    # Configure network groups
    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    network_group_params = network_group.create_params()

    # Get input and output virtual stream info
    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_info = hef.get_output_vstream_infos()[0]
    image_height, image_width, channels = input_vstream_info.shape  # Input image size

    # Preprocess the input image
    input_data = preprocess_image(image_path, image_height, image_width)

    # Create input and output virtual streams params
    input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.UINT8)

    # profile
    profiler.profile(use_model, input_data, input_vstream_info, input_vstreams_params,
                     network_group, network_group_params, output_vstream_info, output_vstreams_params)


def use_model(input_data, input_vstream_info, input_vstreams_params, network_group, network_group_params,
              output_vstream_info, output_vstreams_params):
    # Run inference
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        input_dict = {input_vstream_info.name: input_data}
        with network_group.activate(network_group_params):
            infer_results = infer_pipeline.infer(input_dict)
            output_data = infer_results[output_vstream_info.name]

            # Assuming output is a class index (this depends on your model's output format)
            predicted_class = np.argmax(output_data)
            print(f'Predicted class index: {predicted_class}')
            print(f'Predicted label: {class_labels[predicted_class]}')


# Test the function with your image
image_path = '../img/1.jpg'  # Replace with the actual path to the image
classify_image(image_path)
