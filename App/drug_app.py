import gradio as gr
import skops.io as sio
from skops.io import load, get_untrusted_types
import tensorflow as tf
import numpy as np

# --- START: Load Saved Preprocessors, Label Mapping, and TFLite Model ---

# Load fitted scikit-learn preprocessors
# Need to load them individually as they were saved this way
print("Loading preprocessors...")
with open("Model/cat_imputer.skops", "rb") as f:
    untrusted = get_untrusted_types(file=f)
cat_imputer = load(file=open("Model/cat_imputer.skops", "rb"), trusted=untrusted)

with open("Model/encoder.skops", "rb") as f:
    untrusted = get_untrusted_types(file=f)
encoder = load(file=open("Model/encoder.skops", "rb"), trusted=untrusted)

with open("Model/num_imputer.skops", "rb") as f:
    untrusted = get_untrusted_types(file=f)
num_imputer = load(file=open("Model/num_imputer.skops", "rb"), trusted=untrusted)

with open("Model/scaler.skops", "rb") as f:
    untrusted = get_untrusted_types(file=f)
scaler = load(file=open("Model/scaler.skops", "rb"), trusted=untrusted)
print("Preprocessors loaded.")


# Load the label mapping
print("Loading label mapping...")
# Label mapping is just a dict, it's generally safe to trust if the source is trusted
label_mapping = load(file=open("Model/label_mapping.skops", "rb"), trusted=True)
reverse_label_mapping = {i: label for label, i in label_mapping.items()} # Create reverse mapping for output
print("Label mapping loaded.")

# Load the TFLite model
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path="Model/drug_model_quant.tflite")
interpreter.allocate_tensors()
print("TFLite model loaded.")
# --- END: Load Saved Preprocessors, Label Mapping, and TFLite Model ---


def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k_ratio):
    """Predict drugs based on patient features.

    Args:
        age (int): Age of patient
        sex (str): Sex of patient
        blood_pressure (str): Blood pressure level
        cholesterol (str): Cholesterol level
        na_to_k_ratio (float): Ratio of sodium to potassium in blood

    Returns:
        str: Predicted drug label
    """
    # Input features in the original order: Age, Sex, BP, Cholesterol, Na_to_K
    raw_features = [age, sex, blood_pressure, cholesterol, na_to_k_ratio]

    # Define original categorical and numerical column indices based on the raw input list
    # These indices must match how the preprocessors were fitted in train.py
    cat_cols_idx = [1, 2, 3] # Sex, BP, Cholesterol
    num_cols_idx = [0, 4] # Age, Na_to_K

    # Convert input list to a numpy array (shape 1, 5) for preprocessing
    # Use dtype object to handle mixed types before specific transformers
    raw_features_array = np.array([raw_features], dtype=object)

    # Apply preprocessing steps manually in the correct order
    # 1. Impute & encode categoricals
    cat_features_imputed = cat_imputer.transform(raw_features_array[:, cat_cols_idx])
    cat_features_processed = encoder.transform(cat_features_imputed)

    # 2. Impute & scale numerics
    # Ensure numeric data is treated as numeric
    num_features_array = raw_features_array[:, num_cols_idx].astype(float)
    num_features_imputed = num_imputer.transform(num_features_array)
    num_features_processed = scaler.transform(num_features_imputed)

    # 3. Stack into one feature matrix
    # Ensure the order matches the training data (categorical first, then numerical)
    preprocessed_features = np.hstack([cat_features_processed, num_features_processed])

    # --- START: TFLite Inference ---

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare input tensor (quantization is needed for uint8 input)
    # Dequantize formula: real_value = (quantized_value - zero_point) * scale
    # Quantize formula: quantized_value = real_value / scale + zero_point
    # Round to nearest integer and clip to the uint8 range [0, 255]
    input_scale, input_zero_point = input_details[0]['quantization']
    input_tensor = preprocessed_features / input_scale + input_zero_point
    input_tensor = np.round(input_tensor) # Round to nearest integer
    input_tensor = np.clip(input_tensor, 0, 255) # Clip to uint8 range
    input_tensor = input_tensor.astype(input_details[0]['dtype']) # Ensure dtype matches TFLite input


    # Set input tensor and invoke interpreter
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # Get output tensor and dequantize
    output_tensor = interpreter.get_tensor(output_details[0]['index'])
    # Dequantize the output (it is uint8)
    output_scale, output_zero_point = output_details[0]['quantization']
    output_scores = output_tensor.astype(np.float32) # Convert to float first
    output_scores = (output_scores - output_zero_point) * output_scale


    # Get the predicted class index (highest score)
    predicted_class_index = np.argmax(output_scores)

    # Map the predicted index back to the original drug label
    predicted_drug = reverse_label_mapping.get(predicted_class_index, "Unknown Drug") # Handle potential unknown index

    # --- END: TFLite Inference ---

    label = f"Predicted Drug: {predicted_drug}" # Keep the output format consistent
    return label


inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Sex"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(6.2, 38.2, step=0.1, label="Na_to_K"),
]
outputs = [gr.Label(num_top_classes=5)]

examples = [
    [30, "M", "HIGH", "NORMAL", 15.4],
    [35, "F", "LOW", "NORMAL", 8],
    [50, "M", "HIGH", "HIGH", 34],
]


title = "Drug Classification (TensorFlow Lite)"
description = "Enter the details to get a drug type prediction using a quantized TFLite model."
article = "This app demonstrates a CI/CD pipeline for ML models, including training, evaluation, and deployment to Hugging Face Spaces with a TFLite model."


gr.Interface(
    fn=predict_drug,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
).launch()