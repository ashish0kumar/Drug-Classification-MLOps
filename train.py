import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import skops.io as sio

# --- START: Data Loading and Initial Split ---
drug_df = pd.read_csv("Data/drug.csv")
drug_df = drug_df.sample(frac=1, random_state=125) # Added random_state for reproducibility

# Split out raw arrays
# Assuming the column order is fixed: Age, Sex, BP, Cholesterol, Na_to_K, Drug
# Indices: 0:Age (num), 1:Sex (cat), 2:BP (cat), 3:Cholesterol (cat), 4:Na_to_K (num), 5:Drug (target)
cat_cols_idx = [1, 2, 3] # Sex, BP, Cholesterol
num_cols_idx = [0, 4] # Age, Na_to_K

X_raw = drug_df.drop("Drug", axis=1).values
y_raw = drug_df.Drug.values
# --- END: Data Loading and Initial Split ---


# --- START: Preprocessing Setup and Application ---
# Define and fit preprocessing steps separately using the full raw data
# 1️⃣ Impute & encode categoricals
cat_imputer = SimpleImputer(strategy="most_frequent")
encoder = OrdinalEncoder()

# 2️⃣ Impute & scale numerics
num_imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

# Apply imputers and encoders/scalers sequentially to handle dependencies
# These fitted preprocessors will be saved for the app
print("Fitting preprocessors...")

# Apply categorical imputer first
X_cat_imputed = cat_imputer.fit_transform(X_raw[:, cat_cols_idx])

# Apply ordinal encoder
X_cat_processed = encoder.fit_transform(X_cat_imputed)

# Apply numerical imputer
X_num_imputed = num_imputer.fit_transform(X_raw[:, num_cols_idx])

# Apply standard scaler
X_num_processed = scaler.fit_transform(X_num_imputed)

# 3️⃣ Stack into one feature matrix
X_processed = np.hstack([X_cat_processed, X_num_processed])
print("Preprocessing complete.")

# Map string labels to integers
label_mapping = {label: i for i,label in enumerate(np.unique(y_raw))}
y_encoded = np.vectorize(label_mapping.get)(y_raw)

# Train/test split using the processed features and encoded labels
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.3, random_state=125, stratify=y_encoded # Use stratify with encoded labels
)
# --- END: Preprocessing Setup and Application ---


# --- START: Save Preprocessors and Label Mapping ---
# Save the fitted preprocessors and the label mapping using skops
print("Saving preprocessors and label mapping...")
sio.dump(cat_imputer, "Model/cat_imputer.skops")
sio.dump(encoder, "Model/encoder.skops")
sio.dump(num_imputer, "Model/num_imputer.skops")
sio.dump(scaler, "Model/scaler.skops")
sio.dump(label_mapping, "Model/label_mapping.skops")
print("Preprocessors and label mapping saved.")
# --- END: Save Preprocessors and Label Mapping ---


# --- START: TensorFlow Model Definition and Training ---
# A very small MLP
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(label_mapping), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training TensorFlow model...")
model.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.1, verbose=0) # Reduced verbosity for CI
print("Training complete.")
# --- END: TensorFlow Model Definition and Training ---


# --- START: TFLite Conversion ---
# Set up a representative dataset generator for TFLite quantization
def representative_data_gen():
    # Use a subset of training data
    for i in range(min(100, len(X_train))):
        # pick a sample
        sample = X_train[i]
        yield [sample.astype(np.float32).reshape(1, -1)] # Reshape sample to (1, num_features)

# Convert Keras model to TFLite
print("Converting Keras model to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Force full integer quantization for weights & activations
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
print("TFLite conversion complete.")

# Save TFLite model to disk
tflite_model_path = "Model/drug_model_quant.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("TFLite model size:", len(tflite_model), "bytes")
print(f"TFLite model saved to {tflite_model_path}")
# --- END: TFLite Conversion ---


# --- START: Evaluation and Reporting (Adjusted for TF model) ---

# Evaluate the Keras model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy (Keras) = {accuracy * 100:.2f}%")

# Get predictions from the Keras model for plotting/matrix calculation
keras_predictions_probs = model.predict(X_test)
keras_predictions_indices = np.argmax(keras_predictions_probs, axis=1)

# Decode predictions from indices back to original labels for confusion matrix
reverse_label_mapping = {i: label for label, i in label_mapping.items()}
keras_predicted_labels = np.vectorize(reverse_label_mapping.get)(keras_predictions_indices)
test_true_labels = np.vectorize(reverse_label_mapping.get)(y_test)

# For the confusion matrix plot, we need the unique classes in their original string format
class_names = list(label_mapping.keys()) # Get class names from the mapping

# Calculate confusion matrix using decoded labels
cm = confusion_matrix(test_true_labels, keras_predicted_labels, labels=class_names)

# Plotting the confusion matrix
print("Generating confusion matrix plot...")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)
print("Confusion matrix plot saved to Results/model_results.png")

# Write metrics to file
# Use the accuracy obtained directly from model.evaluate
print(f"Saving Accuracy to Results/metrics.txt: {accuracy:.4f}")
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nTest Accuracy (Keras) = {accuracy:.4f}")
print("Metrics saved to Results/metrics.txt")

# --- END: Evaluation and Reporting ---