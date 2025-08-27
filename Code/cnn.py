import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# ---------------------------
# SETTINGS
# ---------------------------
data_dir = "C:\\Users\\dhnz2\\OneDrive\\Documents\\Dataset"   # <--- change this
window_size = 16                # match INPUT_SIZE for STM32
step_size = 16                  # non-overlapping windows
channels = 3                    # X, Y, Z

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
import re
import numpy as np

def load_file(fpath):
    """Load SWV ITM accelerometer log and return numpy array of shape (timesteps, 3)."""
    data = []
    with open(fpath, "r") as f:
        for line in f:
            # Only process lines with X:, Y:, Z:
            if "X:" in line and "Y:" in line and "Z:" in line:
                # Extract numbers using regex
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if len(nums) == 3:
                    data.append([float(nums[0]), float(nums[1]), float(nums[2])])
    return np.array(data, dtype=float)



def create_windows(arr, label, window_size, step_size):
    """Slice array into windows."""
    windows, labels = [], []
    for start in range(0, len(arr) - window_size + 1, step_size):
        segment = arr[start:start+window_size]
        windows.append(segment)
        labels.append(label)
    return windows, labels

# ---------------------------
# LOAD ALL FILES
# ---------------------------
X_data, y_data = [], []

for fname in os.listdir(data_dir):
    if fname.endswith(".csv") or fname.endswith(".txt"):
        fpath = os.path.join(data_dir, fname)
        arr = load_file(fpath)

        # Normalize per file
        scaler = StandardScaler()
        arr = scaler.fit_transform(arr)

        # Label from filename
        if fname.startswith("nom"):
            label = 0
        elif fname.startswith("walk"):
            label = 1
        else:
            continue

        # Split into windows
        windows, labels = create_windows(arr, label, window_size, step_size)
        X_data.extend(windows)
        y_data.extend(labels)

X_data = np.array(X_data)  # shape: (num_samples, window_size, 3)
y_data = np.array(y_data)

print("Final dataset:", X_data.shape, y_data.shape)

# ---------------------------
# TRAIN-TEST SPLIT
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# Flatten windows for Dense NN (16*3 = 48 inputs)
INPUT_SIZE = window_size * channels
X_train = X_train.reshape(len(X_train), INPUT_SIZE)
X_val = X_val.reshape(len(X_val), INPUT_SIZE)

# ---------------------------
# BUILD DENSE NN MODEL
# ---------------------------
HIDDEN_SIZE = 8
OUTPUT_SIZE = 2   # no movement vs walking

model = Sequential([
    Dense(HIDDEN_SIZE, activation='relu', input_shape=(INPUT_SIZE,)),
    Dense(OUTPUT_SIZE, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------------------
# TRAIN
# ---------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
)

# ---------------------------
# EVALUATE
# ---------------------------
loss, acc = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {acc:.3f}")

# ---------------------------
# EXPORT WEIGHTS TO C
# ---------------------------
def export_to_c(model, filename="nn_weights.h"):
    with open(filename, "w") as f:
        f.write("// Auto-generated weights from Keras model\n\n")
        layer_idx = 0
        for layer in model.layers:
            weights = layer.get_weights()
            if not weights:
                continue

            W, b = weights   # W shape = (in_features, out_features)
            in_size, out_size = W.shape   # <-- FIXED (was reversed)

            f.write(f"// Layer {layer_idx}: Dense ({in_size} -> {out_size})\n")

            # Transpose W to [out_size][in_size] to match C-code convention
            f.write(f"float weights_{layer_idx}[{out_size}][{in_size}] = {{\n")
            for i in range(out_size):
                row = ", ".join([f"{W[j][i]:.6f}f" for j in range(in_size)])
                f.write(f"    {{{row}}},\n")
            f.write("};\n")

            f.write(f"float bias_{layer_idx}[{out_size}] = {{")
            f.write(", ".join([f"{x:.6f}f" for x in b]))
            f.write("};\n\n")

            layer_idx += 1

    print(f"✅ Weights exported correctly to {filename}")


# Export trained weights
export_to_c(model, "nn_weights.h")
