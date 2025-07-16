import joblib

# 1. Load trained artifacts
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

# 2. Define your sample(s)
#    Each sample must be a [height, width, length] list
samples = [
    [22.0, 6.5, 6.5],  # e.g. a ceramic-ish bottle
    [21.0, 6.2, 6.9],  # maybe fabric
    [24.0, 6.8, 6.6],  # likely wood
]

# 3. Predict directly from list-of-lists
y_pred = model.predict(samples)

# 4. Decode labels back to original strings
decoded = le.inverse_transform(y_pred)

# 5. Print results
for (h, w, l), label in zip(samples, decoded):
    print(f"Sample (h={h}, w={w}, l={l}) → Predicted type: {label}")

