import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

# — 1. Load your CSV —
df = pd.read_csv("dataset.csv")

# — 2. Exploratory Visualizations —

# 2.1 Pairplot
sns.pairplot(df, hue="type", diag_kind="kde")
plt.suptitle("Pairplot of Bottle Dimensions by Type", y=1.02)
plt.show()

# 2.2 Histograms
df[["height", "width", "length"]].hist(bins=15, figsize=(8, 4))
plt.suptitle("Histograms of height, width, length")
plt.tight_layout()
plt.show()

# 2.3 Correlation Heatmap
corr = df[["height", "width", "length"]].corr()
plt.figure(figsize=(5, 4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag")
plt.title("Correlation Heatmap")
plt.show()

# 2.4 Boxplots by type
df_melt = df.melt(id_vars="type",
                  value_vars=["height", "width", "length"],
                  var_name="Dimension", value_name="Value")
plt.figure(figsize=(8, 4))
sns.boxplot(x="Dimension", y="Value", hue="type", data=df_melt)
plt.title("Boxplots of Dimensions by Type")
plt.show()

# 2.5 Scatterplot (height vs. length)
plt.figure(figsize=(6, 5))
sns.scatterplot(x="height", y="length", hue="type", data=df, s=60)
plt.title("Scatterplot: height vs. length by Type")
plt.show()

# — 3. Prepare Data for Modeling —
X = df[["height", "width", "length"]]
le = LabelEncoder()
y = le.fit_transform(df["type"])

# 3.1 Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# — 4. Train RandomForest —
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# — 5. Evaluate with Confusion Matrix —
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=le.classes_)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix on Test Set")
plt.tight_layout()
plt.show()

# — 6. Save Artifacts —
joblib.dump(model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Model and encoder saved.")
