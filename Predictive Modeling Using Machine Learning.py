import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.datasets import make_classification

# ==========================================
# 1. GENERATE MOCK DATA (Customer Churn)
# ==========================================
# We create a synthetic dataset of 1,000 customers to predict if they will "Churn" (leave the company)
X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
feature_names = ['Age', 'Account_Balance', 'Num_Products', 'Credit_Score']
df = pd.DataFrame(X, columns=feature_names)
df['Churn'] = y # 0 = Stayed, 1 = Churned

print("--- SAMPLE DATA ---")
print(df.head())
print("\n")

# ==========================================
# 2. TRAIN & TEST SPLIT
# ==========================================
# Separate the features (X) from the target we want to predict (y)
X_data = df.drop('Churn', axis=1)
y_data = df['Churn']

# Split: 70% of data for training the AI, 30% for testing it
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

# ==========================================
# 3. BUILD AND TRAIN THE MODEL
# ==========================================
# Using Random Forest (An ensemble of many decision trees working together)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train) # This is where the actual "learning" happens
print("Model Training Complete...\n")

# ==========================================
# 4. MAKE PREDICTIONS & EVALUATE
# ==========================================
# Ask the model to predict the outcomes for the 30% of data it hasn't seen yet
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] # We need probabilities for the ROC curve

# Calculate standard accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# ==========================================
# 5. VISUALIZE PERFORMANCE
# ==========================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Machine Learning Model Evaluation', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# Plot 2: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (Area = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Random guess line
axes[1].set_title('ROC Curve')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()