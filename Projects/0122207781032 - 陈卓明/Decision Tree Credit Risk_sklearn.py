import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn import metrics # type: ignore
from sklearn.tree import export_text # type: ignore

# Load the dataset
# TODO
# 在同一文件夹中，故未添加路径
data = pd.read_csv('credit_risk_data.csv')

# Inspect the first few rows of the dataset to understand its structure and get an overview of the data
print(data.head())

# Assuming the target variable is named 'Risk' and the features are the rest of the columns
# X contains all the feature columns, while y contains the target variable ('Risk')
# 此处因目标量“Risk”实际为“risk_label”，将其改为了“risk_label”
X = data.drop('risk_label', axis=1)
y = data['risk_label']

# Split the dataset into training and testing sets
# 70% of the data will be used for training and 30% for testing
# random_state ensures reproducibility of the results
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree Classifier
# This classifier will be used to fit the model to the training data
clf = DecisionTreeClassifier()

# Train the classifier using the training data
clf = clf.fit(X_train, y_train)

# Make predictions on the test set using the trained classifier
y_pred = clf.predict(X_test)

# Evaluate the model's performance using accuracy and classification report
# Accuracy is the ratio of correctly predicted instances to the total instances
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# Classification report provides detailed metrics like precision, recall, and F1-score for each class
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))

# Export the decision tree rules as text for better interpretability
# This helps in understanding the logic and decisions made by the decision tree
r = export_text(clf, feature_names=list(X.columns))
print(r)
