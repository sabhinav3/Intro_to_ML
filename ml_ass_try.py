import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, plot_confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, plot_confusion_matrix

# Load diabetes dataset
data = datasets.load_diabetes(as_frame=True)
X = data.data
y = (data.target > data.target.median()).astype(int)  # Make problem binary

# Splitting the data into training and test sets (80% and 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Scaling and standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the logistic regression model
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = log_reg.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Plotting confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Load the breast cancer dataset
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and test sets (80% and 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Scaling and standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the logistic regression model
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = log_reg.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Plotting confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Load breast cancer dataset
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# Splitting the data into training and test sets (80% and 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Scaling and standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

# Evaluating the model
y_pred = nb_model.predict(X_test_scaled)
accuracy_nb = accuracy_score(y_test, y_pred)
precision_nb = precision_score(y_test, y_pred)
recall_nb = recall_score(y_test, y_pred)
f1_nb = f1_score(y_test, y_pred)

print(f'Naive Bayes Classifier Metrics:')
print(f'Accuracy: {accuracy_nb:.4f}')
print(f'Precision: {precision_nb:.4f}')
print(f'Recall: {recall_nb:.4f}')
print(f'F1 Score: {f1_nb:.4f}')



# Load breast cancer dataset
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# Split the data and standardize it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize variables to store results
results = []

# Loop through different k (number of principal components)
for k in range(1, X_train_scaled.shape[1]+1):
    # Perform PCA with k principal components
    pca = PCA(n_components=k)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Train a logistic regression model
    log_reg = LogisticRegression(max_iter=10000)
    log_reg.fit(X_train_pca, y_train)
    
    # Evaluate the model
    y_pred = log_reg.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Save results
    results.append((k, accuracy, precision, recall, f1))

# Convert results to a NumPy array for easy indexing
results_np = np.array(results)

# Identify k that gives the highest accuracy
best_k_index = np.argmax(results_np[:,1])
best_k, best_acc, best_prec, best_rec, best_f1 = results_np[best_k_index]

# Display results
print(f'Optimal k: {best_k}')
print(f'Accuracy: {best_acc:.4f}')
print(f'Precision: {best_prec:.4f}')
print(f'Recall: {best_rec:.4f}')
print(f'F1 Score: {best_f1:.4f}')

# Plot accuracy vs. k
plt.plot(results_np[:,0], results_np[:,1], marker='o', linestyle='-')
plt.title('Accuracy vs. Number of Principal Components')
plt.xlabel('Number of Principal Components (k)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()




# Load breast cancer dataset
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# Split the data and standardize it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize variables to store results
results_nb = []

# Loop through different k (number of principal components)
for k in range(1, X_train_scaled.shape[1]+1):
    # Perform PCA with k principal components
    pca = PCA(n_components=k)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Train a Gaussian Naive Bayes model
    nb_model = GaussianNB()
    nb_model.fit(X_train_pca, y_train)
    
    # Evaluate the model
    y_pred = nb_model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Save results
    results_nb.append((k, accuracy, precision, recall, f1))

# Convert results to a NumPy array for easy indexing
results_nb_np = np.array(results_nb)

# Identify k that gives the highest accuracy
best_k_index_nb = np.argmax(results_nb_np[:,1])
best_k_nb, best_acc_nb, best_prec_nb, best_rec_nb, best_f1_nb = results_nb_np[best_k_index_nb]

# Display results
print(f'Optimal k (Naive Bayes): {best_k_nb}')
print(f'Accuracy: {best_acc_nb:.4f}')
print(f'Precision: {best_prec_nb:.4f}')
print(f'Recall: {best_rec_nb:.4f}')
print(f'F1 Score: {best_f1_nb:.4f}')

# Plot accuracy vs. k
plt.plot(results_nb_np[:,0], results_nb_np[:,1], marker='o', linestyle='-', label='Naive Bayes')
plt.title('Accuracy vs. Number of Principal Components')
plt.xlabel('Number of Principal Components (k)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()