import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")

df

df.describe()

disease_counts = df['diseases'].value_counts()
print(disease_counts)

df.duplicated().sum()



# Check for exact duplicates
exact_duplicates = df[df.duplicated(keep=False)]
print(f"Number of exact duplicates: {len(exact_duplicates)}")



# inspect samples

# Display a sample of duplicated rows
print(exact_duplicates.sample(5))



# object_columns 

df.select_dtypes(include='object').columns

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Loop through all categorical columns and encode them
for column in df.select_dtypes(include='object').columns:
    df[column] = label_encoder.fit_transform(df[column])



# Assuming 'diseases' is your target column
X = df.drop(columns=['diseases'])  # Features (all columns except 'diseases')
y = df['diseases']  # Target (the 'diseases' column)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")



from sklearn.ensemble import IsolationForest

# Define features (X) and target (y)
X = df.drop(columns=['diseases'])  # Features
y = df['diseases']  # Target

# Fit Isolation Forest to detect outliers
iso_forest = IsolationForest(contamination=0.01)  # Adjust contamination as needed
outliers = iso_forest.fit_predict(X)

# -1 indicates outliers, 1 indicates inliers
X_cleaned = X[outliers == 1]
y_cleaned = y[outliers == 1]

print(f"Original dataset shape: {X.shape}")
print(f"Dataset shape after removing outliers: {X_cleaned.shape}")

# Check the cleaned dataset
print(X_cleaned.head())
print(y_cleaned.head())



print(f"Dataset dimensions: {df.shape}")

print(f"Cleaned features dimensions: {X_cleaned.shape}")
print(f"Cleaned target dimensions: {y_cleaned.shape}")







import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Assuming X_cleaned and y_cleaned are already defined
# X_cleaned: Cleaned features (after removing outliers)
# y_cleaned: Cleaned target variable (after removing outliers)

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize the data
    ('pca', PCA())  # Step 2: Apply PCA (without reducing dimensions)
])

# Fit the pipeline to the cleaned data
pipeline.fit(X_cleaned)

# Extract the PCA object from the pipeline
pca = pipeline.named_steps['pca']

# Calculate explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Calculate cumulative explained variance
cumulative_variance = explained_variance.cumsum()

# Plot the cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(cumulative_variance, marker='o', linestyle='-', color='b', label='Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by Number of Components (Cleaned Data)')
plt.grid()
plt.legend()
plt.show()



# Replace `optimal_components` with the number you identified from the graph
optimal_components = 320  # Example: Replace with the actual number from the graph

# Update the PCA step in the pipeline to use the optimal number of components
pipeline.set_params(pca__n_components=optimal_components)

# Fit and transform the data using the updated pipeline
X_pca_scaled = pipeline.fit_transform(X_cleaned)

print(f"Transformed and scaled data shape: {X_pca_scaled.shape}")



# Convert the transformed and scaled data to a DataFrame
X_pca_scaled_df = pd.DataFrame(X_pca_scaled, columns=[f"PC{i+1}" for i in range(X_pca_scaled.shape[1])])

# Display the first 5 rows of the transformed and scaled data
print("Transformed and Scaled Data (First 5 Rows):")
print(X_pca_scaled_df.head())

# Check the shape of the transformed and scaled data
print(f"Shape of the transformed and scaled data: {X_pca_scaled_df.shape}")

from mpl_toolkits.mplot3d import Axes3D

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the first three principal components
ax.scatter(X_pca_scaled[:, 0], X_pca_scaled[:, 1], X_pca_scaled[:, 2], alpha=0.5)

# Add labels and title
ax.set_xlabel('Principal Component 1 (PC1)')
ax.set_ylabel('Principal Component 2 (PC2)')
ax.set_zlabel('Principal Component 3 (PC3)')
ax.set_title('3D Scatter Plot of Transformed and Scaled Data')

# Show the plot
plt.show()



import pandas as pd

# Convert the transformed and scaled data to a DataFrame
X_pca_scaled_df = pd.DataFrame(X_pca_scaled, columns=[f"PC{i+1}" for i in range(X_pca_scaled.shape[1])])

# Add the target variable (y_cleaned) to the DataFrame
X_pca_scaled_df['diseases'] = y_cleaned.reset_index(drop=True)

# Save the DataFrame to a CSV file
X_pca_scaled_df.to_csv('preprocessed_data.csv', index=False)

print("Preprocessed data saved to 'preprocessed_data.csv'")

df_preprocessed = pd.read_csv("preprocessed_data.csv")

df_preprocessed

# Separate features (X) and target (y)

X = df_preprocessed.drop(columns=['diseases'])  # Features (principal components)
y = df_preprocessed['diseases']  # Target variable

# Check the shapes of X and y
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Check the data types of all columns
print(df_preprocessed.dtypes)

# Check for categorical columns (non-numeric columns)
categorical_columns = df_preprocessed.select_dtypes(include=['object', 'category']).columns
print("Categorical columns:", categorical_columns)

# Check the unique values in the 'diseases' column
print("Unique values in 'diseases':", df_preprocessed['diseases'].unique())

# Check the data type of the 'diseases' column
print("Data type of 'diseases':", df_preprocessed['diseases'].dtype)



from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the splits
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing labels shape: {y_test.shape}")













import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np

# Ensure reproducibility
tf.keras.utils.set_random_seed(42)

# Load preprocessed data (assuming X and y are already defined)
# X = df_preprocessed.drop(columns=['diseases'])
# y = df_preprocessed['diseases']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
def create_model(learning_rate=0.0001, dropout_rate=0.2):
    model = Sequential()

    # Input layer
    model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Hidden layers
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Output layer
    num_classes = len(np.unique(y))  # Number of unique classes
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Create the model
model = create_model(learning_rate=0.0001, dropout_rate=0.2)

# Print the model summary
model.summary()

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=10,  # Stop if no improvement for 10 epochs
    restore_best_weights=True  # Restore the best model weights
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss
    factor=0.5,  # Reduce learning rate by 50%
    patience=3,  # Wait for 3 epochs without improvement
    min_lr=1e-6  # Minimum learning rate
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,  # Increased number of epochs
    batch_size=64,  # Batch size
    verbose=1,  # Show progress
    callbacks=[early_stopping, reduce_lr]  # Use early stopping and learning rate scheduler
)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=1)
print(f"Validation Accuracy: {val_accuracy}")

# Save the model
model.save('optimized_disease_prediction_model_v2.h5')
print("Optimized model saved to 'optimized_disease_prediction_model_v2.h5'")






# Define callbacks with increased patience
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=20,  # Increased patience
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6  # Reduced minimum learning rate
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,  # Increased number of epochs
    batch_size=64,
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)

import pickle

# Your data or objects to be pickled
data = {"key": "value"}

# Save the data to a .pkl file
with open('your_file.pkl', 'wb') as file:
    pickle.dump(data, file)


import os
print(os.listdir())




print(os.listdir())










