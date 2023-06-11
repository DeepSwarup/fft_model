import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv('../assets/data2.2.1.csv')
data=df.drop('Domain', axis=1)

# Separate features and target variable
X = data.drop('Type', axis=1)  # Features
y = data['Type']  # Target variable

# Scale or normalize numerical features
scaler = StandardScaler()
numerical_cols = ['I+', 'I-','I0','IPH+','IPH-','IPH0','V+','V-','V0','VPH+','VPH-','VPH0','VA','VB','VC','IA','IB','IC']
# numerical_cols = ['VA','VB','VC','IA','IB','IC']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.025, random_state=42)

# Convert data to TensorFlow format
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Convert target variable to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Build the feedforward neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1])
])

# Compile the model
model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

# Train the model
history=model.fit(X_train, y_train, epochs=50, batch_size=50, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')


# Convert one-hot encoded target variables to actual target values
y_train_actual = np.argmax(y_train, axis=1)
y_test_actual = np.argmax(y_test, axis=1)

predictions = model.predict(X_test)
predicted=np.argmax(predictions, axis=1)
print(predicted)
print(y_test_actual)

# Training history
loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(loss) + 1)
l=range(len(y_test_actual))

# Define the width of each bar
bar_width = 0.35

# Create the figure and axes
fig, ax = plt.subplots()

# Plotting the training loss and validation loss
ax.bar(epochs, loss, width=bar_width, color='red', label='Training Loss')
ax.bar([e + bar_width for e in epochs], val_loss, width=bar_width, color='green', label='Validation Loss')

# Add labels and title
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss')
ax.legend()

# Adjust the x-axis ticks
ax.set_xticks([e + bar_width / 2 for e in epochs])
ax.set_xticklabels(epochs)

# Add grid
ax.grid(True)

# Show the figure
plt.show()

# Create the figure and axes
fig, ax = plt.subplots()

# Plotting the training accuracy and validation accuracy
ax.bar(epochs, accuracy, width=bar_width, color='aqua', label='Training Accuracy')
ax.bar([e + bar_width for e in epochs], val_accuracy, width=bar_width, color='black', label='Validation Accuracy')

# Add labels and title
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Training and Validation Accuracy')
ax.legend()

# Adjust the x-axis ticks
ax.set_xticks([e + bar_width / 2 for e in epochs])
ax.set_xticklabels(epochs)

# Add grid
ax.grid(True)

# Show the figure
plt.show()


#actual vs predicted
fig, ax = plt.subplots(figsize=(20, 6))

# Plotting the training loss and validation loss
ax.bar(l, y_test_actual, width=bar_width, color='grey', label='Actual')
ax.bar([e + bar_width for e in l], predicted, width=bar_width, color='red', label='predicted')

# Add labels and title
ax.set_xlabel('length')
ax.set_ylabel('Type')
ax.set_title('type')
ax.legend()

# Adjust the x-axis ticks
ax.set_xticks([e + bar_width / 2 for e in l])
ax.set_xticklabels(l)

max_value = max(max(y_test_actual), max(predicted))
min_value = min(min(y_test_actual), min(predicted))
ax.set_yticks(np.arange(min_value, max_value+1))
ax.set_yticklabels(np.arange(min_value, max_value+1))

# Add grid
ax.grid(True)

# Show the figure
plt.show()
