import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load the saved model from file
with open('best_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Load the data you want to predict on (assuming it's in a DataFrame)
data = pd.read_csv('student-mat.csv', sep=';')
data = data[['G1', 'G2', 'G3', 'studytime', 'absences']]
predict = 'G3'
x = np.array(data.drop([predict], axis=1))

# Make predictions using the loaded model
predictions = loaded_model.predict(x)

# Get a random subset of 100 data points
subset_indices = np.random.choice(range(len(data)), size=100, replace=False)
subset_data = data.iloc[subset_indices]
subset_predictions = predictions[subset_indices]
subset_actual = subset_data[predict]

# Create a scatter plot to visualize the subset of data
plt.scatter(subset_actual, subset_predictions, label='Predictions')
plt.scatter(subset_actual, subset_actual, label='Actual', color='red', marker='x')
plt.xlabel('Actual G3')
plt.ylabel('Predicted G3')
plt.title('Scatter Plot of Actual vs Predicted G3 (Subset of Data)')
plt.legend()
plt.show()
