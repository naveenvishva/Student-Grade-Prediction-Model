import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load the data
data = pd.read_csv('student-mat.csv', sep=';')
data = data[['G1', 'G2', 'G3', 'studytime', 'absences']]
predict = 'G3'
x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# Train the model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

# Evaluate the model
accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy)

# Save the trained model to a file
with open('best_model.pkl', 'wb') as f:
    pickle.dump(model, f)
