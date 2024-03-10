### Student Grade Prediction Model

This project includes Python scripts for training and evaluating a K-nearest neighbors (KNN) classifier to predict students' final grades based on various features. Additionally, it provides visualization of the predictions against the actual grades using a scatter plot.

### Files Included:

- `train_and_evaluate.py`: Python script for training and evaluating the KNN classifier.
- `visualize_model.py`: Python script for visualizing the predictions against the actual grades using a scatter plot.
- `requirements.txt`: Text file listing the required Python packages and their versions.
- `student-mat.csv`: Input dataset containing student attributes and final grades for training the prediction model.
- `best_model.pkl`: Trained model saved in a binary format.

### Prerequisites:

Ensure you have Python installed on your system. You can install the required packages using pip:

```
pip install -r requirements.txt
```

### Usage:

1. **Training and Evaluation**:
   - Run `train_and_evaluate.py` to train the KNN classifier on the student dataset and evaluate its performance.

2. **Visualization**:
   - After training the model, run `visualize_model.py` to visualize the predictions against the actual grades using a scatter plot.

### Requirements:

- pandas==1.3.3
- numpy==1.21.2
- scikit-learn==0.24.2
- matplotlib==3.4.3

### Data:
- `student-mat.csv`: Input dataset containing student attributes and final grades for training the prediction model.

### Trained Model:
- `best_model.pkl`: Trained KNN classifier model saved for future use.

### Note:
Ensure that you have the necessary permissions to access and load the input dataset (`student-mat.csv`) and the trained model (`best_model.pkl`).

### References:
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
