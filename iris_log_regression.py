from sklearn.datasets import load_iris
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import train_test_split


#Load the IRIS dataset
iris = load_iris()
X = iris.data
y = iris.target

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize the Logostic Regresion Model
model = LogisticRegression(max_iter=1000)

#Train the model
model.fit(X_train, y_train)

#Predict the Test set
y_pred = model.predict(X_test)
#print(y_pred)
#print(model.score(X_test, y_test))

# Now, let's predict some new samples
new_samples = [
    [5.1, 3.5, 1.4, 0.2],  # Setosa
    [6.2, 2.9, 4.3, 1.3],  # Versicolour
    [7.3, 2.8, 6.4, 2.0]   # Virginica
]
pred_class = model.predict(new_samples)
#print(pred_class)

# Map predicted classes to actual class labels
predicted_class_labels = [iris.target_names[i] for i in pred_class]

print("Predicted classes for new samples:")
for sample, predicted_class in zip(new_samples, predicted_class_labels):
    print("Features:", sample, "-> Predicted class:", predicted_class)
