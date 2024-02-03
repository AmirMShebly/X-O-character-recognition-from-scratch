import csv
import numpy as np
import tkinter as tk


def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            dataset.append([int(cell) for cell in row])
    return np.array(dataset)


def train_test_split(dataset, test_size=0.2):
    train_size = int(len(dataset) * (1 - test_size))
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]

    train_inputs = train_data[:, :-1]
    train_outputs = np.array([[1, 0] if label == 1 else [0, 1] for label in train_data[:, -1]])

    test_inputs = test_data[:, :-1]
    test_outputs = np.array([[1, 0] if label == 1 else [0, 1] for label in test_data[:, -1]])

    return train_inputs, test_inputs, train_outputs, test_outputs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def fit(inputs, outputs, hidden_size, epochs, learning_rate):
    input_size = inputs.shape[1]
    output_size = outputs.shape[1]

    hidden_weights = np.random.rand(input_size, hidden_size)
    hidden_bias = np.random.rand(1, hidden_size)
    output_weights = np.random.rand(hidden_size, output_size)
    output_bias = np.random.rand(1, output_size)

    for epoch in range(epochs):
        # Forward propagation
        hidden_activation = sigmoid(np.dot(inputs, hidden_weights) + hidden_bias)
        output_activation = sigmoid(np.dot(hidden_activation, output_weights) + output_bias)

        # Backpropagation
        output_error = outputs - output_activation
        output_delta = output_error * sigmoid_derivative(output_activation)
        hidden_error = output_delta.dot(output_weights.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_activation)

        # Update weights and biases
        output_weights += hidden_activation.T.dot(output_delta) * learning_rate
        output_bias += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        hidden_weights += inputs.T.dot(hidden_delta) * learning_rate
        hidden_bias += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    return hidden_weights, hidden_bias, output_weights, output_bias


def predict(row, hidden_weights, hidden_bias, output_weights, output_bias):
    if len(row) != len(hidden_weights):
        row = np.append(row, 1)

    hidden_activation = sigmoid(np.dot(row, hidden_weights) + hidden_bias)
    output_activation = sigmoid(np.dot(hidden_activation, output_weights) + output_bias)

    return 'X' if output_activation[0, 0] > 0.5 else 'O'


def evaluate_algorithm(test_inputs, test_outputs, hidden_weights, hidden_bias, output_weights, output_bias):
    n = 0
    correct = 0
    for i in range(len(test_inputs)):
        data = test_inputs[i]
        target = test_outputs[i]

        prediction = predict(data, hidden_weights, hidden_bias, output_weights, output_bias)
        if target[0] == 1 and prediction == 'X':
            correct += 1
        elif target[0] == 0 and prediction == 'O':
            correct += 1

    return correct / len(test_inputs) * 100.0


filename = 'training_set.csv'
dataset = load_dataset(filename)
X_train, X_test, y_train, y_test = train_test_split(dataset)


hidden_layer_size = 5
epochs = 1000
learning_rate = 0.01

hidden_weights, hidden_bias, output_weights, output_bias = fit(
X_train, y_train, hidden_layer_size, epochs, learning_rate)

accuracy = evaluate_algorithm(X_test, y_test, hidden_weights, hidden_bias, output_weights, output_bias)
print(f'Accuracy on the test set: {accuracy:.2f}%')


def classify_shape():
    global current_data, hidden_weights, hidden_bias, output_weights, output_bias
    data = np.array(current_data[:-1])
    prediction = predict(data, hidden_weights, hidden_bias, output_weights, output_bias)
    result_label.config(text=f"The shape is: {prediction}")


def toggle_button(button_index):
    if buttons[button_index]['bg'] == 'white':
        buttons[button_index].configure(bg='green')
        current_data[button_index] = 1
    else:
        buttons[button_index].configure(bg='white')
        current_data[button_index] = -1


def clear_grid():
    for i in range(25):
        buttons[i].configure(bg='white')
        current_data[i] = -1


root = tk.Tk()
root.title("Shape Classifier")

buttons = []
current_data = [-1 for _ in range(25)]

for i in range(25):
    row = i // 5
    col = i % 5
    button = tk.Button(root, width=4, height=2, bg='white', command=lambda i=i: toggle_button(i))
    button.grid(row=row, column=col)
    buttons.append(button)

classify_button = tk.Button(root, text='Classify', command=classify_shape)
classify_button.grid(row=5, column=0, columnspan=5)

clear_button = tk.Button(root, text='Clear', command=clear_grid)
clear_button.grid(row=6, column=0, columnspan=5)

result_label = tk.Label(root, text="")
result_label.grid(row=7, column=0, columnspan=5)

root.mainloop()
