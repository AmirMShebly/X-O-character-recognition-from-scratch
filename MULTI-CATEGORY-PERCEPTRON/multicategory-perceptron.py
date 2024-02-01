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


def split_dataset(dataset, split_ratio):
    split_idx = int(len(dataset) * split_ratio)
    train_set = dataset[:split_idx]
    test_set = dataset[split_idx:]
    return train_set, test_set


def step_function(x):
    return 1 if x >= 0 else -1


def predict(row, weights, bias):
    s1 = row.dot(weights[0]) + bias[0]
    s2 = row.dot(weights[1]) + bias[1]

    predictionX = step_function(s1)
    predictionO = step_function(s2) # not necessary

    return 'X' if predictionX == 1 else 'O'


def train_multi_category_perceptron(data, learning_rate, epochs):
    weights = np.array([[0.0] * 25, [0.0] * 25])
    bias = [0.0, 0.0]

    epoch_count = 0
    while epoch_count < epochs:
        epoch_count += 1
        for i in range(len(data)):
            s1 = data[i][:-1].dot(weights[0])
            s2 = data[i][:-1].dot(weights[1])

            predictionX = step_function(s1 + bias[0])
            predictionO = step_function(s2 + bias[1])

            if data[i][25] != predictionX:
                weights[0] += learning_rate * data[i][:-1] * data[i][25]
                bias[0] += learning_rate * data[i][25]

            if (data[i][25] * (-1)) != predictionO:
                weights[1] += learning_rate * data[i][:-1] * (data[i][25] * (-1))
                bias[1] += learning_rate * (data[i][25] * (-1))

    return weights, bias


def evaluate_algorithm(test_set, weights, bias):
    correct = 0
    for row in test_set:
        prediction = predict(np.array(row[:-1]), weights, bias)
        if row[-1] == 1 and prediction == 'X':
            correct += 1
        elif row[-1] == -1 and prediction == 'O':
            correct += 1
    return correct / float(len(test_set)) * 100.0


filename = 'training_set.csv'
dataset = load_dataset(filename)
train_set, test_set = split_dataset(dataset, 0.8)
learning_rate = 0.001
epochs = 100

weights, biases = train_multi_category_perceptron(train_set, learning_rate, epochs)

accuracy = evaluate_algorithm(test_set, weights, biases)
print(f'Accuracy on the test set: {accuracy:.2f}%')


def classify_shape():
    global current_data, weights, biases
    data = np.array(current_data)
    prediction = predict(data, weights, biases)
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

