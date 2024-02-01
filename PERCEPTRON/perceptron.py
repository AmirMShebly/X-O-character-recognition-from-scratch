import csv
import random
import tkinter as tk


def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            dataset.append([int(cell) for cell in row[:-1]] + [int(row[-1])])
    return dataset


def split_dataset(dataset, split_ratio):
    split_idx = int(len(dataset) * split_ratio)
    train_set = dataset[:split_idx]
    test_set = dataset[split_idx:]
    return train_set, test_set


def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1 if activation >= 0 else -1


def train_weights(train, learning_rate, n_epochs):
    weights = [0.0 for _ in range(len(train[0]))]
    for _ in range(n_epochs):
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            weights[0] += learning_rate * error
            for i in range(len(row)-1):
                weights[i + 1] += learning_rate * error * row[i]
    return weights


def evaluate_algorithm(test, weights):
    correct = 0
    for row in test:
        prediction = predict(row, weights)
        if row[-1] == prediction:
            correct += 1
    return correct / float(len(test)) * 100.0


filename = 'training_set.csv'
dataset = load_dataset(filename)
train_set, test_set = split_dataset(dataset, 0.8)
learning_rate = 0.01
n_epochs = 100
weights = train_weights(train_set, learning_rate, n_epochs)

accuracy = evaluate_algorithm(test_set, weights)
print(f'Accuracy on the test set: {accuracy:.2f}%')


def classify_shape():
    global current_data, weights
    prediction = predict(current_data, weights)
    result_label.config(text=f"The shape is: {'X' if prediction == 1 else 'O'}")


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
