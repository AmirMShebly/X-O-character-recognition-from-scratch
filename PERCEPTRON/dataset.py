import tkinter as tk
import csv

current_data = []
csv_filename = 'training_set.csv'
training_data = []

def toggle_button(button_index):
    if not training_finished:
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

def save_shape(label):
    global training_data
    shape_data = current_data + [label]

    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            header = [f'cell_{i}' for i in range(1, 26)] + ['label']
            writer.writerow(header)

        writer.writerow(shape_data)

    clear_grid()
    print(f"Training data saved for shape with label: {label}")

def finish_train():
    global training_finished
    training_finished = True

def train_clicked():
    global training_finished
    training_finished = False
    print("Please click on the cells to form a shape, then specify 'X' or 'O'.")

def specify_shape_x():
    save_shape(1)

def specify_shape_o():
    save_shape(-1)

root = tk.Tk()
root.title("Specify Shape as 'X' or 'O'")

buttons = []
for i in range(25):
    row = i // 5
    col = i % 5
    button = tk.Button(root, width=4, height=2, bg='white', command=lambda i=i: toggle_button(i))
    button.grid(row=row, column=col)
    buttons.append(button)
    current_data.append(-1)


train_button = tk.Button(root, text="Train", command=train_clicked)
train_button.grid(row=5, column=0, columnspan=2)

x_button = tk.Button(root, text="X", command=specify_shape_x)
x_button.grid(row=5, column=2)

o_button = tk.Button(root, text="O", command=specify_shape_o)
o_button.grid(row=5, column=3, columnspan=2)

training_finished = False

finish_train_button = tk.Button(root, text="Finish Train", command=finish_train)
finish_train_button.grid(row=6, column=0, columnspan=5)

root.mainloop()
