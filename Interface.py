import tkinter as tk
from PIL import Image, ImageTk
from Testing import openWebcam

class RadioButtonApp:
    def __init__(self, root, image_path, button_positions):
        self.root = root
        self.root.title("Facial Landmark Choosing")

        self.image = Image.open(image_path)
        self.photo = ImageTk.PhotoImage(self.image)

        self.canvas = tk.Canvas(root, width=self.image.width, height=self.image.height)
        self.canvas.pack()

        self.radio_buttons = []
        for i, position in enumerate(button_positions, 1):
            x, y = position
            var = tk.IntVar()
            radio_button = tk.Radiobutton(self.canvas, text=str(i), variable=var, value=i, command=lambda v=var, idx=i: self.on_button_click(v, idx))
            self.radio_buttons.append((radio_button, x, y, var))
            self.canvas.create_window(x, y, window=radio_button, anchor=tk.NW)

        self.canvas.create_image(10, -70, anchor=tk.NW, image=self.photo)

        self.chosen_points = []


        # Add "Submit" button inside the canvas
        self.submit_button = tk.Button(self.canvas, text="Submit", command=self.submit_data)
        self.submit_button_window = self.canvas.create_window(10, 10, window=self.submit_button, anchor=tk.NW)

    def on_button_click(self, var, idx):
        selected_value = var.get()
        if selected_value not in self.chosen_points:
            self.chosen_points.append(selected_value)
            print(f"{self.chosen_points}")




    def submit_data(self):
        # Write the chosen points as a Python list to a .txt file
        with open("chosen_points.txt", "w") as file:
            file.write(str(self.chosen_points))

        # Read the list from the .txt file
        with open("chosen_points.txt", "r") as file:
            selected_landmark_indices = eval(file.read())
        # Run the second code with the selected landmark indices
        openWebcam(selected_landmark_indices)


if __name__ == "__main__":
    # Specify the image path and button positions
    image_path = "/Users/homa/Desktop/Face Detection And Landmarking With Homa Taherpour/resized_image 19.38.45.jpeg"
    button_positions = [
        (73, 132),  # 1
        (72, 178),  # 2
        (72, 215),  # 3
        (74, 261),  # 4
        (83, 298),  # 5
        (92, 338),  # 6
        (107, 380),  # 7
        (123, 418),  # 8
        (135, 455),  # 9
        (153, 487),  # 10
        (177, 522),  # 11
        (200, 547),  # 12
        (230, 579),  # 13
        (260, 606),  # 14
        (292, 631),  # 15
        (330, 649),  # 16
        (378, 658),  # 17
        (429, 649),  # 18
        (475, 634),  # 19
        (514, 606),  # 20
        (544, 579),  # 21
        (574, 555),  # 22
        (600, 518),  # 23
        (623, 487),  # 24
        (641, 448),  # 25
        (654, 407),  # 26
        (667, 369),  # 27
        (671, 329),  # 28
        (680, 288),  # 29
        (686, 251),  # 30
        (691, 209),  # 31
        (691, 167),  # 32
        (111, 96),  # 33
        (154, 71),  # 34
        (200, 69),  # 35
        (251, 78),  # 36
        (295, 97),  # 37
        (290, 121),  # 38
        (244, 107),  # 39
        (195, 101),  # 40
        (154, 97),  # 41
        (422, 89),  # 42
        (472, 70),  # 43
        (528, 60),  # 44
        (581, 57),  # 45
        (632, 86),  # 46
        (579, 88),  # 47
        (530, 92),  # 48
        (481, 102),  # 49
        (430, 116),  # 50
        (363, 169),  # 51
        (360, 235),  # 52
        (360, 297),  # 53
        (361, 357),  # 54
        (321, 190),  # 55
        (306, 322),  # 56
        (289, 374),  # 57
        (312, 390),  # 58
        (338, 396),  # 59
        (366, 407),  # 60
        (395, 394),  # 61
        (421, 385),  # 62
        (446, 366),  # 63
        (428, 315),  # 64
        (408, 187),  # 65
        (167, 176),  # 66
        (190, 157),  # 67
        (223, 152),  # 68
        (260, 162),  # 69
        (286, 189),  # 70
        (255, 195),  # 71
        (223, 198),  # 72
        (190, 190),  # 73
        (190, 193),  # 74
        (450, 189),  # 75
        (478, 157),  # 76
        (514, 143),  # 77
        (549, 148),  # 78
        (577, 170),  # 79
        (552, 183),  # 80
        (525, 188),  # 81
        (486, 187),  # 82
        (486, 187),  # 83
        (274, 477),  # 84
        (308, 467),  # 85
        (349, 457),  # 86
        (375, 464),  # 87
        (402, 455),  # 88
        (445, 464),  # 89
        (493, 472),  # 90
        (461, 504),  # 91
        (425, 525),  # 92
        (380, 535),  # 93
        (335, 528),  # 94
        (302, 507),  # 95
        (298, 480),  # 96
        (328, 484),  # 97
        (378, 487),  # 98
        (431, 482),  # 99
        (470, 475),  # 100
        (439, 494),  # 101
        (382, 500),  # 102
        (329, 493),  # 103
        (230, 172),  # 104
        (517, 164)  # 105
    ]

    # Create the Tkinter window
    root = tk.Tk()

    # Create and run the application
    app = RadioButtonApp(root, image_path, button_positions)
    root.mainloop()
