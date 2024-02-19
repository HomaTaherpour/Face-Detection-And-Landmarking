import tkinter as tk
from PIL import Image, ImageTk

class ImageClicker:
    def __init__(self, root, image_path):
        self.root = root
        self.image_path = image_path

        self.image = Image.open(self.image_path)
        self.photo = ImageTk.PhotoImage(self.image)

        self.label = tk.Label(root, image=self.photo)
        self.label.pack()

        self.label.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        x, y = event.x, event.y
        width, height = self.image.size
        scale_factor = self.photo.width() / width

        x_in_image = int(x / scale_factor)
        y_in_image = int(y / scale_factor)

        print(f"({x_in_image}, {y_in_image})")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Image Clicker")

    image_path = "/Users/homa/Desktop/Face detection and landmarking Homa Taherpour/resized_image.jpg"
    clicker = ImageClicker(root, image_path)

    root.mainloop()