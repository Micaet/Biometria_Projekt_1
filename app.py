import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tkinter import ttk
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class BiometriaApp:
    def __init__(self, root):
        self.main_container = None
        self.brightness_slider = None
        self.notebook = None
        self.page_hist = None
        self.page_image = None
        self.root = root
        self.root.title("Biometria - Projekt 1")
        self.root.geometry("1100x700")
        self.is_szarosc = False
        self.original_np = None
        self.base_np = None
        self.processed_np = None
        self.brightness_value = 0
        self.hist_frame = None
        self.setup_ui()


    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.page_image = tk.Frame(self.notebook)
        self.page_hist = tk.Frame(self.notebook)

        self.notebook.add(self.page_image, text="Obraz")
        self.notebook.add(self.page_hist, text="Histogram")
        self.main_container = tk.Frame(self.page_image)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.hist_frame = tk.Frame(self.page_hist)
        self.hist_frame.pack(fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(self.main_container, width=220, bg="#e0e0e0")
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(sidebar, text="Menu Projektu", font=("Arial", 11, "bold"), bg="#e0e0e0").pack(pady=10)

        tk.Button(sidebar, text="Wczytaj Obraz", command=self.load_image).pack(fill=tk.X, padx=10, pady=2)
        tk.Button(sidebar, text="Zapisz Obraz", command=self.save_image).pack(fill=tk.X, padx=10, pady=2)
        tk.Button(sidebar, text="Resetuj do oryginału", command=self.reset_image).pack(fill=tk.X, padx=10, pady=2)

        ttk.Separator(sidebar).pack(fill=tk.X, pady=10)

        tk.Label(sidebar, text="Jasność", bg="#e0e0e0").pack(pady=5)
        self.brightness_slider = tk.Scale(sidebar, from_=-255, to=255, orient=tk.HORIZONTAL,
                                          command=self.update_brightness)
        self.brightness_slider.pack(fill=tk.X, padx=10, pady=5)
        # Dodaj w sidebar:
        tk.Button(sidebar, text="Logarytm", command=self.log_transform).pack(fill=tk.X, padx=10, pady=2)
        tk.Button(sidebar, text="Pierwiastek (Gamma 0.5)", command=lambda: self.gamma_correction(0.5)).pack(fill=tk.X,
                                                                                                            padx=10,
                                                                                                            pady=2)
        tk.Button(sidebar, text="Potęga (Gamma 2.0)", command=lambda: self.gamma_correction(2.0)).pack(fill=tk.X,
                                                                                                       padx=10, pady=2)
        tk.Button(sidebar, text="Szarość", command=self.color_to_grayscale).pack(fill=tk.X, padx=10, pady=2)
        tk.Button(sidebar, text="Negatyw", command=self.negatyw).pack(fill=tk.X, padx=10, pady=2)
        tk.Button(sidebar, text="Binaryzacja Prosta", command=self.binaryzacja).pack(fill=tk.X, padx=10, pady=2)
        tk.Label(sidebar, text="Filtry", font=("Arial", 10, "bold"), bg="#e0e0e0").pack(pady=10)

        filter_button = tk.Menubutton(sidebar, text="Filtry liniowe ▼", relief=tk.RAISED)
        filter_button.pack(fill=tk.X, padx=10, pady=2)

        filter_menu = tk.Menu(filter_button, tearoff=0)
        filter_button.config(menu=filter_menu)

        filter_menu.add_command(label="Średni (Average)", command=self.filter_average)
        filter_menu.add_command(label="Gaussowski", command=self.filter_gaussian)
        filter_menu.add_command(label="Wyostrzający (Sharpen)", command=self.filter_sharpen)

        tk.Button(sidebar, text="Histogram",
                  command=lambda: self.notebook.select(self.page_hist)
                  ).pack(fill=tk.X, padx=10, pady=2)

        btns = ["Projekcje H/V", "Krawędzie"]
        for txt in btns:
            tk.Button(sidebar, text=txt, command=lambda t=txt: self.not_implemented(t)).pack(fill=tk.X, padx=10, pady=1)

        self.canvas = tk.Canvas(self.main_container, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.canvas.bind("<Configure>", self.on_resize)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.png *.bmp *.jpeg")])
        if path:
            img = Image.open(path).convert("RGB")
            self.original_np = np.array(img)
            self.base_np = self.original_np.copy()
            self.processed_np = self.original_np.copy()
            self.brightness_slider.set(0)
            self.update_display()

    def reset_image(self):
        if self.original_np is not None:
            self.base_np = self.original_np.copy()
            self.brightness_slider.set(0)
            self.apply_modifications()

    def color_to_grayscale(self):
        if self.base_np is not None:
            self.is_szarosc = True
            if len(self.base_np.shape) == 3:
                self.base_np = np.dot(self.base_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                self.apply_modifications()
        else:
            messagebox.showwarning("Błąd", "Najpierw wczytaj obraz.")

    def update_brightness(self, value):
        self.brightness_value = int(value)
        self.apply_modifications()

    def apply_modifications(self):
        if self.base_np is not None:
            self.processed_np = np.clip(self.base_np.astype(np.int16) + self.brightness_value, 0, 255).astype(np.uint8)
            self.update_display()
            self.update_histogram()

    def negatyw(self):
        if self.base_np is not None:
            self.base_np = 255 - self.base_np
            self.apply_modifications()

    def gamma_correction(self, gamma=0.5):
        if self.base_np is not None:
            img_float = self.base_np.astype(np.float32)
            img_normalized = img_float / 255.0
            gamma_img = np.power(img_normalized, gamma)
            self.base_np = (gamma_img * 255).astype(np.uint8)
            self.apply_modifications()

    def log_transform(self):
        if self.base_np is not None:
            img_float = self.base_np.astype(np.float32)
            c = 255 / np.log(1 + np.max(img_float))
            log_img = c * (np.log(1 + img_float))
            self.base_np = log_img.astype(np.uint8)
            self.apply_modifications()

    def binaryzacja(self):
        if self.base_np is not None and self.is_szarosc:
            threshold = 128
            self.base_np = np.where(self.base_np > threshold, 255, 0).astype(np.uint8)
            self.apply_modifications()
        elif self.base_np is not None:
            messagebox.showwarning("Błąd", "Binaryzacja wymaga obrazu w skali szarości. Najpierw kliknij 'Szarość'.")
        else:
            messagebox.showwarning("Błąd", "Najpierw wczytaj obraz.")

    def update_display(self):
        if self.processed_np is None: return

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 2 or canvas_h < 2: return

        img_full = Image.fromarray(self.processed_np)

        img_w, img_h = img_full.size
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        new_w, new_h = int(img_w * ratio), int(img_h * ratio)

        img_resized = img_full.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.display_img = ImageTk.PhotoImage(img_resized)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.display_img)

    def apply_filter(self, filter_type):
        if self.base_np is None: return
        kernels = {
            "average": np.ones((3, 3)) / 9.0,
            "gaussian": np.array([[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]]) / 16.0,
            "sharpen": np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])
        }
        kernel = kernels[filter_type]
        if len(self.base_np.shape) == 3:
            channels = []
            for i in range(3):
                channels.append(convolve(self.base_np[:, :, i], kernel))
            self.base_np = np.stack(channels, axis=2).astype(np.uint8)
        else:
            self.base_np = convolve(self.base_np, kernel).astype(np.uint8)

        self.apply_modifications()

    def filter_average(self):
        self.apply_filter("average")

    def filter_gaussian(self):
        self.apply_filter("gaussian")

    def filter_sharpen(self):
        self.apply_filter("sharpen")

    def save_image(self):
        if self.processed_np is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
            if path:
                Image.fromarray(self.processed_np).save(path)
                messagebox.showinfo("OK", "Zapisano pomyślnie!")

    def on_resize(self, event):
        self.update_display()

    def update_histogram(self):

        if self.processed_np is None:
            if self.base_np is not None:
                self.processed_np = self.base_np
            else:
                return

        for widget in self.hist_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots()

        if len(self.processed_np.shape) == 3:
            ax.hist(self.processed_np[:, :, 0].flatten(), bins=256,
                    color='red', alpha=0.4, label='R')
            ax.hist(self.processed_np[:, :, 1].flatten(), bins=256,
                    color='green', alpha=0.4, label='G')
            ax.hist(self.processed_np[:, :, 2].flatten(), bins=256,
                    color='blue', alpha=0.4, label='B')
            ax.legend()
        else:
            ax.hist(self.processed_np.flatten(), bins=256, color='gray')

        canvas = FigureCanvasTkAgg(fig, master=self.hist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def not_implemented(self, name):
        messagebox.showinfo("Zadanie", f"Tu zaimplementuj ręcznie algorytm: {name}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BiometriaApp(root)
    root.mainloop()
