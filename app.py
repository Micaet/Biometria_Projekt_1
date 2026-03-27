import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tkinter import ttk
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import simpledialog
class BiometriaApp:
    def __init__(self, root):
        self.edge_threshold = None
        self.main_container = None
        self.brightness_slider = None
        self.notebook = None
        self.page_hist = None
        self.page_image = None
        self.root = root
        self.root.title("Biometria - Projekt 1")
        self.root.geometry("1100x800")
        self.is_szarosc = False
        self.original_np = None
        self.base_np = None
        self.processed_np = None
        self.brightness_value = 0
        self.hist_frame = None
        self.hist_figure = None
        self.page_proj = None
        self.page_edges = None
        self.edges_np = None
        self.grad_np = None
        self.last_edge_method = 'Krzyż Robertsa'
        self.setup_ui()


    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.page_image = tk.Frame(self.notebook)
        self.page_hist = tk.Frame(self.notebook)
        self.page_proj = tk.Frame(self.notebook)
        self.page_edges = tk.Frame(self.notebook)


        self.notebook.add(self.page_image, text="Obraz")
        self.notebook.add(self.page_hist, text="Histogram")
        self.notebook.add(self.page_proj, text="Projekcje")
        self.notebook.add(self.page_edges, text="Krawędzie")
        self.main_container = tk.Frame(self.page_image)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.hist_figure = None

        self.hist_frame = tk.Frame(self.page_hist)
        self.hist_frame.pack(fill=tk.BOTH, expand=True)

        self.proj_frame = tk.Frame(self.page_proj)
        self.proj_frame.pack(fill=tk.BOTH, expand=True)
        self.proj_figure = None

        self.edges_frame = tk.Frame(self.page_edges)
        self.edges_frame.pack(fill=tk.BOTH, expand=True)

        self.edges_toolbar = tk.Frame(self.edges_frame, bg="#d0d0d0", height=40)
        self.edges_toolbar.pack(fill=tk.X)

        tk.Button(self.edges_toolbar, text="Zapisz krawędzie",
                  command=self.save_edges).pack(side=tk.LEFT, padx=10, pady=5)

        self.edge_label = tk.Label(self.edges_toolbar, text="Brak danych", bg="#d0d0d0")
        self.edge_label.pack(side=tk.LEFT, padx=10)
        tk.Button(self.edges_toolbar, text="Odśwież",
                  command=lambda: self.show_edges(self.last_edge_method)).pack(side=tk.LEFT, padx=5)

        self.edges_canvas = tk.Canvas(self.edges_frame, bg="#2b2b2b")
        self.edges_canvas.pack(fill=tk.BOTH, expand=True)

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
        tk.Button(sidebar, text="Logarytm", command=self.log_transform).pack(fill=tk.X, padx=10, pady=2)
        tk.Button(sidebar, text="Pierwiastek (Gamma 0.5)", command=lambda: self.gamma_correction(0.5)).pack(fill=tk.X,
                                                                                                            padx=10,
                                                                                                            pady=2)
        tk.Button(sidebar, text="Potęga (Gamma 2.0)", command=lambda: self.gamma_correction(2.0)).pack(fill=tk.X,
                                                                                                       padx=10, pady=2)

        greyscale_button = tk.Menubutton(sidebar, text="Szarość ▼", relief=tk.RAISED)
        greyscale_button.pack(fill=tk.X, padx=10, pady=2)

        greyscale_menu = tk.Menu(greyscale_button, tearoff=0)
        greyscale_button.config(menu=greyscale_menu)

        greyscale_menu.add_command(label="Średnia (Average)", command=lambda: self.greyscale('Srednia'))
        greyscale_menu.add_command(label="Luminacja", command=lambda: self.greyscale('Luminacja'))
        greyscale_menu.add_command(label="Lightness", command=lambda: self.greyscale('Lightness'))
        greyscale_menu.add_command(label="PCA", command=lambda: self.greyscale('PCA'))
        greyscale_menu.add_command(label="Dobierz wagi", command=lambda: self.greyscale('Custom'))







        tk.Button(sidebar, text="Negatyw", command=self.negatyw).pack(fill=tk.X, padx=10, pady=2)
        bin_button = tk.Menubutton(sidebar, text="Binaryzacja ▼", relief=tk.RAISED)
        bin_button.pack(fill=tk.X, padx=10, pady=2)

        bin_menu = tk.Menu(bin_button, tearoff=0)
        bin_button.config(menu=bin_menu)

        bin_menu.add_command(label="Ręczny próg (Suwak)", command=self.binaryzacja_manual)
        bin_menu.add_command(label="Metoda Otsu (Auto)", command=self.binaryzacja_otsu)
        bin_menu.add_command(label="Progowanie lokalne (Średnia)", command=self.binaryzacja_local)
        tk.Label(sidebar, text="Filtry", font=("Arial", 10, "bold"), bg="#e0e0e0").pack(pady=10)

        filter_button = tk.Menubutton(sidebar, text="Filtry liniowe ▼", relief=tk.RAISED)
        filter_button.pack(fill=tk.X, padx=10, pady=2)
        morph_button = tk.Menubutton(sidebar, text="Morfologia ▼", relief=tk.RAISED)
        morph_button.pack(fill=tk.X, padx=10, pady=2)

        morph_menu = tk.Menu(morph_button, tearoff=0)
        morph_button.config(menu=morph_menu)

        morph_menu.add_command(label="Erozja", command=lambda: self.manual_morphology("erosion"))
        morph_menu.add_command(label="Dylatacja", command=lambda: self.manual_morphology("dilation"))
        morph_menu.add_command(label="Otwarcie", command=lambda: self.manual_morphology("opening"))
        morph_menu.add_command(label="Zamknięcie", command=lambda: self.manual_morphology("closing"))
        filter_menu = tk.Menu(filter_button, tearoff=0)
        filter_button.config(menu=filter_menu)

        filter_menu.add_command(label="Średni (Average)", command=self.filter_average)
        filter_menu.add_command(label="Gaussowski", command=self.filter_gaussian)
        filter_menu.add_command(label="Wyostrzający (Sharpen)", command=self.filter_sharpen)
        filter_menu.add_command(label="Custom", command=self.filter_from_string)

        tk.Button(sidebar, text="Histogram",
                  command=self.show_histogram).pack(fill=tk.X, padx=10, pady=2)
        tk.Button(sidebar, text="Projekcje H/V", command=self.show_projections).pack(fill=tk.X, padx=10, pady=2)

        

        edge_button_edgebar = tk.Menubutton(self.edges_toolbar, text="Krawędzie ▼", relief=tk.RAISED)
        edge_button_edgebar.pack(fill=tk.X, padx=10, pady=2)

        edge_menu_edgebar = tk.Menu(edge_button_edgebar, tearoff=0)
        edge_button_edgebar.config(menu=edge_menu_edgebar)

        edge_menu_edgebar.add_command(label="Krzyż Robertsa", command=lambda: self.show_edges("Krzyż Robertsa"))
        edge_menu_edgebar.add_command(label="Operator Sobela", command=lambda: self.show_edges("Operator Sobela"))

        self.edge_threshold = tk.Scale(self.edges_toolbar, from_=0, to=255, orient=tk.HORIZONTAL)
        self.edge_threshold.set(100)
        self.edge_threshold.pack(fill=tk.X, padx=10)

        self.canvas = tk.Canvas(self.main_container, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.canvas.bind("<Configure>", self.on_resize)

    def on_tab_changed(self, event):
        selected_tab = event.widget.select()
        tab_widget = event.widget.nametowidget(selected_tab)

        if tab_widget == self.page_hist:
            self.update_histogram()
        elif tab_widget == self.page_proj:
            self.update_projections()

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

    def greyscale(self, type_of_alg):
        if self.base_np is None:
            messagebox.showwarning("Błąd", "Najpierw wczytaj obraz.")
            return

        if len(self.base_np.shape) != 3:
            messagebox.showinfo("Info", "Obraz jest już w skali szarości.")
            return

        self.is_szarosc = True
        img = self.base_np.astype(np.float32)

        if type_of_alg == "Srednia":
            gray = np.mean(img, axis=2)

        elif type_of_alg == "Luminacja":
            gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

        elif type_of_alg == "Lightness":
            gray = (np.max(img, axis=2) + np.min(img, axis=2)) / 2

        elif type_of_alg == "PCA":
            pixels = img.reshape(-1, 3)

            mean = np.mean(pixels, axis=0)
            pixels_centered = pixels - mean

            cov = np.cov(pixels_centered, rowvar=False)

            eigvals, eigvecs = np.linalg.eigh(cov)

            principal_component = eigvecs[:, np.argmax(eigvals)]

            gray = pixels_centered @ principal_component
            gray = gray.reshape(img.shape[:2])
            gray = (gray - gray.min()) / (gray.max() - gray.min()) * 255

        elif type_of_alg == "Custom":
            r = simpledialog.askfloat("Waga R", "Podaj wagę dla R:", parent=self.root)
            g = simpledialog.askfloat("Waga G", "Podaj wagę dla G:", parent=self.root)
            b = simpledialog.askfloat("Waga B", "Podaj wagę dla B:", parent=self.root)

            if r is None or g is None or b is None:
                return

            suma = r + g + b
            if suma == 0:
                messagebox.showerror("Błąd", "Suma wag nie może być zerowa!")
                return

            gray = (r * img[..., 0] + g * img[..., 1] + b * img[..., 2]) / suma

        else:
            return

        self.base_np = np.clip(gray, 0, 255).astype(np.uint8)
        self.apply_modifications()

    def update_brightness(self, value):
        self.brightness_value = int(value)
        self.apply_modifications()

    def apply_modifications(self):
        if self.base_np is not None:
            self.processed_np = np.clip(self.base_np.astype(np.int16) + self.brightness_value, 0, 255).astype(np.uint8)
            self.update_display()

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

    def apply_filter(self, filter_type,kernel_size=3):
        if self.base_np is None: return
        kernels = {
            "average": np.ones((3, 3)) / 9.0,
            "gaussian": np.array([[1, 2, 1],
                                  [2, 4, 2],
                                  [1, 2, 1]]) / 16.0,
            "sharpen": np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]]),
            "custom": np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

        }
        if filter_type == "custom":
            tk
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
        self.apply_filter("average",3)

    def filter_gaussian(self):
        self.apply_filter("gaussian",3)

    def filter_sharpen(self):
        self.apply_filter("sharpen",3)

    def filter_custom(self):
        if self.base_np is None:
            messagebox.showwarning("Błąd", "Najpierw wczytaj obraz.")
            return
        h, w = self.base_np.shape[:2]
        max_size = 5

        k = simpledialog.askinteger("Rozmiar filtra",
                                    f"Podaj rozmiar okienka k (1 - 5):",
                                    parent=self.root,
                                    minvalue=1,
                                    maxvalue=max_size)

        if k:
            self.apply_filter("custom", k)

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
            return

        for widget in self.hist_frame.winfo_children():
            widget.destroy()

        if self.hist_figure is not None:
            plt.close(self.hist_figure)

        self.hist_figure, ax = plt.subplots()

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

        canvas = FigureCanvasTkAgg(self.hist_figure, master=self.hist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_histogram(self):
        self.notebook.select(self.page_hist)
        self.update_histogram()

    def update_projections(self):
        if self.processed_np is None:
            return

        for widget in self.proj_frame.winfo_children():
            widget.destroy()
        if self.proj_figure is not None:
            plt.close(self.proj_figure)

        if len(self.processed_np.shape) == 3:
            gray = np.mean(self.processed_np, axis=2)
        else:
            gray = self.processed_np

        hor_proj = np.sum(gray, axis=1)
        ver_proj = np.sum(gray, axis=0)

        self.proj_figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        self.proj_figure.tight_layout(pad=3.0)

        ax1.plot(hor_proj, color='blue')
        ax1.set_title("Projekcja Horyzontalna (Suma wierszy)")

        ax2.plot(ver_proj, color='green')
        ax2.set_title("Projekcja Wertykalna (Suma kolumn)")

        canvas = FigureCanvasTkAgg(self.proj_figure, master=self.proj_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_projections(self):
        self.notebook.select(self.page_proj)
        self.update_projections()

    def filter_from_string(self):
        if self.base_np is None:
            messagebox.showwarning("Błąd", "Najpierw wczytaj obraz.")
            return

        k = simpledialog.askinteger("Rozmiar", "Podaj rozmiar boku macierzy (k):",
                                    parent=self.root, minvalue=2, maxvalue=10)
        if not k: return

        input_str = simpledialog.askstring("Wartości",
                                           f"Wpisz {k * k} wartości oddzielonych spacją:",
                                           parent=self.root)
        if not input_str: return

        try:

            values = [float(x) for x in input_str.split()]

            if len(values) != k * k:
                messagebox.showerror("Błąd", f"Podałeś {len(values)} liczb, a potrzeba dokładnie {k * k}!")
                return

            kernel = np.array(values).reshape((k, k))

            suma = np.sum(kernel)
            if suma != 0:
                kernel = kernel / suma

            self.apply_custom_kernel(kernel)

        except ValueError:
            messagebox.showerror("Błąd", "Wprowadź poprawne liczby (używaj kropki jako separatora dziesiętnego)!")

    def apply_custom_kernel(self, kernel):
        if len(self.base_np.shape) == 3:
            channels = [convolve(self.base_np[:, :, i], kernel) for i in range(3)]
            self.base_np = np.stack(channels, axis=2).astype(np.uint8)
        else:
            self.base_np = convolve(self.base_np, kernel).astype(np.uint8)
        self.apply_modifications()

    def find_edges(self, kernel_type):
        if self.base_np is None:
            return

        kernels = {
            'Krzyż Robertsa': [
                np.array([[1, 0], [0, -1]]),
                np.array([[0, 1], [-1, 0]])
            ],
            'Operator Sobela': [
                np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
                np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            ],
        }

        kernel = kernels[kernel_type]

        if len(self.base_np.shape) == 3:
            greyscale = np.dot(self.base_np[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            greyscale = self.base_np.astype(np.float32)

        greyscale = greyscale.astype(np.float32)

        kernel_x = convolve(greyscale, kernel[0])
        kernel_y = convolve(greyscale, kernel[1])

        grad = np.sqrt(kernel_x ** 2 + kernel_y ** 2)

        if grad.max() != 0:
            grad = (grad / grad.max()) * 255

        grad = grad.astype(np.uint8)

        threshold = self.edge_threshold.get()
        edges_np = grad > threshold

        edges_np = (edges_np * 255).astype(np.uint8)

        self.edges_np = edges_np
        self.grad = grad

    def show_edges(self, method):
        self.find_edges(method)
        self.last_edge_method = method
        if self.edges_np is None:
            return

        canvas_w = self.edges_canvas.winfo_width()
        canvas_h = self.edges_canvas.winfo_height()

        if canvas_w < 2 or canvas_h < 2:
            return

        self.edge_label.config(text=f"Metoda: {method}")

        img_full = Image.fromarray(self.edges_np)

        img_w, img_h = img_full.size
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        new_w, new_h = int(img_w * ratio), int(img_h * ratio)

        img_resized = img_full.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.edges_img = ImageTk.PhotoImage(img_resized)

        self.edges_canvas.delete("all")
        self.edges_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.edges_img)

        self.notebook.select(self.page_edges)

    def save_edges(self):
        if not hasattr(self, 'edges_np') or self.edges_np is None:
            messagebox.showwarning("Błąd", "Najpierw wykryj krawędzie.")
            return

        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if path:
            Image.fromarray(self.edges_np).save(path)
            messagebox.showinfo("OK", "Krawędzie zapisane!")

    def manual_erosion(self, image_array):
        h, w = image_array.shape
        output = np.zeros((h, w), dtype=np.uint8)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                roi = image_array[y - 1:y + 2, x - 1:x + 2]
                if np.all(roi == 255):
                    output[y, x] = 255
        return output

    def manual_dilation(self, image_array):
        h, w = image_array.shape
        output = np.zeros((h, w), dtype=np.uint8)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                roi = image_array[y - 1:y + 2, x - 1:x + 2]
                if np.any(roi == 255):
                    output[y, x] = 255
        return output

    def manual_morphology(self, mode):
        if self.base_np is None:
            messagebox.showwarning("Błąd", "Wczytaj obraz!")
            return
        if len(self.base_np.shape) == 3:
            messagebox.showwarning("Błąd", "Najpierw wykonaj binaryzację!")
            return

        img = self.base_np.copy()

        if mode == "erosion":
            self.base_np = self.manual_erosion(img)
        elif mode == "dilation":
            self.base_np = self.manual_dilation(img)
        elif mode == "opening":
            eroded = self.manual_erosion(img)
            self.base_np = self.manual_dilation(eroded)
        elif mode == "closing":
            dilated = self.manual_dilation(img)
            self.base_np = self.manual_erosion(dilated)

        self.apply_modifications()

    def binaryzacja_manual(self):
        if self.base_np is None: return

        threshold = simpledialog.askinteger("Próg", "Podaj próg binaryzacji (0-255):",
                                            parent=self.root, minvalue=0, maxvalue=255)

        if threshold is not None:
            if len(self.base_np.shape) == 3:
                gray = np.dot(self.base_np[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                gray = self.base_np

            self.base_np = np.where(gray > threshold, 255, 0).astype(np.uint8)
            self.is_szarosc = True
            self.apply_modifications()

    def binaryzacja_otsu(self):
        if self.base_np is None: return

        # Musimy mieć obraz w skali szarości
        if len(self.base_np.shape) == 3:
            gray = np.dot(self.base_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        else:
            gray = self.base_np

        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        total = gray.size

        current_max = -1
        threshold = 0

        sum_total = np.dot(np.arange(256), hist)
        sum_back = 0
        weight_back = 0

        for i in range(256):
            weight_back += hist[i]
            if weight_back == 0: continue

            weight_fore = total - weight_back
            if weight_fore == 0: break

            sum_back += i * hist[i]
            mean_back = sum_back / weight_back
            mean_fore = (sum_total - sum_back) / weight_fore

            var_between = weight_back * weight_fore * (mean_back - mean_fore) ** 2

            if var_between > current_max:
                current_max = var_between
                threshold = i

        messagebox.showinfo("Otsu", f"Wyznaczony próg: {threshold}")
        self.base_np = np.where(gray > threshold, 255, 0).astype(np.uint8)
        self.is_szarosc = True
        self.apply_modifications()

    def binaryzacja_local(self):
        if self.base_np is None: return

        if len(self.base_np.shape) == 3:
            gray = np.dot(self.base_np[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = self.base_np.astype(np.float32)

        h, w = gray.shape
        output = np.zeros_like(gray, dtype=np.uint8)
        window_size = 15
        offset = 10

        r = window_size // 2
        for y in range(r, h - r):
            for x in range(r, w - r):

                roi = gray[y - r:y + r + 1, x - r:x + r + 1]
                local_mean = np.mean(roi)


                if gray[y, x] > (local_mean - offset):
                    output[y, x] = 255
                else:
                    output[y, x] = 0

        self.base_np = output
        self.is_szarosc = True
        self.apply_modifications()

    def not_implemented(self, name):
        messagebox.showinfo("Zadanie", f"Tu zaimplementuj ręcznie algorytm: {name}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BiometriaApp(root)
    root.mainloop()
