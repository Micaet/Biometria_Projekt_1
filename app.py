import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tkinter import ttk


class BiometriaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Biometria - Projekt 1")
        self.root.geometry("1100x700")

        # Dane obrazu
        self.original_np = None  # Oryginał (NumPy array) - tu robisz obliczenia
        self.processed_np = None  # Po obróbce (NumPy array)
        self.display_img = None  # Obiekt dla Tkinter (przeskalowany)

        self.setup_ui()

    def setup_ui(self):
        # Główny kontener
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Panel boczny (lewy) - Przyciski
        sidebar = tk.Frame(self.main_container, width=220, bg="#e0e0e0")
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(sidebar, text="Menu Projektu", font=("Arial", 11, "bold"), bg="#e0e0e0").pack(pady=10)

        # Przyciski sterowania
        tk.Button(sidebar, text="Wczytaj Obraz", command=self.load_image).pack(fill=tk.X, padx=10, pady=2)
        tk.Button(sidebar, text="Zapisz Obraz", command=self.save_image).pack(fill=tk.X, padx=10, pady=2)

        ttk.Separator(sidebar).pack(fill=tk.X, pady=10)

        # Miejsca na Twoje algorytmy (Punkty z dokumentacji)
        btns = [
            "Konwersja Szarości", "Jasność / Kontrast", "Negatyw / Binaryzacja",
            "Filtry Liniowe", "Histogram", "Projekcje H/V", "Krawędzie (Sobel/Rob)"
        ]
        for txt in btns:
            tk.Button(sidebar, text=txt, command=lambda t=txt: self.not_implemented(t)).pack(fill=tk.X, padx=10, pady=1)

        # Panel obrazu (prawy) - Canvas ze skalowaniem
        self.canvas = tk.Canvas(self.main_container, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # To zdarzenie wywoła się przy każdej zmianie rozmiaru okna
        self.canvas.bind("<Configure>", self.on_resize)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Obrazy", "*.jpg *.png *.bmp *.jpeg")])
        if path:
            img = Image.open(path).convert("RGB")
            self.original_np = np.array(img)
            self.processed_np = self.original_np.copy()
            self.update_display()

    def save_image(self):
        if self.processed_np is not None:
            path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
            if path:
                # Zamiana NumPy -> PIL przed zapisem
                Image.fromarray(self.processed_np).save(path)
                messagebox.showinfo("OK", "Zapisano pomyślnie!")
        else:
            messagebox.showwarning("Błąd", "Najpierw wczytaj obraz.")

    def update_display(self, event=None):
        if self.processed_np is None:
            return

        # Pobieramy aktualne wymiary canvasu
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 2 or canvas_h < 2:  # Zabezpieczenie przed startem
            return

        # Zamiana tablicy NumPy na obiekt Image do wyświetlenia
        img_full = Image.fromarray(self.processed_np)

        # Obliczanie proporcji (skalowanie z zachowaniem ratio)
        img_w, img_h = img_full.size
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)

        # Skalowanie tylko podglądu (oryginał w processed_np zostaje bez zmian!)
        img_resized = img_full.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.display_img = ImageTk.PhotoImage(img_resized)

        # Centrowanie na środku canvasu
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.display_img)

    def on_resize(self, event):
        # Wywoływane automatycznie przy zmianie rozmiaru okna
        self.update_display()

    def not_implemented(self, name):
        messagebox.showinfo("Zadanie", f"Tu zaimplementuj ręcznie algorytm: {name}")


class tk_Separator(tk.Frame):  # Mały pomocnik do UI
    def __init__(self, parent):
        tk.Frame.__init__(self, parent, height=2, bd=1, relief=tk.SUNKEN)


if __name__ == "__main__":
    root = tk.Tk()
    app = BiometriaApp(root)
    root.mainloop()