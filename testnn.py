import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
rep = filedialog.askopenfilenames(parent=root,
                                      initialdir='data/test/',
                                      initialfile='',
                                      filetypes=[("JPEG", "*.jpg")])
print(rep)