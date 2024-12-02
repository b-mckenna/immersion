from tkinter import *
from tkinter import ttk

class TestUI:
    def __init__(self):
        self.root = Tk()
        self.frm = ttk.Frame(self.root, padding=10)
        self.frm.grid()
        
        # Create a StringVar to hold the dynamic text
        self.label_text = StringVar()
        self.label_text.set("Hello World!")  # default text
        
        # Use textvariable instead of text for dynamic updates
        ttk.Label(self.frm, textvariable=self.label_text).grid(column=0, row=0)
        ttk.Button(self.frm, text="Quit", command=self.root.destroy).grid(column=1, row=0)

    def update_text(self, new_text):
        """Method to update the label text from outside"""
        self.label_text.set(new_text)

    def run(self):
        self.root.mainloop()

# Only run this if the script is run directly
if __name__ == "__main__":
    app = TestUI()
    app.run()