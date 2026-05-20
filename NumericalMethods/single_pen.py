import tkinter as tk
import math
import time

class SwingingPole:
    def __init__(self, root):
        self.canvas = tk.Canvas(root, width=400, height=400, bg='white')
        self.canvas.pack()
        
        self.pivot_x, self.pivot_y = 200, 50
        self.length = 200
        self.max_angle = math.radians(45)  
        
        self.pole = self.canvas.create_line(0, 0, 0, 0, width=5, fill="brown")
        
        self.animate()

    def animate(self):
        t = time.time() * 3 
        angle = self.max_angle * math.sin(t)
        
        end_x = self.pivot_x + self.length * math.sin(angle)
        end_y = self.pivot_y + self.length * math.cos(angle)
        
        self.canvas.coords(self.pole, self.pivot_x, self.pivot_y, end_x, end_y)
        
        self.canvas.after(16, self.animate)

root = tk.Tk()
app = SwingingPole(root)
root.mainloop()
