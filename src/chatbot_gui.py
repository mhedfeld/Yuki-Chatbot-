import tkinter as tk
from ttkbootstrap import Style, ttk, Window
from PIL import Image, ImageTk, ImageDraw
import math

def blend_colors(color1, color2, alpha):
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
    r = int(r1 * alpha + r2 * (1 - alpha))
    g = int(g1 * alpha + g2 * (1 - alpha))
    b = int(b1 * alpha + b2 * (1 - alpha))
    return f'#{r:02x}{g:02x}{b:02x}'

class RoundedMessage(tk.Canvas):
    def __init__(self, parent, message, is_user=True, **kwargs):
        super().__init__(parent, **kwargs)
        self.is_user = is_user
        self.message = message
        self.draw_bubble()

    def draw_bubble(self):
        width = 300
        font = ('Helvetica', 12)
        padding = 10
        radius = 15

        # Wrap text and calculate height
        wrapped_text = self.wrap_text(self.message, font, width - 2*padding)
        text_height = len(wrapped_text) * font[1] + 2*padding

        height = max(text_height, 40)

        # Create bubble shape
        bg_color = self['bg']
        if self.is_user:
            base_color = "#007AFF"
            text_color = "white"
        else:
            base_color = "#34C759"
            text_color = "white"
        
        fill_color = blend_colors(base_color, bg_color, 0.7)  # 70% opacity

        self.create_rounded_rectangle(0, 0, width, height, radius, fill=fill_color, outline="")

        # Add text
        for i, line in enumerate(wrapped_text):
            y = padding + i * font[1]
            self.create_text(padding, y, text=line, anchor="nw", fill=text_color, font=font)

        self.configure(width=width, height=height)

    def create_rounded_rectangle(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1+radius, y1,
            x1+radius, y1,
            x2-radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1+radius,
            x1, y1
        ]
        return self.create_polygon(points, **kwargs, smooth=True)

    def wrap_text(self, text, font, max_width):
        words = text.split()
        lines = []
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            width = self.font_measure(font, test_line)
            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(word)
        if current_line:
            lines.append(' '.join(current_line))
        return lines

    def font_measure(self, font, text):
        return self.tk.call('font', 'measure', font, text)

class ThinkingAnimation(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, height=30, width=60, bg="#F0F0F0", highlightthickness=0, **kwargs)
        self.dots = []
        for i in range(3):
            x = 20 * (i + 1)
            dot = self.create_oval(x-5, 15-5, x+5, 15+5, fill="#34C759", outline="")
            self.dots.append(dot)
        self.animate()

    def animate(self):
        for i, dot in enumerate(self.dots):
            self.animate_dot(dot, i * 100)
        self.after(1000, self.animate)

    def animate_dot(self, dot, delay):
        self.after(delay, lambda: self.itemconfig(dot, state='hidden'))
        self.after(delay + 100, lambda: self.itemconfig(dot, state='normal'))

class ChatbotGUI:
    def __init__(self, master, response_generator):
        self.master = master
        self.response_generator = response_generator
        
        style = Style(theme="flatly")
        self.master.title("AI Customer Service Chatbot")
        self.master.geometry("400x600")

        self.create_widgets()

    def create_widgets(self):
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        # Header
        header = ttk.Frame(self.master, style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="AI Customer Service", style="Header.TLabel").pack(pady=10)

        # Chat area
        self.chat_frame = ttk.Frame(self.master)
        self.chat_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        self.canvas = tk.Canvas(self.chat_frame, bg="#F0F0F0")
        scrollbar = ttk.Scrollbar(self.chat_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.chat_frame.grid_rowconfigure(0, weight=1)
        self.chat_frame.grid_columnconfigure(0, weight=1)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Input area
        input_frame = ttk.Frame(self.master)
        input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        input_frame.grid_columnconfigure(0, weight=1)

        self.msg_entry = ttk.Entry(input_frame)
        self.msg_entry.grid(row=0, column=0, sticky="ew")

        send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        send_button.grid(row=0, column=1, padx=(10, 0))

        self.msg_entry.bind("<Return>", lambda event: self.send_message())

    def send_message(self):
        user_input = self.msg_entry.get()
        if user_input.strip() != "":
            self.add_message(user_input, is_user=True)
            self.msg_entry.delete(0, tk.END)

            # Show thinking animation
            thinking = ThinkingAnimation(self.scrollable_frame)
            thinking.pack(anchor='w', padx=10, pady=5)
            self.master.update()

            # Generate response
            response = self.response_generator(user_input)

            # Remove thinking animation
            thinking.destroy()

            # Add bot response
            self.add_message(response, is_user=False)

    def add_message(self, message, is_user=True):
        message_bubble = RoundedMessage(self.scrollable_frame, message, is_user=is_user, bg="#F0F0F0", highlightthickness=0)
        message_bubble.pack(anchor='e' if is_user else 'w', padx=10, pady=5)
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)