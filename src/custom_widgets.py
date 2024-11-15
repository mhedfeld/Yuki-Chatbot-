import tkinter as tk
from ttkbootstrap import Style, ttk
from PIL import Image, ImageTk

class MessageBubble(ttk.Frame):
    def __init__(self, master, message, is_user=True, **kwargs):
        super().__init__(master, **kwargs)
        self.pack(pady=5, anchor='e' if is_user else 'w', fill='x')

        style = Style()
        bubble_color = style.colors.primary if is_user else style.colors.secondary
        text_color = style.colors.light if is_user else style.colors.dark

        content_frame = ttk.Frame(self)
        content_frame.pack(side='right' if is_user else 'left')

        icon_path = "images/user_icon.png" if is_user else "images/bot_icon.png"
        try:
            icon_image = Image.open(icon_path).resize((32, 32))
            icon_photo = ImageTk.PhotoImage(icon_image)

            icon_label = ttk.Label(content_frame, image=icon_photo)
            icon_label.image = icon_photo
            icon_label.pack(side='left' if is_user else 'right', padx=(0, 5) if is_user else (5, 0))
        except FileNotFoundError:
            print(f"Warning: Icon file not found: {icon_path}")

        self.bubble = ttk.Label(
            content_frame,
            text=message,
            wraplength=250,
            justify='right' if is_user else 'left',
            style='MessageBubble.TLabel'
        )
        self.bubble.pack(side='right' if is_user else 'left')

        style.configure(
            'MessageBubble.TLabel',
            background=bubble_color,
            foreground=text_color,
            borderwidth=1,
            relief="raised",
            bordercolor=style.colors.border,
            padding=10
        )

class ThinkingAnimation(ttk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.dots = ttk.Label(self, text="", style="Thinking.TLabel")
        self.dots.pack()

        style = Style()
        style.configure("Thinking.TLabel", font=("Helvetica", 24))

        self.dot_count = 0
        self.animate()

    def animate(self):
        self.dot_count = (self.dot_count % 3) + 1
        self.dots.config(text="." * self.dot_count)
        self.after(500, self.animate)