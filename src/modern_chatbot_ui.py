import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import time
import os

ctk.set_appearance_mode("light")  # Modes: system (default), light, dark
ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

class MessageBubble(ctk.CTkFrame):
    def __init__(self, master, message, is_user=True, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.is_user = is_user
        self.message = message

        self.grid_columnconfigure(1, weight=1)

        self.create_widgets()

    def create_widgets(self):
        if self.is_user:
            bubble_color = "#007AFF"  # iMessage blue
            text_color = "white"
            icon_path = os.path.join("images", "user_icon.png")
        else:
            bubble_color = "#E5E5EA"  # iMessage gray
            text_color = "black"
            icon_path = os.path.join("images", "bot_icon.png")

        # Get the absolute path to the image file
        icon_path = os.path.abspath(icon_path)

        # Load and resize icon
        icon = Image.open(icon_path)
        icon = icon.resize((30, 30))  
        icon = ImageTk.PhotoImage(icon)

        # Icon
        icon_label = ctk.CTkLabel(self, image=icon, text="")
        icon_label.image = icon
        icon_label.grid(row=0, column=0, padx=(0, 5))

        # Message bubble
        bubble = ctk.CTkLabel(
            self,
            text=self.message,
            fg_color=bubble_color,
            text_color=text_color,
            corner_radius=15,
            wraplength=200
        )
        bubble.grid(row=0, column=1, sticky="ew", padx=(0, 5) if self.is_user else (5, 0))

        # Adjust layout based on sender
        if self.is_user:
            self.grid_columnconfigure(0, weight=1)
            icon_label.grid(column=2)
            bubble.grid(column=1)
        else:
            self.grid_columnconfigure(2, weight=1)

class ThinkingAnimation(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self.dots = []
        for i in range(3):
            dot = ctk.CTkLabel(self, text="•", font=("Arial", 24))
            dot.grid(row=0, column=i, padx=2)
            self.dots.append(dot)

        self.animate()

    def animate(self):
        for i, dot in enumerate(self.dots):
            self.after(i * 200, lambda d=dot: d.configure(text=""))
            self.after(i * 200 + 100, lambda d=dot: d.configure(text="•"))
        self.after(1000, self.animate)

class ModernChatbotUI(ctk.CTk):
    def __init__(self, response_generator, feedback_handler):
        super().__init__()

        self.feedback_handler = feedback_handler
        self.response_generator = response_generator

        self.title("Modern Chatbot")
        self.geometry("400x600")
        self.minsize(300, 400)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Set custom fonts
        self.font_large = ctk.CTkFont(family="Arial", size=16)
        self.font_medium = ctk.CTkFont(family="Arial", size=12)
        self.font_small = ctk.CTkFont(family="Arial", size=10)

        self.create_widgets()
        
        # Add welcome message
        self.add_message("Welcome! How can I assist you today?", is_user=False)

    def create_widgets(self):
        # Main frame
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Chat area
        self.chat_frame = ctk.CTkScrollableFrame(self.main_frame)
        self.chat_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Input area
        self.input_frame = ctk.CTkFrame(self.main_frame)
        self.input_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.message_entry = ctk.CTkEntry(self.input_frame, placeholder_text="Type a message...")
        self.message_entry.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.send_button = ctk.CTkButton(self.input_frame, text="Send", command=self.send_message)
        self.send_button.grid(row=0, column=1)

        # Style the input area
        self.message_entry.configure(font=self.font_medium, height=35)
        self.send_button.configure(font=self.font_medium, width=60, height=35)

    def send_message(self):
        message = self.message_entry.get()
        if message:
            self.add_message(message, is_user=True)
            self.message_entry.delete(0, ctk.END)
            self.show_thinking_animation()
            self.after(2000, lambda: self.add_bot_message(self.response_generator(message)))

    def add_message(self, message, is_user=True):
        message_bubble = MessageBubble(self.chat_frame, message, is_user=is_user)
        message_bubble.pack(anchor='e' if is_user else 'w', pady=5, padx=10)
        self.chat_frame._parent_canvas.yview_moveto(1.0)  # Scroll to bottom

    def add_bot_message(self, message):
        self.add_message(message, is_user=False)
        feedback_frame = ctk.CTkFrame(self.chat_frame)
        helpful_btn = ctk.CTkButton(feedback_frame, text="👍 Helpful", command=lambda: self.handle_feedback(message, True))
        not_helpful_btn = ctk.CTkButton(feedback_frame, text="👎 Not Helpful", command=lambda: self.handle_feedback(message, False))
        helpful_btn.pack(side="left", padx=5)
        not_helpful_btn.pack(side="left", padx=5)
        feedback_frame.pack(anchor='w', pady=5, padx=10)

    def handle_feedback(self, message, is_helpful):
        user_message = self.get_last_user_message()
        self.feedback_handler(user_message, message, is_helpful)
        feedback_frame = self.chat_frame.winfo_children()[-1]
        feedback_frame.destroy()
        thank_you = ctk.CTkLabel(self.chat_frame, text="Thank you for your feedback!", text_color="gray")
        thank_you.pack(anchor='w', pady=5, padx=10)
        self.after(2000, thank_you.destroy)

    def get_last_user_message(self):
        for widget in reversed(self.chat_frame.winfo_children()):
            if isinstance(widget, MessageBubble) and widget.is_user:
                return widget.message
        return ""

    def show_thinking_animation(self):
        thinking = ThinkingAnimation(self.chat_frame)
        thinking.pack(anchor='w', pady=5, padx=10)
        self.chat_frame._parent_canvas.yview_moveto(1.0)  # Scroll to bottom
        self.after(2000, thinking.destroy)
