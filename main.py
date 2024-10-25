import asyncio
import os
import sys
import json
import base64
import tkinter as tk
from tkinter import scrolledtext, Menu, font, Frame
import tkinter.messagebox
import traceback
import markdown
from bs4 import BeautifulSoup


from computer_use_demo.loop import sampling_loop, APIProvider
from computer_use_demo.tools import ToolResult
from anthropic.types.beta import BetaMessage, BetaMessageParam
from anthropic import APIResponse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ChatInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Mac AI")
        self.root.geometry("400x300")  # Kept the reduced window size
        self.root.configure(bg="#f0f0f0")

        # Configure row and column weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Create a menu bar
        menubar = Menu(root)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        root.config(menu=menubar)

        # Create main frame
        main_frame = Frame(root, bg="#f0f0f0")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Create chat area with support for markdown
        self.chat_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, state='disabled', font=("SF Pro", 12, "normal"), bg="#f0f0f0", fg="#333333", borderwidth=0, highlightthickness=0)
        self.chat_area.grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        # Configure the scrollbar colors
        self.chat_area.vbar.config(troughcolor="#f0f0f0", bg="#d0d0d0")

        # Configure tags for markdown styling
        self.chat_area.tag_configure("bold", font=("SF Pro", 12, "bold"))
        self.chat_area.tag_configure("italic", font=("SF Pro", 12, "italic"))
        self.chat_area.tag_configure("code", font=("Courier", 10, "normal"), background="#e0e0e0")

        # Create entry field and send button
        entry_frame = Frame(main_frame, bg="#f0f0f0")
        entry_frame.grid(row=1, column=0, sticky="ew")
        entry_frame.grid_columnconfigure(0, weight=1)

        self.entry_field = tk.Entry(entry_frame, font=("SF Pro", 13, "normal"), bg="#f0f0f0", fg="#0a0a0a", borderwidth=0, highlightthickness=1, highlightcolor="#e0e0e0")
        self.entry_field.grid(row=0, column=0, sticky="ew", padx=(5, 10), pady=(5, 5), ipady=3)  # Added ipady=3
        self.entry_field.bind("<Return>", self.send_message)
        self.entry_field.insert(0, "Enter text here")
        self.entry_field.bind("<FocusIn>", self.on_entry_click)
        self.entry_field.bind("<FocusOut>", self.on_focusout)
        self.entry_field.config(fg='grey')

        send_button = tk.Button(entry_frame, text="Send", command=self.send_message, font=("SF Pro", 12, "bold"), fg="#fafafa", background="#1a1a1a", activebackground="#1a1a1a", activeforeground="#fafafa", padx=6, pady=2, borderwidth=0, highlightthickness=0)
        send_button.grid(row=0, column=1, sticky="e")

        self.messages = []

    def display_message(self, message, sender="You"):
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, f"{sender}:\n", sender)
        
        if sender == "Assistant":
            # Convert markdown to HTML
            html = markdown.markdown(message)
            
            # Parse HTML and apply styling
            soup = BeautifulSoup(html, 'html.parser')
            self.insert_formatted_text(soup)
        else:
            self.chat_area.insert(tk.END, f"{message}\n")
        
        self.chat_area.insert(tk.END, "\n\n")  # Add extra newlines for spacing
        self.chat_area.tag_config("You", foreground="#808080")
        self.chat_area.tag_config("Assistant", foreground="#000000")
        self.chat_area.tag_config("Tool", foreground="#808080")
        self.chat_area.tag_config("System", foreground="#808080")
        self.chat_area.tag_config("Error", foreground="#ff0000")
        self.chat_area.config(state='disabled')
        self.chat_area.yview(tk.END)

    def insert_formatted_text(self, element):
        if element.name == 'p':
            self.chat_area.insert(tk.END, element.get_text() + "\n")
        elif element.name == 'strong':
            self.chat_area.insert(tk.END, element.get_text(), "bold")
        elif element.name == 'em':
            self.chat_area.insert(tk.END, element.get_text(), "italic")
        elif element.name == 'code':
            self.chat_area.insert(tk.END, element.get_text(), "code")
        elif element.name == 'pre':
            self.chat_area.insert(tk.END, element.get_text() + "\n", "code")
        else:
            for child in element.children:
                if isinstance(child, str):
                    self.chat_area.insert(tk.END, child)
                else:
                    self.insert_formatted_text(child)

    def on_entry_click(self, event):
        """Function that gets called whenever entry is clicked"""
        if self.entry_field.get() == 'Enter text here':
            self.entry_field.delete(0, "end") # delete all the text in the entry
            self.entry_field.insert(0, '') #Insert blank for user input
            self.entry_field.config(fg = 'black')

    def on_focusout(self, event):
        """Function that gets called whenever entry loses focus"""
        if self.entry_field.get() == '':
            self.entry_field.insert(0, 'Enter text here')
            self.entry_field.config(fg = 'grey')

    def send_message(self, message):
        if message and message != 'Enter text here':
            self.display_message(message)
            self.messages.append({"role": "user", "content": message})
            self.entry_field.delete(0, tk.END)
            self.root.after(0, self.process_message)

    def process_message(self):
        asyncio.run(self.run_sampling_loop())

    async def run_sampling_loop(self):
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key == "YOUR_API_KEY_HERE":
                    raise ValueError("Please set your API key in the ANTHROPIC_API_KEY environment variable")
                
                # Debug print statement to check if the API key is being loaded correctly
                print(f"Debug: API key loaded: {api_key[:5]}...{api_key[-5:]}")

                provider = APIProvider.ANTHROPIC

                def output_callback(content_block):
                    if isinstance(content_block, dict) and content_block.get("type") == "text":
                        self.display_message(content_block.get("text"), sender="Assistant")

                def tool_output_callback(result: ToolResult, tool_use_id: str):
                    if result.output:
                        self.display_message(f"> Tool Output [{tool_use_id}]: {result.output}", sender="Tool")
                    if result.error:
                        self.display_message(f"!!! Tool Error [{tool_use_id}]: {result.error}", sender="Tool")
                    if result.base64_image:
                        os.makedirs("screenshots", exist_ok=True)
                        image_data = result.base64_image
                        with open(f"screenshots/screenshot_{tool_use_id}.png", "wb") as f:
                            f.write(base64.b64decode(image_data))
                        self.display_message(f"Took screenshot screenshot_{tool_use_id}.png", sender="Tool")

                def api_response_callback(response: APIResponse[BetaMessage]):
                    content = json.loads(response.text)["content"]
                    filtered_content = self.filter_api_response(content)
                    self.display_message(filtered_content, sender="Assistant")

                self.display_message("Processing your request...", sender="System")
                new_messages = await sampling_loop(
                    model="claude-3-5-sonnet-20241022",
                    provider=provider,
                    system_prompt_suffix="",
                    messages=self.messages,
                    output_callback=output_callback,
                    tool_output_callback=tool_output_callback,
                    api_response_callback=api_response_callback,
                    api_key=api_key,
                    only_n_most_recent_images=10,
                    max_tokens=4096,
                )
                
                # Update self.messages with the new messages
                self.messages = new_messages
                
                self.display_message("Request processed successfully.", sender="System")
                break  # Exit the loop if successful
            except Exception as e:
                retry_count += 1
                error_message = f"Encountered Error (Attempt {retry_count}/{max_retries}):\n{str(e)}"
                self.display_message(error_message, sender="Error")
                print(error_message)
                print(f"Error type: {type(e)}")
                print(f"Error traceback: {traceback.format_exc()}")
                
                if retry_count < max_retries:
                    self.display_message(f"Retrying in 5 seconds...", sender="System")
                    await asyncio.sleep(5)
                else:
                    self.display_message("Max retries reached. Please try again later.", sender="System")
                    # Reset messages to the last known good state
                    self.messages = self.messages[:-1]  # Remove the last user message that caused the error

    def filter_api_response(self, content):
        filtered_content = []
        for item in content:
            if item['type'] == 'text':
                filtered_content.append(item['text'])
        return "\n".join(filtered_content)

    def show_about(self):
        tk.messagebox.showinfo("About", "Claude Computer Use Chat\nVersion 1.0")


def main():
    root = tk.Tk()
    chat_interface = ChatInterface(root)
    
    if len(sys.argv) > 1:
        initial_message = ' '.join(sys.argv[1:])
        root.after(100, lambda: chat_interface.send_message(initial_message))
    
    root.mainloop()


if __name__ == "__main__":
    main()
