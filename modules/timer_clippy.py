import time
import argparse
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont

text_to_add = "I see you are just sitting there, would you like to get the F up?!"
def wrap_text(text, font, max_width, draw):
    """
    Wrap the text so that it fits within the given max_width.
    """
    words = text.split(' ')
    lines = []
    current_line = []
    current_width = 0
    
    for word in words:
        # Calculate the width of the word using textbbox
        word_width = draw.textbbox((0, 0), word + ' ', font=font)[2] - draw.textbbox((0, 0), word + ' ', font=font)[0]
        
        if current_width + word_width <= max_width:
            current_line.append(word)
            current_width += word_width
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_width = word_width
    lines.append(' '.join(current_line))  # Add the last line
    
    return '\n'.join(lines)

def start_timer(seconds):
    seconds = int(seconds)
    print(f"Timer started for {seconds} seconds...")
    time.sleep(seconds)
    
    # Create the root window and remove the title bar
    root = tk.Tk()
    root.overrideredirect(1)  # Remove title bar
    root.attributes("-topmost", True)  # Keep window on top of others
    
    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Load the image
    image = Image.open("images/clippy.png")  # Replace with your image path

    # Initialize ImageDraw to modify the image
    draw = ImageDraw.Draw(image)
    
    # Load a font. You can use default fonts or specify your own .ttf file.
    try:
        font = ImageFont.truetype("tahoma.ttf", 24)  # You can replace this with any font available on your system
    except IOError:
        font = ImageFont.load_default()  # Fallback to default if font not found
    
    # Get the width and height of the image
    img_width, img_height = image.size
    
    # Set a maximum text width to allow word wrapping
    max_text_width = img_width - 40  # Padding of 20px on each side
    
    # Wrap the text to fit within the max_text_width
    wrapped_text = wrap_text(text_to_add, font, max_text_width, draw)
    
    # Add padding for the text position
    text_x = 20  # Left alignment padding
    text_y = img_height // 2 - 130  # Higher position for the text
    
    # Add wrapped text to the image (left-aligned)
    draw.text((text_x, text_y), wrapped_text, font=font, fill="black", align="left")
    
    # Convert the image to a format Tkinter can display
    photo = ImageTk.PhotoImage(image)

    # Display image with text
    label_image = tk.Label(root, image=photo)
    label_image.pack()
    
    # Add a wider close button ("X")
    close_button = tk.Button(root, text="  X  ", command=root.destroy, bg="red", fg="white", bd=0)  # Extra spaces for width
    close_button.place(relx=1.0, rely=0.0, anchor="ne")
    
    # Define padding for bottom-right position (e.g., 20px padding)
    padding_x = 20
    padding_y = 70
    
    # Calculate position to place window in the bottom-right of the screen with padding
    window_x = screen_width - img_width - padding_x
    window_y = screen_height - img_height - padding_y

    # Position the window at the bottom-right corner with padding
    root.geometry(f"{img_width}x{img_height}+{window_x}+{window_y}")
    
    # Auto-close the window after 5 seconds if not closed manually
    root.after(5000, root.destroy)
    
    # Run the main loop to display the window and handle events
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Timer Script")
    parser.add_argument("seconds", type=int, help="Number of seconds to set the timer for")
    args = parser.parse_args()

    start_timer(args.seconds)
