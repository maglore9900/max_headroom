import win32gui
import win32con

class WindowFocusManager:
    def __init__(self):
        self.windows = []

    def enum_windows_callback(self, hwnd, window_list):
        # Append the window handle and title to the list if it's visible
        if win32gui.IsWindowVisible(hwnd):
            window_list.append((hwnd, win32gui.GetWindowText(hwnd)))

    def find_windows(self, partial_window_title):
        self.windows = []
        win32gui.EnumWindows(self.enum_windows_callback, self.windows)
        
        # Filter windows that match the partial title
        matching_windows = [hwnd for hwnd, title in self.windows if partial_window_title.lower() in title.lower()]
        return matching_windows

    def bring_window_to_front(self, hwnd):
        # Bring the window to the foreground
        win32gui.SetForegroundWindow(hwnd)
        
        # If the window is minimized, restore it
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        
        window_title = win32gui.GetWindowText(hwnd)
        print(f"Brought window '{window_title}' to the front.")

    def bring_specific_instance_to_front(self, partial_window_title):
        matching_windows = self.find_windows(partial_window_title)
        
        if matching_windows:
            # If there are multiple matches, select the first one (or customize as needed)
            hwnd = matching_windows[0]
            self.bring_window_to_front(hwnd)
        else:
            print(f"No windows found with title containing '{partial_window_title}'.")

# Example usage:
# window_manager = WindowFocusManager()
# window_manager.bring_specific_instance_to_front("outlook")  # Bring the first matching instance of Visual Studio Code to the front
# window_manager.bring_specific_instance_to_front("Notepad")  # Bring the first matching instance of Notepad to the front
