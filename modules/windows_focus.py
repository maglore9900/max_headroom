import win32gui
import win32con
import win32api
import pywintypes

class WindowFocusManager:
    def __init__(self):
        self.windows = []

    def enum_windows_callback(self, hwnd, window_list):
        try:
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                window_list.append((hwnd, win32gui.GetWindowText(hwnd)))
        except pywintypes.error as e:
            print(f"Error enumerating window: {e}")

    def find_windows(self, partial_window_title):
        try:
            self.windows = []
            win32gui.EnumWindows(self.enum_windows_callback, self.windows)
            
            matching_windows = [hwnd for hwnd, title in self.windows if partial_window_title.lower() in title.lower()]
            return matching_windows
        except pywintypes.error as e:
            print(f"Error finding windows: {e}")
            return []

    def bring_window_to_front(self, hwnd):
        try:
            # Ensure the window is not minimized
            if win32gui.IsIconic(hwnd):
                try:
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                except pywintypes.error as e:
                    print(f"Error restoring window: {e}")
            
            # Bring the window to the foreground
            try:
                win32gui.SetForegroundWindow(hwnd)
            except pywintypes.error as e:
                print(f"Error setting foreground window: {e}")
            
            # Optionally, send a series of ALT key presses to help with focus
            try:
                win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
                win32gui.SetForegroundWindow(hwnd)
            except pywintypes.error as e:
                print(f"Error sending ALT key event: {e}")
            
            window_title = win32gui.GetWindowText(hwnd)
            print(f"Brought window '{window_title}' to the front.")
        
        except pywintypes.error as e:
            print(f"Failed to bring window to front: {e}")

    def bring_specific_instance_to_front(self, partial_window_title):
        try:
            matching_windows = self.find_windows(partial_window_title)
            
            if matching_windows:
                hwnd = matching_windows[0]
                self.bring_window_to_front(hwnd)
            else:
                print(f"No windows found with title containing '{partial_window_title}'.")
        except pywintypes.error as e:
            print(f"Error bringing specific instance to front: {e}")

# Example usage:
# window_manager = WindowFocusManager()
# window_manager.bring_specific_instance_to_front("Visual Studio Code")  # Bring the first matching instance of Visual Studio Code to the front
# window_manager.bring_specific_instance_to_front("Chrome")  # Bring the first matching instance of Chrome to the front
