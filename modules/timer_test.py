# import time
# from win10toast import ToastNotifier

# def start_timer(seconds):
#     print(f"Timer started for {seconds} seconds...")
#     time.sleep(seconds)
#     toaster = ToastNotifier()
#     toaster.show_toast("Timer Finished", "Your time is up!", duration=10)

# # Example: Set a timer for 60 seconds
# start_timer(5)

import time
from plyer import notification

def start_timer(seconds):
    seconds = int(seconds)  # Convert to integer
    print(f"Timer started for {seconds} seconds...")
    time.sleep(seconds)  # Sleep for the desired time
    notification.notify(
        title="Timer Finished",
        message="Your time is up!",
        timeout=10  # Notification will disappear after 10 seconds
    )

# Example: Set a timer for 60 seconds
# start_timer(5)
