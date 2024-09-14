import time
import argparse
# import agent

# spk = agent.Agent().spk

# def timer(seconds):
#     print(f"Timer started for {seconds} seconds.")
#     time.sleep(seconds)
#     print("Time's up!")
#     spk.glitch_stream_output("Time's up!")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Simple Timer Script")
#     parser.add_argument("seconds", type=int, help="Number of seconds to set the timer for")
#     args = parser.parse_args()

#     timer(args.seconds)


# import time
from plyer import notification

def start_timer(seconds):
    print(f"Timer started for {seconds} seconds...")
    time.sleep(seconds)  # Sleep for the desired time
    notification.notify(
        title="Timer Finished",
        message="Your time is up!",
        timeout=5  # Notification will disappear after 10 seconds
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Timer Script")
    parser.add_argument("seconds", type=int, help="Number of seconds to set the timer for")
    args = parser.parse_args()

    start_timer(args.seconds)