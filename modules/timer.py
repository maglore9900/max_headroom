import time
import argparse
import agent

spk = agent.Agent().spk

def timer(seconds):
    print(f"Timer started for {seconds} seconds.")
    time.sleep(seconds)
    print("Time's up!")
    spk.glitch_stream_output("Time's up!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Timer Script")
    parser.add_argument("seconds", type=int, help="Number of seconds to set the timer for")
    args = parser.parse_args()

    timer(args.seconds)
