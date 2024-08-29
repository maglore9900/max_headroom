import os
import json
import subprocess
from difflib import get_close_matches
from datetime import datetime, timedelta

class AppLauncher:
    def __init__(self, search_paths=None, index_file='tmp/app_index.json'):
        if search_paths is None:
            search_paths = [
                r'C:\Program Files',
                r'C:\Program Files (x86)',
                r'C:\Windows\System32'
            ]
        self.search_paths = search_paths
        self.index_file = index_file
        self.index = {}
        self.load_index()

    def index_applications(self):
        app_index = {}
        current_date = datetime.now().strftime('%Y-%m-%d')

        # Walk through the directories and index all executables
        for path in self.search_paths:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.exe'):
                        app_name = file.lower()
                        app_path = os.path.join(root, file)
                        app_index[app_name] = {'path': app_path, 'date_indexed': current_date}

        # Save the index to a JSON file
        with open(self.index_file, 'w') as f:
            json.dump(app_index, f, indent=4)

        self.index = app_index
        print(f"Indexing complete. {len(app_index)} applications indexed.")

    def load_index(self):
        try:
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)

            # Check if the index is older than 5 days
            any_entry = next(iter(self.index.values()))
            date_indexed = datetime.strptime(any_entry['date_indexed'], '%Y-%m-%d')
            if datetime.now() - date_indexed > timedelta(days=5):
                print("Index is older than 5 days. Reindexing...")
                self.index_applications()

        except (FileNotFoundError, ValueError, KeyError):
            # If the index file doesn't exist or is corrupted, reindex
            print("Index file not found or corrupted. Creating new index...")
            self.index_applications()

    def find_and_open_app(self, app_name):
        match = get_close_matches(app_name.lower(), self.index.keys(), n=1, cutoff=0.6)

        if match:
            app_path = self.index[match[0]]['path']
            subprocess.Popen(app_path)  # Use Popen instead of run
            print(f"Opening {match[0]} at {app_path}")
        else:
            print(f"No matching application found for '{app_name}'")


