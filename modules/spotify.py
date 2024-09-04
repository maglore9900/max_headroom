import spotipy
import environ
from spotipy.oauth2 import SpotifyOAuth
from requests.exceptions import ConnectionError, HTTPError
import time
import functools

env = environ.Env()
environ.Env.read_env()

def handle_spotify_errors_and_device(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        attempts = 3
        for attempt in range(attempts):
            try:
                # Fetch the active device before calling the function
                device_id = self.get_active_device()
                if device_id is None:
                    print("No active device found.")
                    return None

                # Inject the device_id into the kwargs
                kwargs['device_id'] = device_id

                return func(self, *args, **kwargs)
            except (spotipy.exceptions.SpotifyException, ConnectionError, HTTPError) as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if "token" in str(e).lower():
                    self.refresh_token()
                time.sleep(2)  # Wait before retrying
            except Exception as e:
                print(f"Unexpected error: {e}")
                break
    return wrapper

class Spotify:
    def __init__(self):
        self.auth_manager = SpotifyOAuth(
            client_id=env("spotify_client_id"),
            client_secret=env("spotify_client_secret"),
            redirect_uri=env("spotify_redirect_uri"),
            scope="user-modify-playback-state user-read-playback-state user-library-modify"
        )
        self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
        
    def get_active_device(self):
        try:
            devices = self.sp.devices()
            if devices['devices']:
                active_device_id = devices['devices'][0]['id']
                return active_device_id
            else:
                return None
        except spotipy.exceptions.SpotifyException as e:
            print(f"Error fetching devices: {e}")
            return None

    def refresh_token(self):
        try:
            self.sp = spotipy.Spotify(auth_manager=self.auth_manager)
            print("Token refreshed successfully.")
        except spotipy.exceptions.SpotifyException as e:
            print(f"Failed to refresh token: {e}")
    
    @handle_spotify_errors_and_device
    def play(self, device_id=None):
        if device_id:
            self.sp.start_playback(device_id=device_id)
            print("Playback started successfully.")
        else:
            print("No active device found.")

    @handle_spotify_errors_and_device
    def pause(self, device_id=None):
        if device_id:
            self.sp.pause_playback(device_id=device_id)
            print("Playback paused successfully.")
        else:
            print("No active device found.")
            
    @handle_spotify_errors_and_device       
    def next_track(self, device_id=None):
        if device_id:
            self.sp.next_track(device_id=device_id)
        else:
            print("Failed to skip to the next track")
            
    @handle_spotify_errors_and_device
    def previous_track(self, device_id=None):
        if device_id:
            self.sp.previous_track(device_id=device_id)
        else:
            print("Failed to go to the previous track")
            
    @handle_spotify_errors_and_device
    def favorite_current_song(self, device_id=None):
        if device_id:
            current_track = self.sp.current_playback()
            if current_track and current_track['item']:
                track_id = current_track['item']['id']
                self.sp.current_user_saved_tracks_add([track_id])
                print(f"Added '{current_track['item']['name']}' to favorites")
            else:
                print("No song is currently playing")
        else:
            print("Failed to add current song to favorites")
            
    @handle_spotify_errors_and_device
    def search_song_and_play(self, song_name, device_id=None):
        try:
            results = self.sp.search(q='track:' + song_name, type='track')
            if results['tracks']['items']:
                track_uri = results['tracks']['items'][0]['uri']
                if device_id:
                    self.sp.start_playback(device_id=device_id, uris=[track_uri])
                else:
                    print("No active device found. Please start Spotify on a device and try again.")
            else:
                print(f"No results found for song: {song_name}")
        except Exception as e:
            print(f"Failed to search and play song '{song_name}': {e}")
            
    @handle_spotify_errors_and_device
    def search_artist_and_play(self, artist_name, device_id=None):
        try:
            results = self.sp.search(q='artist:' + artist_name, type='artist')
            if results['artists']['items']:
                artist_uri = results['artists']['items'][0]['uri']
                if device_id:
                    self.sp.start_playback(device_id=device_id, context_uri=artist_uri)
                else:
                    print("No active device found. Please start Spotify on a device and try again.")
            else:
                print(f"No results found for artist: {artist_name}")
        except Exception as e:
            print(f"Failed to search and play artist '{artist_name}': {e}")
            
    @handle_spotify_errors_and_device
    def search_album_and_play(self, album_name, device_id=None):
        try:
            results = self.sp.search(q='album:' + album_name, type='album')
            if results['albums']['items']:
                album_uri = results['albums']['items'][0]['uri']
                if device_id:
                    self.sp.start_playback(device_id=device_id, context_uri=album_uri)
                else:
                    print("No active device found. Please start Spotify on a device and try again.")
            else:
                print(f"No results found for album: {album_name}")
        except Exception as e:
            print(f"Failed to search and play album '{album_name}': {e}")
