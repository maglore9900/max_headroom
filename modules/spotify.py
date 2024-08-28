import spotipy
import environ
from spotipy.oauth2 import SpotifyOAuth

env = environ.Env()
environ.Env.read_env()

class Spotify:
    def __init__(self):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=env("spotify_client_id"),
                                                            client_secret=env("spotify_client_secret"),
                                                            redirect_uri=env("spotify_redirect_uri"),
                                                            scope="user-modify-playback-state user-read-playback-state user-library-modify"))
    def get_active_device(self):
        devices = self.sp.devices()
        if devices['devices']:
            # Select the first active device
            active_device_id = devices['devices'][0]['id']
            return active_device_id
        else:
            return None
    
    def play(self):
        try:
            device_id = self.get_active_device()
            self.sp.start_playback(device_id=device_id)
        except Exception as e:
            print(f"Failed to play: {e}")
    
    def pause(self):
        try:
            device_id = self.get_active_device()
            self.sp.pause_playback(device_id=device_id)
        except Exception as e:
            print(f"Failed to pause playback: {e}")
    
    def stop(self):
        try:
            device_id = self.get_active_device()
            self.sp.pause_playback(device_id=device_id)
        except Exception as e:
            print(f"Failed to stop playback: {e}")

    def next_track(self):
        try:
            device_id = self.get_active_device()
            self.sp.next_track(device_id=device_id)
        except Exception as e:
            print(f"Failed to skip to the next track: {e}")

    def previous_track(self):
        try:
            device_id = self.get_active_device()
            self.sp.previous_track(device_id=device_id)
        except Exception as e:
            print(f"Failed to go to the previous track: {e}")

    def favorite_current_song(self):
        try:
            current_track = self.sp.current_playback()
            if current_track and current_track['item']:
                track_id = current_track['item']['id']
                self.sp.current_user_saved_tracks_add([track_id])
                print(f"Added '{current_track['item']['name']}' to favorites")
            else:
                print("No song is currently playing")
        except Exception as e:
            print(f"Failed to add current song to favorites: {e}")

    def search_song_and_play(self, song_name):
        try:
            results = self.sp.search(q='track:' + song_name, type='track')
            if results['tracks']['items']:
                track_uri = results['tracks']['items'][0]['uri']
                device_id = self.get_active_device()
                if device_id:
                    self.sp.start_playback(device_id=device_id, uris=[track_uri])
                else:
                    print("No active device found. Please start Spotify on a device and try again.")
            else:
                print(f"No results found for song: {song_name}")
        except Exception as e:
            print(f"Failed to search and play song '{song_name}': {e}")

    def search_artist_and_play(self, artist_name):
        try:
            results = self.sp.search(q='artist:' + artist_name, type='artist')
            if results['artists']['items']:
                artist_uri = results['artists']['items'][0]['uri']
                device_id = self.get_active_device()
                if device_id:
                    self.sp.start_playback(device_id=device_id, context_uri=artist_uri)
                else:
                    print("No active device found. Please start Spotify on a device and try again.")
            else:
                print(f"No results found for artist: {artist_name}")
        except Exception as e:
            print(f"Failed to search and play artist '{artist_name}': {e}")

    def search_album_and_play(self, album_name):
        try:
            results = self.sp.search(q='album:' + album_name, type='album')
            if results['albums']['items']:
                album_uri = results['albums']['items'][0]['uri']
                device_id = self.get_active_device()
                if device_id:
                    self.sp.start_playback(device_id=device_id, context_uri=album_uri)
                else:
                    print("No active device found. Please start Spotify on a device and try again.")
            else:
                print(f"No results found for album: {album_name}")
        except Exception as e:
            print(f"Failed to search and play album '{album_name}': {e}")
