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
        device_id = self.get_active_device()
        self.sp.start_playback(device_id=device_id)
    
    def pause(self):
        device_id = self.get_active_device()
        self.sp.pause_playback(device_id=device_id)
    
    def next_track(self):
        device_id = self.get_active_device()
        self.sp.next_track(device_id=device_id)
    
    def previous_track(self):
        device_id = self.get_active_device()
        self.sp.previous_track(device_id=device_id)
       
    def favorite_current_song(self):
        current_track = self.sp.current_playback()
        if current_track and current_track['item']:
            track_id = current_track['item']['id']
            self.sp.current_user_saved_tracks_add([track_id])
            print(f"Added '{current_track['item']['name']}' to favorites")
        else:
            print("No song is currently playing")

    def search_song_and_play(self, song_name):
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

    def search_artist_and_play(self, artist_name):
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

    def search_album_and_play(self, album_name):
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

    def favorite_current_song(self):
        current_track = self.sp.current_playback()
        if current_track and current_track['item']:
            track_id = current_track['item']['id']
            self.sp.current_user_saved_tracks_add([track_id])
            print(f"Added '{current_track['item']['name']}' to favorites")
        else:
            print("No song is currently playing")
