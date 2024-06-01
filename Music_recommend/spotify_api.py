import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

client_id = '7bad7e0ca19a4cccbbdc025ee3d50773'
client_secret = '9e7da161b6ba4418ba057090076cc5fe'

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

def fetch_songs_by_year(sp, year, popularity_min=0.5, limit=10):
    query = f'year:{year}'
    # query += f' popularity:{popularity_min}-1'

    results = sp.search(q=query, type='track', limit=limit)
    tracks = results['tracks']['items']
    song_data = []

    for track in tracks:
        audio_features = sp.audio_features(track['id'])[0]

        song_data.append({
            'track_id': track['id'],
            'title': track['name'],
            'artist': ', '.join([artist['name'] for artist in track['artists']]),
            'album': track['album']['name'],
            'release_date': track['album']['release_date'],
            'popularity': track['popularity'],
            'key': audio_features['key'],
            'mode': audio_features['mode'],
            'acousticness': audio_features['acousticness'],
            'danceability': audio_features['danceability'],
            'instrumentalness': audio_features['instrumentalness'],
            'duration_ms': audio_features['duration_ms'],
            'energy': audio_features['energy'],
            'liveness': audio_features['liveness'],
            'loudness': audio_features['loudness'],
            'speechiness': audio_features['speechiness'],
            'valence': audio_features['valence'],
            'tempo': audio_features['tempo']
        })


    return song_data

# # Example usage for a single year
# songs_1970 = fetch_songs_by_year(sp, 1970)
# df_1970 = pd.DataFrame(songs_1970)

all_songs = []

for year in range(1970, 2024):  # Adjust the range for the desired years
    print(f"Fetching songs for year {year}")
    songs = fetch_songs_by_year(sp, year)
    all_songs.extend(songs)

# Create DataFrame and save to CSV
df_all_songs = pd.DataFrame(all_songs)
df_all_songs.to_csv('spotify_songs_1970_to_present.csv', index=False)