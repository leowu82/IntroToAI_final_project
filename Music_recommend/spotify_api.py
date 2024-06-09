import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
from langdetect import detect

# Spotify API credentials
client_id = 'eea1a03234324e1b9a16923690281451'
client_secret = '1181d79abbe549219128d593314f25ba'

# Initialize Spotify API client
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

def fetch_songs_by_year(sp, year, language, popularity_min=30, valence_min=0.3, danceability_min=0.6012, energy_min=0.8104, loudness_min=0.7858, speechiness_min=0.0822, acousticness_min=0.0185, limit=50):
    query = f'year:{year}'
    # query += f' popularity:{popularity_min}-1'

    results = sp.search(q=query, type='track', limit=limit, offset=200)
    tracks = results['tracks']['items']
    song_data = []

    for track in tracks:
        audio_features = sp.audio_features(track['id'])[0]
        
        if audio_features is None:
            # Skip tracks that don't have audio features
            continue

        # # Check Language
        # title_language = detect(track['name'])
        # artist_language = detect(track['artists'][0]['name'])
        # if title_language == language or artist_language == language: print("Valid Lang")
        # else: continue

        # # Or
        # if track['language_of_performace'] is not None and track['language_of_performace'] != language: continue
        
        score = 0
        # if audio_features['popularity'] > popularity_min: continue
        # if audio_features['valence'] > valence_min: continue
        if audio_features['danceability'] > danceability_min: score += 0.170203
        if audio_features['energy'] > energy_min: score += 0.159384
        if audio_features['speechiness'] > speechiness_min: score += 0.111102
        if audio_features['acousticness'] > acousticness_min: score += 0.089827
        # if audio_features['loudness'] > loudness_min: score += 1

        if score < ((0.170203+0.159384+0.111102+0.089827)/2): continue

        song_data.append({
            'spotify_url': track['external_urls']['spotify'],
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

# Function to fetch songs of desired year
def fetch_desired_year_songs(year): 
    all_songs = []

    print(f"Fetching songs for year {year}")
    try:
        songs = fetch_songs_by_year(sp, year, ['en', 'zh'])
        all_songs.extend(songs)
    except Exception as e:
        print(f"An error occurred for year {year}: {e}")

    # Create DataFrame and save to CSV
    df_all_songs = pd.DataFrame(all_songs)
    df_all_songs.to_csv('spotify_songs_request_year.csv', index=False)

    return df_all_songs
