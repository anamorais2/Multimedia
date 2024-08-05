from genericpath import isfile
from ntpath import join
import os
import librosa
import numpy as np
import scipy
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from scipy.spatial import distance


def read_Directory(path):
        return [f for f in os.listdir(path) if isfile(join(path, f))]

def save_features(output, features):
    np.savetxt(output, features, fmt="%0.5f", delimiter=',')

def read_features(input):
    return np.genfromtxt(input, delimiter=',')

def read_top_10(input):
    data = np.genfromtxt(input, delimiter=',', dtype=None, encoding=None)
    return [(name, float(dist)) for name, dist in data]


def get_musics(dir):
    files = os.listdir(dir)
    return sorted(files)

def extract_stats(features):

    mean = np.mean(features)
    stddv = np.std(features)
    skew = scipy.stats.skew(features)
    kurtosis = scipy.stats.kurtosis(features)
    median = np.median(features)
    max = np.max(features)
    min = np.min(features)
        
    return np.array([mean, stddv, skew, kurtosis, median, max, min], dtype=object)

def extract_features(audios, output,sr=22050):

    if os.path.isfile(output):
        return read_features(output)

    listSongsNames = read_Directory(audios)
    len_Songs = int(np.shape(listSongsNames)[0])

    features_list = np.arange(len_Songs*190, dtype=object).reshape((len_Songs, 190)) # 13 MFCC bands * 7 stats + 6 other features * 7 stats + 1 tempo = 190 features
    
    for song in range(len_Songs):
        songName = listSongsNames[song]
        file_path = os.path.join(audios, songName)
        y, fs = librosa.load(file_path, sr=sr, mono = True)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        f0 = librosa.yin(y, fmin=20, fmax=sr/2) 
        f0[f0 == librosa.note_to_hz('C7')] = 0 # Set the f0 value for C7 to 0
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        tempo = librosa.feature.tempo(y=y)[0]

        #features_list[song, 0] = songName

        for i in range(len(mfcc)):
            features_list[song, i*7:i*7+7] = extract_stats(mfcc[i])

        features_list[song, 91:91+7] = extract_stats(spectral_centroid)

        features_list[song, 98:98+7] = extract_stats(spectral_bandwidth)

        for i in range(len(spectral_contrast)): 
            features_list[song, 105+i*7:105+i*7+7] = extract_stats(spectral_contrast[i])

        features_list[song, 154:154+7] = extract_stats(spectral_flatness)

        features_list[song, 161:161+7] = extract_stats(spectral_rolloff)

        features_list[song, 168:168+7] = extract_stats(f0)

        features_list[song, 175:175+7] = extract_stats(rms)

        features_list[song, 182:182+7] = extract_stats(zcr)

        features_list[song, -1] = tempo
    
    print("Features extracted successfully!")

    save_features(output, features_list)

    return features_list

def save_normalized_features(output, features_list, min_vals, max_vals):


    data_to_save = [min_vals, max_vals]
    data_to_save.extend(features_list)

    np.savetxt(output, data_to_save, fmt="%0.5f", delimiter=',')

def normalized_features(a, b, features_list, output): # Normalization of features to the range [a,b]

    if os.path.isfile(output):
        data = read_features(output)
        min_vals = data[0]
        max_vals = data[1]
        features_list = data[2:]
        return features_list, min_vals, max_vals

    num_rows, num_cols = np.shape(features_list)

    min_vals = []
    max_vals = []
    
    for column in range(num_cols):
        fMin = features_list[:, column].min()
        fMax = features_list[:, column].max()
        min_vals.append(fMin)
        max_vals.append(fMax)

        if fMax == fMin:
            features_list[:, column] = 0
        else:
            try:
                features_list[:, column] = a + ((features_list[:, column]-fMin) * (b-a)) / (fMax-fMin)
            except:
                print(features_list[:, column])

    print("Features normalized successfully!")
    
    save_normalized_features(output, features_list, min_vals, max_vals)
    
    return features_list, min_vals, max_vals



def normalized_features_query(a, b, features_list, min_vals, max_vals, output): 
    
    if os.path.isfile(output):
        return read_features(output)
    
    num_rows, num_cols = np.shape(features_list)

    for column in range(num_cols):
        fMin = min_vals[column]
        fMax = max_vals[column]

        if fMax == fMin:
            features_list[:, column] = 0
        else:
            features_list[:, column] = a + ((features_list[:, column] - fMin) * (b - a)) / (fMax - fMin)

    print("Features normalized successfully!")

    save_normalized_features(output, features_list, min_vals, max_vals)
    
    return features_list



def my_spectral_centroid(signal, sr=22050, frame_size=2048, hop_size=512):
    window = np.hanning(frame_size)
    
    # Cálculo manual das frequências
    #frequencies = np.linspace(0, sr / 2, frame_size // 2 + 1)
    frequencies = np.fft.rfftfreq(frame_size, 1.0/sr)

    spectrogram = []
    num_frames = (len(signal) - frame_size) // hop_size + 1  # Cálculo do número de janelas
    
    # Compute the spectrogram for each frame
    for i in range(0, num_frames * hop_size, hop_size):
        frame = signal[i:i+frame_size] * window
        spectrum = np.abs(np.fft.rfft(frame))
        spectrogram.append(spectrum)
    
    # Calculate the spectral centroid for each frame
    centroids = []
    for frame in spectrogram:
        sum_magnitude = np.sum(frame)
        if sum_magnitude == 0:
            centroids.append(0)
        else:
            sc = np.sum(frequencies * frame) / sum_magnitude
            centroids.append(sc)
    
    return np.array(centroids)

def calculate_and_compare(audios_folder, sr=22050):

    if os.path.isfile("pearson_rmse_results.csv"):
        return read_features("pearson_rmse_results.csv")
     
    results = []

    for filename in os.listdir(audios_folder):
        file_path = os.path.join(audios_folder, filename)
        signal, sr = librosa.load(file_path, sr=sr, mono=True)
        my_centroid = my_spectral_centroid(signal)
        librosa_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)

        # Adjust the starting index as necessary to match the data
        sc = librosa_centroid[:, 2:]  
        my_centroid = my_centroid[:sc.shape[1]]  

        # Ensure that the two arrays have the same length
        minSize = min(sc.shape[1], len(my_centroid))
        sc = sc[:, :minSize]
        my_centroid = my_centroid[:minSize]


        sc_flat = sc.flatten() # Flatten the 2D array to 1D

        correlation, _ = pearsonr(sc_flat, my_centroid)
        rmse = np.sqrt(mean_squared_error(sc_flat, my_centroid))

        results.append((correlation, rmse))
        
    print("Results saved successfully!")
    save_features("pearson_rmse_results.csv", results)

    return results


def euclidean_distance(F1, F2):
    return np.sqrt(np.sum((F1 - F2) ** 2))

def manhattan_distance(F1, F2):
    return np.sum(np.abs(F1 - F2))

def cosine_distance(F1, F2):
    if np.all(F1 == 0) or np.all(F2 == 0):
        return 0
    return distance.cosine(F1, F2)

def similarity_measurements(query_features, features_list_normalized):

    if os.path.isfile("euclideanDistance.csv") and os.path.isfile("manhattanDistance.csv") and os.path.isfile("cosineDistance.csv"):
        euclideanDistance = read_features("euclideanDistance.csv")
        manhattanDistance = read_features("manhattanDistance.csv")
        cosineDistance = read_features("cosineDistance.csv")
        return euclideanDistance, manhattanDistance, cosineDistance

    num_songs = len(features_list_normalized) 

    # Inicializa os arrays para as distâncias
    euclideanDistance = np.zeros(num_songs)
    manhattanDistance = np.zeros(num_songs)
    cosineDistance = np.zeros(num_songs)

    # Calcula as distâncias da consulta para cada música no dataset
    for i in range(num_songs):
        euclideanDistance[i] = euclidean_distance(query_features, features_list_normalized[i])
        manhattanDistance[i] = manhattan_distance(query_features, features_list_normalized[i])
        cosineDistance[i] = cosine_distance(query_features, features_list_normalized[i])

    print("Distances calculated successfully!")

    save_features("euclideanDistance.csv", euclideanDistance)
    save_features("manhattanDistance.csv", manhattanDistance)
    save_features("cosineDistance.csv", cosineDistance)
    
    return euclideanDistance, manhattanDistance, cosineDistance


def create_similarity_rankings(euclideanDistance, manhattanDistance, cosineDistance, audio_folder_path):

    if os.path.isfile("ranking_euclidean.csv") and os.path.isfile("ranking_manhattan.csv") and os.path.isfile("ranking_cosine.csv"):
        top_10_euclidean = read_top_10("ranking_euclidean.csv")
        top_10_manhattan = read_top_10("ranking_manhattan.csv")
        top_10_cosine = read_top_10("ranking_cosine.csv")
        return top_10_euclidean, top_10_manhattan, top_10_cosine
    
    euclideanRanking = np.argsort(euclideanDistance)[:10] # Get the indexes of the 10 smallest distances
    manhattanRanking = np.argsort(manhattanDistance)[:10] 
    cosineRanking = np.argsort(cosineDistance)[:10]

    listSongsNames = read_Directory(audio_folder_path)

    top_10_euclidean = [(listSongsNames[i], euclideanDistance[i]) for i in euclideanRanking] # Create a list of tuples with the song name and the distance
    top_10_manhattan  = [(listSongsNames[i], manhattanDistance[i]) for i in manhattanRanking]
    top_10_cosine = [(listSongsNames[i], cosineDistance[i]) for i in cosineRanking]


    print("Top 10 Euclidean Distance Songs:")
    for name, dist in top_10_euclidean:
        print(f"{name}: {dist:.5f}")

    print("Top 10 Manhattan Distance Songs:")
    for name, dist in top_10_manhattan:
        print(f"{name}: {dist:.5f}")

    print("Top 10 Cosine Distance Songs:")
    for name, dist in top_10_cosine:
        print(f"{name}: {dist:.5f}")

    print("Rankings created successfully!")

    np.savetxt("ranking_euclidean.csv", top_10_euclidean, fmt="%s", delimiter=",")
    np.savetxt("ranking_manhattan.csv", top_10_manhattan, fmt="%s", delimiter=",")
    np.savetxt("ranking_cosine.csv", top_10_cosine, fmt="%s", delimiter=",")

    return top_10_euclidean, top_10_manhattan, top_10_cosine

def metadataExtraction(data):

    metadataDefault = np.genfromtxt(data, delimiter=",", dtype="str")
    metadata = metadataDefault[1:, [1,9, 11]] # Extract the columns with the metadata
    scores = np.zeros((900, 900))
    np.fill_diagonal(scores, -1)
    for i in range(metadata.shape[0]):
        for j in range(i+1, metadata.shape[0]):
            for k in range(metadata.shape[1]):
                listA = metadata[i, k][1:-1].split('; ')
                listB = metadata[j, k][1:-1].split('; ')
                for elem in listB:
                    scores[i, j] = scores[j, i] = scores[i, j] + (1 if elem in listA else 0)

    np.savetxt("similarities.csv", scores, delimiter=",")

    return scores

def get_top_metadata_match(music, score_matrix, top=10):
    music_index = music_dict[music]
    scores = score_matrix[music_index, :]
    scores_sorted = np.argsort(scores, )[-top:][::-1].astype('int16')
    return [musics[i] for i in scores_sorted[:top]]


def metadataRanking(matrix):

    top = []
    for music in get_musics(query_path):
        top_matches = get_top_metadata_match(music, matrix)
        print(f"Metadata for {music} :")
        top.append(f"Metadata for {music}:")
        for match in top_matches:
            score = matrix[music_dict[music], music_dict[match]]
            print(f"-> Music {match}: {score}")
            top.append(f" {match}: {score}")

    np.savetxt("metadataRanking.csv", top, fmt="%s", delimiter=",")
    return top_matches

def precision(m1, m2, top):
    precision_score = len(np.intersect1d(m1, m2)) / len(m2) *100
    print(f"Precision {top}: {precision_score}")
    return precision_score
        
if __name__ == '__main__':

    audio_folder_path = "C:\\Users\\User\\Desktop\\DEI\\3ano\\2Semestre\\Multimedia\\PL\\TP2\\MER_audio_taffc_dataset\\AllQ"
    query_path = "C:\\Users\\User\\Desktop\\DEI\\3ano\\2Semestre\\Multimedia\\PL\\TP2\\Queries"
    features_list = extract_features(audio_folder_path, "featuresStats.csv")
    features_list_normalized, min_value, max_value = normalized_features(0,1,features_list,"featuresNormalized.csv")
    features_list_query = extract_features(query_path, "featuresStatsQuery.csv")
    features_list_normalized_query = normalized_features_query(0,1,features_list_query, min_value, max_value, "featuresNormalizedQuery.csv")
    results = calculate_and_compare(audio_folder_path)
    euclideanDistance, manhattanDistance, cosineDistance = similarity_measurements(features_list_normalized_query[2], features_list_normalized)
    top_10_euclidean, top_10_manhattan, top_10_cosine = create_similarity_rankings(euclideanDistance, manhattanDistance, cosineDistance, audio_folder_path)

    datas = "C:\\Users\\User\\Desktop\\DEI\\3ano\\2Semestre\\Multimedia\\PL\\TP2\\panda_dataset_taffc_metadata.csv"

    musics = get_musics(audio_folder_path)
    music_dict = dict((music, i) for i, music in enumerate(musics))
    metadata = metadataExtraction(datas)
    top_matches = metadataRanking(metadata)
    precision_euclidean = precision([music[0] for music in top_10_euclidean], top_matches,"Euclidean")
    precision_manhattan = precision([music[0] for music in top_10_manhattan], top_matches, "Manhattan")
    precision_cosine = precision([music[0] for music in top_10_cosine], top_matches, "Cosine")


