import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

def extract_mfcc(file_path, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract MFCC features from an audio file.
    
    :param file_path: Path to the audio file
    :param n_mfcc: Number of MFCCs to return
    :param n_fft: Length of the FFT window
    :param hop_length: Number of samples between successive frames
    :return: MFCCs feature
    """
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        
        # Transpose to get (n_mfcc, time) shape
        mfccs = mfccs.T
        
        return mfccs
    except Exception as e:
        print("Error processing file {}: {}".format(file_path, str(e)))
        return None

def process_audio_files(path):
    """
    Process .wav files in the given path (file or directory) and extract MFCCs.
    
    :param path: Path to a .wav file or directory containing .wav files
    :return: DataFrame with file names and corresponding MFCCs
    """
    data = []
    
    if os.path.isfile(path):
        if path.endswith(".wav"):
            files = [path]
        else:
            raise ValueError("The file {} is not a .wav file".format(path))
    elif os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".wav")]
    else:
        raise ValueError("The path {} is neither a file nor a directory".format(path))
    
    # Iterate through all files
    for file_path in tqdm(files):
        try:
            # Extract MFCCs
            mfccs = extract_mfcc(file_path)
            
            if mfccs is not None:
                # Calculate mean of MFCCs across time
                mfcc_mean = np.mean(mfccs, axis=0)
                
                # Add to data list
                data.append({
                    'filename': os.path.basename(file_path),
                    'mfcc_features': mfcc_mean.tolist()  # Convert to list for JSON serialization
                })
        except Exception as e:
            print("Error processing file {}: {}".format(file_path, str(e)))
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

def main():
    # Set the path to your .wav file or directory containing .wav files
    audio_path = "/Users/Mrishal/Desktop/Chicken_Audio_Dataset/Unhealthy"

    try:
        # Process the audio file(s)
        result_df = process_audio_files(audio_path)

        # Save the results to a CSV file
        output_file = "poultry_vocalization_mfccs.csv"
        result_df.to_csv(output_file, index=False)

        print("Processing complete. Results saved to {}".format(output_file))
        print("Processed {} files successfully.".format(len(result_df)))
    except Exception as e:
        print("An error occurred: {}".format(str(e)))

def main():
    # Set the path to your .wav file or directory containing .wav files
    audio_path = "/Users/Mrishal/Desktop/Chicken_Audio_Dataset/healthy"

    try:
        # Process the audio file(s)
        result_df = process_audio_files(audio_path)

        # Save the results to a CSV file
        output_file = "poultry_vocalization_mfccs.csv"
        result_df.to_csv(output_file, index=False)

        print("Processing complete. Results saved to {}".format(output_file))
        print("Processed {} files successfully.".format(len(result_df)))
    except Exception as e:
        print("An error occurred: {}".format(str(e)))

def main():
    # Set the path to your .wav file or directory containing .wav files
    audio_path = "/Users/Mrishal/Desktop/Chicken_Audio_Dataset/noise"

    try:
        # Process the audio file(s)
        result_df = process_audio_files(audio_path)

        # Save the results to a CSV file
        output_file = "poultry_vocalization_mfccs.csv"
        result_df.to_csv(output_file, index=False)

        print("Processing complete. Results saved to {}".format(output_file))
        print("Processed {} files successfully.".format(len(result_df)))
    except Exception as e:
        print("An error occurred: {}".format(str(e)))

if __name__ == "__main__":
    main()