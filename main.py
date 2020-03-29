import librosa


if __name__ == "__main__":
    import torch
    print("ok")

    original_wav = r"C:\Users\amitk\data_for_experiments\gan_denoising\p232_023.wav"
    downsampled_wav = r"C:\Users\amitk\data_for_experiments\gan_denoising\p232_023_downsampled.wav"
    y, s = librosa.load(original_wav, sr=16000)  # Downsample 44.1kHz to 8kHz
    librosa.output.write_wav(downsampled_wav, y, sr=16000)