from matplotlib import colors
from scipy.io import wavfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchaudio
import math

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

config = {"n_mels":  128,
              "batch_size": 8,
              "k_folds": 10,
              "classes": (0,1,2,3,4),
              "resample_freq": 22050,
              "lr": 0.001,
              "main_folder": "/kaggle/input/motorsounds",
              "sound_folder": "sound_files",
              "meta_file": "motor_sounds.csv",
              "num_epochs": 15,
              "num_samples": 1
              }


def plot_audio(sound_files):
    rows = int(math.ceil(len(sound_files)/3))
    columns = len(sound_files) if len(sound_files)<3 else 3
    fig, axs = plt.subplots(rows,columns, squeeze=False)
    fig.set_size_inches(10 * columns, 5 * rows)
    if len(sound_files) == 1:
        _, signal = wavfile.read(sound_files[0])
        axs[0][0].set_title(f"Wave signal for {sound_files[0].name}")
        x = np.linspace(0, 6, len(signal))
        y = signal[:]
        axs[0][0].plot(x, y)
        axs[0][0].set_ylabel('Amplitude')
        axs[0][0].set_xlabel('Time (in s)')
    else:
        for i, (sound_file, ax) in enumerate(zip(sound_files, axs.flat)):
            plt.rc('font', **font)
            _, signal = wavfile.read(sound_file)
            #ax.set_title(f"Wave signal for {sound_file.name}")
            x = np.linspace(0, 6, len(signal))
            y = signal[:]
            ax.plot(x, y)
            ax.set_ylabel('Amplitude')
            ax.set_xlabel('Time (in s)')
    return fig

def plot_spectogram(sound_files):
    rows = int(math.ceil(len(sound_files)/3))
    columns = len(sound_files) if len(sound_files)<3 else 3
    fig, axs = plt.subplots(rows,columns,squeeze=False)
    fig.set_size_inches(10 * columns, 5 * rows)
    if len(sound_files) == 1:
        signal, _ = librosa.load(sound_files[0])
        S = librosa.feature.melspectrogram(signal, sr=22050, n_fft=2048, hop_length=512, n_mels=128)
        S_DB = librosa.power_to_db(S, ref=np.max)
        axs[0][0].set_title(f"Melspectogram for {sound_files[0].name}")
        im = librosa.display.specshow(S_DB, sr=22050, hop_length=512, x_axis='time', y_axis='mel', ax=axs[0][0])
        fig.colorbar(im, format='%+2.0f dB', ax=axs[0][0])

    else:
        for i, (sound_file, ax) in enumerate(zip(sound_files, axs.flat)):
            signal, _ = librosa.load(sound_file)
            S = librosa.feature.melspectrogram(signal, sr=22050, n_fft=2048, hop_length=512, n_mels=128)
            S_DB = librosa.power_to_db(S, ref=np.max)
            #ax.set_title(f"Melspectogram for {sound_file.name}")
            im = librosa.display.specshow(S_DB, sr=22050, hop_length=512, x_axis='time', y_axis='mel', ax=ax)
            fig.colorbar(im, format='%+2.0f dB', ax=ax)
    return fig  

def plot_predictions(outputs, sound_file):
    fig = plt.figure(figsize=(10,5))
    plt.gca()
    plt.bar(config["classes"], outputs.numpy(), color=plt.get_cmap("Pastel1").colors, width = 1, edgecolor = "black")
    
    plt.xlabel("Class Id")
    plt.ylabel("Cross entropy loss for class")
    plt.title(f"Output predictions for {sound_file.name}")
    #plt.tight_layout(pad=0.05)
    plt.show()
    return fig

def create_model(config):
    model = models.resnet18(pretrained=True)
    model.conv1=nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size[0], 
                          stride=model.conv1.stride[0], padding=model.conv1.padding[0])
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(*[nn.Dropout(p=0.25), nn.Linear(num_ftrs, len(config["classes"]))])
    return model

def predict_unknown_sample(file_path, model_path="motor_sounds_model_1628741297.390562.pt", config=config):
    spectogram, _ = load_and_transform_to_spectogram(file_path, config["resample_freq"], config["n_mels"]) 
    spectogram = spectogram.unsqueeze(0)   
    model = create_model(config)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.eval()
    torch.manual_seed(42)
    with torch.no_grad():
        outputs = model(spectogram)
        _, predicted = torch.max(outputs, 1)
    return predicted, outputs[0]


def load_and_transform_to_spectogram(path, resample_freq, n_mels):
    # load audio from file path
    signal, sample_rate = torchaudio.load(path)

    signal = torch.mean(signal, dim=0, keepdim=True)

    # resample audio signal to different frequency
    resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=resample_freq)
    signal = resample_transform(signal)

    # generate melspectogram out of the signal
    melspectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
    melspectrogram = melspectrogram_transform(signal)
    melspectogram_db = torchaudio.transforms.AmplitudeToDB()(melspectrogram)

    return melspectogram_db, signal

label_format = {None:None,
                0:"0 (Class Id 0)",
                1:"1 (Class Id 1)",
                2:"2 (Class Id 2)",
                3:"1* (Class Id 3)",
                4:"7 (Class Id 4)"}