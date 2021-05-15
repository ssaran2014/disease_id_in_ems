import datasets
import transformers
import soundfile
import jiwer
import librosa
import torchaudio
#import ffmpeg

from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from datasets import load_dataset, load_metric
import soundfile as sf
import librosa
import torch
import torchaudio
import numpy as np
import os
import sys
from pathlib import Path, PureWindowsPath
import subprocess

#ignore warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

class Transcription():
    """
    Simple class to upload the data in the sound file and transcribe it.
    """
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    #initialize file names
    origin_file = 'audio.wav'
    destination_file = 'rec4.wav'

    file_name = 'rec4.wav'
    file_path = os.path.join('.', file_name)

#    def __init__(self, origin_file):
#        self.origin_file = origin_file


    def change_filename(self):
        "Change the audio file from .oga to .wav"

        if os.path.exists(self.destination_file):
            os.remove(self.destination_file)

        process = subprocess.run(['ffmpeg', '-hide_banner','-i', self.origin_file, self.destination_file])
        if process.returncode != 0:
            raise Exception("Something went wrong")


    def map_to_array(self):
        "Read file and convert to a format that the model can accept"

        self.speech, self.sampling_rate = torchaudio.load(self.origin_file)
        self.resample_rate = 16000
        self.speech = librosa.resample(np.asarray(self.speech).reshape(-1,), self.sampling_rate, self.resample_rate)
        self.speech = librosa.to_mono(self.speech)
        return self.speech, self.resample_rate


    def indicate_transcription(self):
        "Transcribe"

        #self.change_filename()
        self.speech, self.sampling_rate = self.map_to_array()
        input_values = self.tokenizer(self.speech, return_tensors="pt", padding="longest").input_values
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.tokenizer.batch_decode(predicted_ids)
        transcription = ''.join(transcription)
        return transcription.lower()

    def __str__(self):
        return self.indicate_transcription()
