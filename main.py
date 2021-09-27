import numpy as np
import torch
import matplotlib
from hparams import create_hparams
from model import Tacotron2
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

class MikochiSpeechSynthesis:
    def __init__(self):
        self.hparams = self.set_hparams()
        self.model = self.set_model()
        self.waveglow = self.set_waveglow()
        
    def set_hparams(self):
        hparams = create_hparams()
        hparams.sampling_rate = 22050
        return hparams
    
    def set_model(self):
        checkpoint_path = "/content/MikochiSpeechSynthesis/model/ver_1_0_0"
        model = load_model(self.hparams)
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        _ = model.cuda().eval().half()
        return model

    def set_waveglow(self):
        waveglow_path = 'waveglow_256channels_universal_v5.pt'
        waveglow = torch.load(waveglow_path)['model']
        waveglow.cuda().eval().half()
        for k in waveglow.convinv:
            k.float()
        denoiser = Denoiser(waveglow)
        return waveglow
    
    def generate_voice(self, text):
        sequence = np.array(text_to_sequence(text, ['basic_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()
        mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence)
        with torch.no_grad():
            audio = self.waveglow.infer(mel_outputs_postnet, sigma=0.666)
        wav = audio[0].data.cpu().numpy()
        return wav
