import sys
sys.path.append('waveglow/')
import IPython.display as ipd
import numpy as np
import torch
from hparams import create_hparams
from model import Tacotron2
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

hparams = create_hparams()
hparams.sampling_rate = 22050

checkpoint_path = "/content/MikochiSpeechSynthesis/model/ver_1_0_0"
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

waveglow_path = 'waveglow_256channels_universal_v5.pt'
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

phones = input("テキストを入力してください。")
sequence = np.array(text_to_sequence(phones, ['basic_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
