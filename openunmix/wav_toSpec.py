import musdb
import os
from tqdm import *
import torch
import torch.nn as nn
from openunmix.transforms import ComplexNorm
from openunmix import transforms
import numpy as np

from typing import Optional, Callable


def wav_toSpec(
        nfft: int = 4096, 
        nhop: int = 1024,
        out_root: str = "../musdb18hq_spec",
        in_root: str = "../musdb18hq",
        encoder: Optional[Callable] = None,
        device: torch.device = torch.device("cpu"),
        subsets: str = "train",
        split: Optional[str] = "train"):
    if os.path.isdir(out_root) == False:
            os.makedirs(out_root)

    if encoder == None:
        stft, _ = transforms.make_filterbanks(
            n_fft=nfft, n_hop=nhop, sample_rate=44100.0
        )
        encoder = torch.nn.Sequential(stft, ComplexNorm(mono=0))
    encoder = encoder.to(device)

    mus = musdb.DB(
                root=in_root,
                is_wav=True,
                subsets=subsets,
                split=split,
                download=False,
            )

    out_split = os.path.join(out_root, split)
    if os.path.isdir(out_split) == False:
        os.makedirs(out_split)

    pbar = tqdm(mus.tracks)

    for track in pbar:
        out_dir = os.path.join(out_split, track.name)
        if os.path.isdir(out_dir) == False:
            os.makedirs(out_dir)
        for src_key in (list(track.sources.keys()) + ["mixture"]):
            out_name = os.path.join(out_dir, src_key)+".pt"
            if os.path.isfile(out_name) == False:
                audio = track.audio.T if src_key == "mixture" else track.sources[src_key].audio.T
                audio = torch.tensor(audio, dtype=torch.float16).to(device)
                spec = encoder(audio).to(torch.device("cpu"))
                print(spec.shape)
                torch.save(spec, out_name)
        
if __name__ == "__main__":
    # wav_toSpec(device=torch.device("mps"), split = "valid")
    wav_toSpec(device=torch.device("mps"), split = "train")
