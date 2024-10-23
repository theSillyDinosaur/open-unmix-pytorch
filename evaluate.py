import argparse
import functools
import json
import multiprocessing
from typing import Optional, Union
import os

import musdb
import museval
import torch
import tqdm
import mir_eval

from openunmix import utils, data
from openunmix.transforms import ComplexNorm
from openunmix import transforms
import torchaudio
import soundfile as sf
import numpy as np


def separate_and_evaluate(
    track: musdb.MultiTrack,
    unmix,
    encoder,
    griffin_lim,
    targets: str,
    output_dir: str,
    device: Union[str, torch.device] = "cpu",
) -> str:

    unmix = unmix.to(device)
    unmix = unmix.eval()

    encoder = encoder.to(device)
    with torch.no_grad():

        audio = torch.tensor(track.audio.T, dtype=torch.float16).to(device)
        audio = encoder(audio).unsqueeze(0)

        s = torch.tensor(track.sources[targets].audio.T, dtype=torch.float16).to(device)

        m_hat = unmix(audio).squeeze(0)
        if griffin_lim != None:
            m_hat = m_hat.cpu()
            s_hat = griffin_lim(m_hat).to(device)
            m_hat = m_hat.to(device)
        m = encoder(s)
        

    spec_sdr = 0
    for i in range(2):
        if(m_hat[i, :].shape[0] > m[i, :].shape[0]):
            shape = (m_hat[i, :].shape[0], m_hat[i, :].shape[1] - m[i, :].shape[1])
            m_mono = torch.cat((m[i, :], torch.zeros(shape)))
            m_hat_mono = m_hat[i, :]
            m_delta_mono = torch.norm(m_mono-m_hat_mono, dim=0)
        elif m_hat[i, :].shape[0] < m[i, :].shape[0]:
            shape = (m_hat[i, :].shape[0], m[i, :].shape[1] - m_hat[i, :].shape[1])
            m_hat_mono = torch.cat((m_hat[i, :], torch.zeros(shape)))
            m_mono = m[i, :]
            m_delta_mono = torch.norm(m_mono-m_hat_mono, dim=0)
        else:
            m_hat_mono = m_hat[i, :]
            m_mono = m[i, :]
            m_delta_mono = torch.norm(m_mono-m_hat_mono, dim=0)
        spec_sdr += 10 * torch.log(torch.norm(torch.norm(m_mono, dim=0)) / torch.norm(m_delta_mono))

    wav_sdr = 0
    if(griffin_lim != None):
        if os.path.isdir(output_dir) == False:
            os.mkdir(output_dir)
        save_path = os.path.join(output_dir, str(track))+".wav"
        torchaudio.save(save_path, s_hat.cpu(), 44100)
        for i in range(2):
            if(s_hat[i, :].shape[0] > s[i, :].shape[0]):
                shape = s_hat[i, :].shape[0] - s[i, :].shape[0]
                s_mono = torch.cat((s[i, :], torch.zeros(shape).to(device)))
                s_hat_mono = s_hat[i, :]
            else:
                shape =  s[i, :].shape[0] - s_hat[i, :].shape[0]
                s_hat_mono = torch.cat((s_hat[i, :], torch.zeros(shape).to(device)))
                s_mono = s[i, :]
            wav_sdr += 10 * torch.log(torch.norm(s_mono) / torch.norm(s_mono-s_hat_mono))
    return spec_sdr/2, wav_sdr/2


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="MUSDB18 Evaluation", add_help=False)

    parser.add_argument(
        "--target",
        nargs="+",
        default="vocals",
        type=str,
        help="provide targets to be processed. \
              If none, all available targets will be computed",
    )

    parser.add_argument(
        "--model",
        default="umxl",
        type=str,
        help="path to mode base directory of pretrained models",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Results path where audio evaluation results are stored",
    )

    parser.add_argument("--evaldir", type=str, help="Results path for museval estimates")

    parser.add_argument("--root", type=str, help="Path to MUSDB18")

    parser.add_argument("--subsets", type=str, default="test", help="MUSDB subset (`train`/`test`)")

    parser.add_argument("--cores", type=int, default=1)

    parser.add_argument("--no-gpu", action="store_true", default=False, help="disables CUDA inference")

    parser.add_argument(
        "--is-wav",
        action="store_true",
        default=False,
        help="flags wav version of the dataset",
    )

    parser.add_argument(
        "--niter",
        type=int,
        default=1,
        help="number of iterations for refining results.",
    )

    parser.add_argument(
        "--wiener-win-len",
        type=int,
        default=300,
        help="Number of frames on which to apply filtering independently",
    )

    parser.add_argument(
        "--residual",
        type=str,
        default=None,
        help="if provided, build a source with given name" "for the mix minus all estimated targets",
    )

    parser.add_argument(
        "--aggregate",
        type=str,
        default=None,
        help="if provided, must be a string containing a valid expression for "
        "a dictionary, with keys as output target names, and values "
        "a list of targets that are used to build it. For instance: "
        '\'{"vocals":["vocals"], "accompaniment":["drums",'
        '"bass","other"]}\'',
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="musdb_spec",
        choices=[
            "musdb_spec",
            "musdb",
            "aligned",
            "sourcefolder",
            "trackfolder_var",
            "trackfolder_fix",
        ],
        help="Name of the dataset.",
    )

    # Model Parameters
    parser.add_argument(
        "--log",
        action="store_true",
        help="whether to use log scale"
    )
    parser.add_argument(
        "--seq-dur",
        type=float,
        default=6.0,
        help="Sequence duration in seconds" "value of <=0.0 will use full/variable length",
    )
    parser.add_argument(
        "--unidirectional",
        action="store_true",
        default=False,
        help="Use unidirectional LSTM",
    )
    parser.add_argument("--nfft", type=int, default=4096, help="STFT fft size and window size")
    parser.add_argument("--nhop", type=int, default=1024, help="STFT hop size")
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
        help="hidden size parameter of bottleneck layers",
    )
    parser.add_argument(
        "--bandwidth", type=int, default=16000, help="maximum model bandwidth in herz"
    )
    parser.add_argument(
        "--nb-channels",
        type=int,
        default=2,
        help="set number of channels for model (1, 2)",
    )
    parser.add_argument(
        "--nb-workers", type=int, default=1, help="Number of workers for dataloader."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Speed up training init for dev purposes",
    )

    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )

    args = parser.parse_args()

    use_cuda = not args.no_gpu and torch.cuda.is_available()
    use_mps = not args.no_gpu and torch.backends.mps.is_available()
    device = torch.device("cuda" if use_cuda else ("mps" if use_mps else "cpu"))
    print(args.root)
    dataset, args = data.load_datasets(parser, args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    mus = musdb.DB(
        root=args.root,
        is_wav=True,
        subsets="test",
        split=None,
        download=False,
    )
    print(mus.tracks)
    aggregate_dict = None if args.aggregate is None else json.loads(args.aggregate)

    unmix = utils.load_target_models(
        args.target, model_str_or_path=args.model, device=device, pretrained=True
    )[args.target]
    stft, _ = transforms.make_filterbanks(
        n_fft=args.nfft, n_hop=args.nhop, sample_rate=44100.0
    )
    encoder = torch.nn.Sequential(stft, ComplexNorm(mono=0))
    griffin_lim = torchaudio.transforms.GriffinLim(n_fft=args.nfft, hop_length=args.nhop)


    spec_sdrs, wav_sdrs = 0, 0
    pbar = tqdm.tqdm(mus.tracks)

    for i, track in enumerate(pbar):
        spec_sdr, wav_sdr = separate_and_evaluate(
            track,
            targets=args.target,
            output_dir=args.outdir,
            device=device,
            griffin_lim=griffin_lim if isinstance(args.outdir, str) else None,
            unmix=unmix,
            encoder=encoder
        )
        spec_sdrs += spec_sdr
        wav_sdrs += wav_sdr
        pbar.set_description(f"Current spec_sdr = {spec_sdr}, Avg spec_sdr = {spec_sdrs/(i+1)}")
    print(f"Spec SDR = {spec_sdrs/len(pbar)}, Wav SDR = {wav_sdrs/len(pbar)}")
        