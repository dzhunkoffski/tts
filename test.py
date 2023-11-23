import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import tts.model as module_model
from tts.trainer import Trainer
from tts.utils import ROOT_PATH
from tts.utils.object_loading import get_dataloaders
from tts.utils.parse_config import ConfigParser

from tts.text.symbols import symbols

from scipy.io.wavfile import write
import numpy as np

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    min_pitch, max_pitch = dataloaders['train'].dataset.pitch_min, dataloaders['train'].dataset.pitch_max
    print("Pitch range:", min_pitch, max_pitch)
    min_energy, max_energy = dataloaders['train'].dataset.energy_min, dataloaders['train'].dataset.energy_max
    print("Energy range:", min_energy, max_energy)

    # build model architecture
    model = config.init_obj(
        config["arch"], module_model, 
        vocab_size=len(symbols), min_pitch=min_pitch, max_pitch=max_pitch, min_energy=min_energy, max_energy=max_energy
    )
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    write_to = "audio_samples"
    os.mkdir(write_to)
    text_samples = [
        'A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest',
        'Massachusetts Institute of Technology may be best known for its math, science and engineering education',
        'Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space'
    ]
    duration_coeffs = [0.8, 1, 1.2]
    pitch_coeffs = [0.5, 0.8, 1, 1.2, 1.5]
    energy_coeffs = [0.8, 1, 1.2]

    with torch.no_grad():
        for i, text in enumerate(text_samples):
            for dur in duration_coeffs:
                for pitch in pitch_coeffs:
                    for energy in energy_coeffs:
                        output = model.text2voice(
                            text, dataset=dataloaders['train'].dataset,
                            duration_coeff=dur, pitch_coeff=pitch, energy_coeff=energy
                        )
                        write(
                            f'{write_to}/{i}_d{dur}_p{pitch}_e{energy}.wav', 
                            dataloaders['train'].dataset.sample_rate,
                            output['audio']
                        )
                        


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(test_data_folder / "audio"),
                            "transcription_dir": str(
                                test_data_folder / "transcriptions"
                            ),
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output)
