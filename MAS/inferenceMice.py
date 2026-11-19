import os 
from os import path
import shutil
from pathlib import Path

def _dlc():
    import deeplabcut
    return deeplabcut

from MAS.config import *
import glob


def inference_mice(video_path):
    folder = Path(video_path).resolve()
    config = Path(DLCDETECTOR)
    if not config.is_file():
        raise FileNotFoundError(f"DLC model configuration not found: {config}. Set MAS_DLC_CONFIG to your trained project's config.yaml.")
    videos = sorted(str(p) for p in folder.glob('*.mp4'))
    if not videos:
        raise ValueError(f"No MP4 videos found in {folder}")
    destination = folder.parent / 'csv'
    destination.mkdir(parents=True, exist_ok=True)
    # Let DLC determine the trained shuffle's engine; do not reinterpret TF weights as PyTorch.
    return _dlc().analyze_videos(
        config=str(config), videos=videos, shuffle=1,
        save_as_csv=True, destfolder=str(destination), auto_track=True,
    )

def show_pred(video_path, pcutoff):
    folder = Path(video_path).resolve()
    destination = folder.parent / 'csv'
    videos = sorted(str(p) for p in folder.glob('*.mp4'))
    if not videos:
        raise ValueError(f"No MP4 videos found in {folder}")
    return _dlc().create_labeled_video(
        config=DLCDETECTOR, videos=videos, destfolder=str(destination),
        color_by='individual', shuffle=1, pcutoff=pcutoff,
    )
