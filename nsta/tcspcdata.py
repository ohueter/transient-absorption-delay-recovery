import os
from pathlib import Path
from typing import List

import numpy as np


class TCSPCData:
    DELAYS_FILENAME = "delays.dat"

    def __init__(self):
        self._delays_directory_path: Path = None
        self._delays_files: List[Path] = None
        self._delays: List[np.ndarray] = None

    @property
    def delays_directory(self):
        return self._delays_directory_path

    @delays_directory.setter
    def delays_directory(self, path: Path):
        self._delays_directory_path = Path(path)
        self._delays_files = self._get_delays_files(self._delays_directory_path)
        self._delays = self._load_delays_from_delays_files(self._delays_files)

    @staticmethod
    def _get_delays_folders(delays_directory_path: Path) -> List[str]:
        delays_folder_names = [
            f for f in os.listdir(delays_directory_path) if not f.startswith(".")
        ]
        return sorted(delays_folder_names, key=lambda f: f.lower())

    @staticmethod
    def _load_delays_from_delays_files(delays_files: List[str]):
        return [np.loadtxt(delays_file) for delays_file in delays_files]
        # return np.stack(delays, axis=0)

    @classmethod
    def _get_delays_files(cls, delays_directory_path: Path) -> List[Path]:
        delays_folder_names = cls._get_delays_folders(delays_directory_path)
        delays_files = [
            delays_directory_path / folder / cls.DELAYS_FILENAME
            for folder in delays_folder_names
        ]
        return delays_files

    @property
    def delays(self) -> List[np.ndarray]:
        return self._delays
