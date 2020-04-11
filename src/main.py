"""delay recovery"""

import os
from pathlib import Path
from typing import List

import numpy as np


class TAData:
    def __init__(self):
        # Filename: <measurement_name>_NS_2D_DeltaOD_uncor.dat
        self._delta_od_path: Path = None
        self._delta_od: np.ndarray = None

        # Filename: <measurement_name>_NS_missed_shots.dat
        self._missed_shots_path: Path = None
        self._missed_shots: np.ndarray = None

        # counts
        self.num_pixel: int = None
        self.num_spectra: int = None
        self.num_steps: int = None  # refers to number of steps set in TA acquisition program
        self.num_missed_shots: int = None

        # delay step indices where shots were missed
        self.delay_steps_with_missed_shots: np.ndarray = None
        # number of missed shots in delay steps with missed shots
        self.missed_shots_per_delay_step: np.ndarray = None

    @property
    def delta_od(self) -> np.ndarray:
        return self._delta_od

    @delta_od.setter
    def delta_od(self, path: Path):
        self._delta_od_path = Path(path)
        self._delta_od = np.loadtxt(self._delta_od_path).T
        self.num_pixel, self.num_spectra = self._delta_od.shape

    @property
    def missed_shots(self) -> np.ndarray:
        return self._missed_shots

    @missed_shots.setter
    def missed_shots(self, path: Path):
        self._missed_shots_path = Path(path)
        self._missed_shots = np.loadtxt(self._missed_shots_path, dtype=np.int)
        (_, self.num_steps) = self._missed_shots.shape
        self._process_missed_shots()
        self._calculate_missed_shots_per_delay_step()

    def _process_missed_shots(self):
        # 4 laser shots per index
        self._missed_shots[self._missed_shots > 0] = (
            self._missed_shots[self._missed_shots > 0] / 4
        )
        self.num_missed_shots = np.sum(self._missed_shots > 0)

        # "falsche" 0en, die von Labview erzeugt wurden, durch -1 ersetzen
        for i, row in enumerate(self._missed_shots.T):
            if len(row) <= 1:
                # nur eine spalte vorhanden, nichts zu tun
                break

            if row[0] == -1:
                # wenn der erste wert -1 ist, kommen danach keine anderen werte mehr
                row[1:] = -1
                continue

            (where_zero,) = np.where(row[1:] <= row[0])
            if where_zero.size > 0:
                idx_val_smaller = np.min(where_zero)
                row[idx_val_smaller + 1 :] = -1

    def _calculate_missed_shots_per_delay_step(self):
        (self.delay_steps_with_missed_shots,) = np.where(self._missed_shots[0] > 0)
        self.missed_shots_per_delay_step = np.array(
            [
                np.where(self._missed_shots.T[idx] > 0)[0].size
                for idx in self.delay_steps_with_missed_shots
            ],
            dtype=np.int,
        )
        assert len(self.delay_steps_with_missed_shots) == len(
            self.missed_shots_per_delay_step
        )


class TCSPCData:
    DELAYS_FILENAME = "delays.dat"

    def __init__(self):
        self._delays_directory_path: Path = None
        self._delays_files: List[Path] = None
        self._delays: np.ndarray = None

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
    def delays(self):
        return self._delays


class TATCSCPAnalysis:
    def __init__(self, ta_data: TAData, tcspc_data: TCSPCData):
        self.ta_data = ta_data
        self.tcspc_data = tcspc_data


if __name__ == "__main__":
    ta_data = TAData()
    # ta_data.delta_od = "../HQC/HQC_MeOH_387nm_66ns_2001ms_0fs/TA/0001/HQC_MeOH_387nm_NS_2D_DeltaOD_uncor.dat"
    # ta_data.missed_shots = "../HQC/HQC_MeOH_387nm_66ns_2001ms_0fs/TA/0001/HQC_MeOH_387nm_NS_missed_shots.dat"

    tcspc_data = TCSPCData()
    tcspc_data.delays_directory = "../HQC/HQC_MeOH_387nm_66ns_2001ms_0fs/TCSPC"

    ana = TATCSCPAnalysis(ta_data, tcspc_data)

quit()


# daten laden

tadata = np.loadtxt(
    "./HQC_MeOH_387nm_66ns_2001ms_0fs/TA/0001/HQC_MeOH_387nm_NS_2D_DeltaOD_uncor.dat"
)
tadata = tadata.T
n_pixel, n_spectra = tadata.shape

avg_tadata = np.loadtxt(
    "./HQC_MeOH_387nm_66ns_2001ms_0fs/TA/0001/HQC_MeOH_387nm_Delta_OD_avg_uncor.dat"
)

delays = np.loadtxt(
    "./HQC_MeOH_387nm_66ns_2001ms_0fs/TA/0001/HQC_MeOH_387nm_delays.dat"
)
(n_steps,) = delays.shape


missed_shots = np.loadtxt(
    "./HQC_MeOH_387nm_66ns_2001ms_0fs/TA/0001/HQC_MeOH_387nm_NS_missed_shots.dat",
    dtype=np.int,
)
# 4 laser shots per index
missed_shots[missed_shots > 0] = missed_shots[missed_shots > 0] / 4
n_missed_shots = np.sum(missed_shots > 0)


# "falsche" 0en, die von Labview erzeugt wurden, durch -1 ersetzen
for i, row in enumerate(missed_shots.T):
    if len(row) <= 1:
        # nur eine spalte vorhanden, nichts zu tun
        break

    if row[0] == -1:
        # wenn der erste wert -1 ist, kommen danach keine anderen werte mehr
        row[1:] = -1
        continue

    (where_zero,) = np.where(row[1:] <= row[0])
    if where_zero.size > 0:
        idx_val_smaller = np.min(where_zero)
        row[idx_val_smaller + 1 :] = -1

missed_shots.T

(delay_steps_with_missed_shots,) = np.where(missed_shots[0] > 0)
number_missed_shots_per_delay_step = np.array(
    [
        np.where(missed_shots.T[idx] > 0)[0].size
        for idx in delay_steps_with_missed_shots
    ],
    dtype=np.int,
)


# delays sortieren


status_no = np.loadtxt(
    "./HQC_MeOH_387nm_66ns_2001ms_0fs/TA/0001/HQC_MeOH_387nm_NS_cmbstatusnumber.dat"
)
n_avg, n_steps = status_no.shape

delay_folder = sorted(
    [
        f
        for f in os.listdir("./HQC_MeOH_387nm_66ns_2001ms_0fs/TCSPC")
        if not f.startswith(".")
    ],
    key=lambda f: f.lower(),
)

delay_path = [
    "./HQC_MeOH_387nm_66ns_2001ms_0fs/TCSPC/" + folder + "/delays.dat"
    for folder in delay_folder
]

# for measurement with too few delays, does not work, just kept for reference

spectra_indices_to_delete = np.array([], dtype=np.int)
all_delays = np.array([])
num_delays = np.array([], dtype=int)
delays_too_few = 0

# falls schon delays aufgezeichnet wurden, bevor spektren gemessen wurden
offset_delays = np.zeros(n_steps, dtype=np.int) + 2

# falls schon spektren gemessen wurden, bevor delays aufgezeichnet wurden
offset_spectra = np.zeros(n_steps, dtype=np.int) + 0

for idx, delays_file in enumerate(delay_path):
    if idx in delay_steps_with_missed_shots:
        continue
    else:
        delays = np.loadtxt(delays_file)
        delays = delays[offset_delays[idx] : offset_delays[idx] + n_avg]
        if len(delays) < n_avg:
            print(idx)
            raise
        all_delays = np.concatenate((all_delays, delays))
        num_delays = np.concatenate((num_delays, [len(delays)]))
