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

        # Filename: <measurement_name>_NS_cmbstatusnumber.dat
        self._status_numbers_path: Path = None
        self._status_numbers: np.ndarray = None

        # counts
        self.num_pixel: int = None
        self.num_spectra: int = None
        self.num_steps: int = None  # refers to number of steps set in TA acquisition program
        self.num_avg: int = None  # averages set in TA acquisition program
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
    def status_numbers(self):
        return self._status_numbers

    @status_numbers.setter
    def status_numbers(self, path: Path):
        self._status_numbers_path = Path(path)
        self._status_numbers = np.loadtxt(self._status_numbers_path, dtype=np.int)
        self.num_avg, _ = self._status_numbers.shape

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
    def delays(self) -> List[np.ndarray]:
        return self._delays


class TATCSCPAnalysis:
    def __init__(self, ta_data: TAData, tcspc_data: TCSPCData):
        self.ta_data = ta_data
        self.tcspc_data = tcspc_data

        # positive values means shift pulse delays forward:
        #
        # for +1:
        # pulse delays:        | 1 | 2 | 3 | 4 | 5 | ...
        # spectra:         | 1 | 2 | 3 | 4 | 5 | ...
        #
        #  for -1:
        # pulse delays:    | 1 | 2 | 3 | 4 | 5 | ...
        # spectra:             | 1 | 2 | 3 | 4 | 5 | ...
        self._delay_offsets = np.zeros(ta_data.num_steps, dtype=np.int)

    def remove_pulse_delays_of_missed_shots(pulse_delays: np.ndarray, offset: int):
        pass

    def opt(self, delay_offsets: np.ndarray = None):
        if delay_offsets is None:
            delay_offsets = self._delay_offsets

        # contains spectra indices that need to be deleted because there are no recorded delay times for them
        spectra_indices_to_delete = np.array([], dtype=np.int)

        # all delay times after deleting missed shots
        all_delays = np.array([])

        # total missed shots
        n_missed_shots = 0

        for idx, delays in enumerate(self.tcspc_data.delays):
            delay_offset = 0
            spectra_offset = 0
            if delay_offsets[idx] < 0:
                delay_offset = abs(delay_offsets[idx])
            elif delay_offsets[idx] > 0:
                spectra_offset = delay_offsets[idx]

            delays = delays[delay_offset : delay_offset + self.ta_data.num_avg].copy()
            num_pulse_delays = len(delays)
            if num_pulse_delays < self.ta_data.num_avg:
                # not enough recorded delay times
                raise

            # first delete delays for which no spectra were recorded (because of "missed shots")
            missed_shots_at_this_idx = self.ta_data.missed_shots.T[idx][
                self.ta_data.missed_shots.T[idx] >= 0
            ]
            num_missed_shots_at_this_idx = len(missed_shots_at_this_idx)
            if num_missed_shots_at_this_idx > 0:
                # if idx in delay_steps_with_missed_shots:
                missed_shots_at_this_idx -= spectra_offset

                # make sure array indices are not out of bounds
                if (
                    np.sum(missed_shots_at_this_idx >= num_pulse_delays) == 0
                    and np.sum(missed_shots_at_this_idx < 0) == 0
                ):
                    delays = np.delete(delays, missed_shots_at_this_idx)
                    num_pulse_delays -= num_missed_shots_at_this_idx
                    # n_avg_subtract = num_missed_shots_at_this_idx

            # second: after cropping and deleting delays, there may be more spectra than delay times.
            # we need to check that and delete the corresponding spectra.
            # also: implement spectra offset
            n_missed_shots += num_missed_shots_at_this_idx
            missing_delays_indices = (
                idx * self.ta_data.num_avg
                - n_missed_shots
                + np.arange(0, spectra_offset)
            )
            all_delays = np.concatenate(
                (
                    all_delays,
                    delays[
                        : self.ta_data.num_avg
                        - num_missed_shots_at_this_idx
                        - spectra_offset
                    ],
                )
            )
            spectra_indices_to_delete = np.concatenate(
                (spectra_indices_to_delete, missing_delays_indices)
            )

        print(len(all_delays))


if __name__ == "__main__":
    ta_data = TAData()
    ta_data.delta_od = "../HQC/HQC_MeOH_387nm_66ns_2001ms_0fs/TA/0001/HQC_MeOH_387nm_NS_2D_DeltaOD_uncor.dat"
    ta_data.missed_shots = "../HQC/HQC_MeOH_387nm_66ns_2001ms_0fs/TA/0001/HQC_MeOH_387nm_NS_missed_shots.dat"
    ta_data.status_numbers = "../HQC/HQC_MeOH_387nm_66ns_2001ms_0fs/TA/0001/HQC_MeOH_387nm_NS_cmbstatusnumber.dat"

    print(ta_data.num_spectra)

    tcspc_data = TCSPCData()
    tcspc_data.delays_directory = "../HQC/HQC_MeOH_387nm_66ns_2001ms_0fs/TCSPC"

    ana = TATCSCPAnalysis(ta_data, tcspc_data)
    ana.opt()
