from pathlib import Path

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
        if len(self._missed_shots.shape) == 1:
            self._missed_shots = np.array([self._missed_shots])
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
