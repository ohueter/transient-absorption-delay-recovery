import matplotlib.pyplot as plt
import numpy as np

from .tadata import TAData
from .tcspcdata import TCSPCData


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
        self._delay_offsets = np.zeros(ta_data.num_steps, dtype=np.int) - 1

        # TODO: positive Werte funktionieren nicht korrekt

        self.ta_data_processed = None
        self.tcspc_data_processed = None
        self.unique_delays = None
        self.delay_statistics = None

        self.all_delays_list = None
        self.tadata_cleaned = None

        self.s_det_pump = 0
        self.s_det_probe = 0
        self.s_el_pump = 0
        self.s_el_probe = 0
        self.c = 299792458 * 1e-9  # c in nanoseconds
        self.fract_c = 0.66  # fraction of c for signal propagation in coaxial cable

    def correct_delays_for_signal_propagation_time(self):
        optical_delay = (self.s_det_pump - self.s_det_probe) / self.c
        electrical_delay = (self.s_el_pump - self.s_el_probe) / (self.fract_c * self.c)
        self.unique_delays = self.unique_delays + optical_delay + electrical_delay

    def remove_pulse_delays_of_missed_shots(pulse_delays: np.ndarray, offset: int):
        pass

    def assign_delays_spectra(self, delay_offsets: np.ndarray = None):
        if delay_offsets is None:
            delay_offsets = self._delay_offsets

        # contains spectra indices that need to be deleted because there are no recorded delay times for them
        spectra_indices_to_delete = np.array([], dtype=np.int)

        # all delay times after deleting missed shots
        all_delays = np.array([])

        self.all_delays_list = []

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
                raise ValueError("not enough delay values at idx", idx)

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
            self.all_delays_list.append(
                delays[
                    : self.ta_data.num_avg
                    - num_missed_shots_at_this_idx
                    - spectra_offset
                ]
            )
            spectra_indices_to_delete = np.concatenate(
                (spectra_indices_to_delete, missing_delays_indices)
            )

        print("num spectra after processing:", len(all_delays))
        return all_delays, spectra_indices_to_delete

    def process_data(self, delay_offsets: np.ndarray = None):
        delays, spectra_indices = self.assign_delays_spectra(delay_offsets)

        tadata_cleaned = np.delete(self.ta_data.delta_od, spectra_indices, axis=1)
        self.tadata_cleaned = tadata_cleaned

        sorted_indices = np.argsort(delays)
        delays_sorted = delays[sorted_indices]
        tadata_cleaned_sorted = tadata_cleaned[:, sorted_indices]
        unique_delays = np.unique(delays_sorted)

        delay_statistics = np.array([])
        delay_statistics.shape = (0, 2)
        tadata_cleaned_sorted_reduced = np.array([])
        tadata_cleaned_sorted_reduced.shape = (0, self.ta_data.num_pixel)
        for dt in unique_delays:
            dt_indices = np.where(delays_sorted == dt)
            delay_statistics = np.vstack(
                (delay_statistics, np.array([dt, len(dt_indices[0])]))
            )
            tadata_dt_mean = np.mean(tadata_cleaned_sorted[:, dt_indices[0]], axis=1)
            tadata_cleaned_sorted_reduced = np.vstack(
                (tadata_cleaned_sorted_reduced, tadata_dt_mean)
            )
        tadata_cleaned_sorted_reduced = tadata_cleaned_sorted_reduced.T
        self.ta_data_processed = tadata_cleaned_sorted_reduced
        self.tcspc_data_processed = unique_delays
        self.delay_statistics = delay_statistics.T

        # plt.plot(delay_statistics.T[0], delay_statistics.T[1])
        # plt.savefig("delay_statistics.png")

    def _wl_defaults(self, wl_min: int = None, wl_max: int = None):
        if wl_min is None or wl_min < 0 or wl_min >= wl_max:
            wl_min = 0
        if wl_max is None or wl_min >= self.ta_data.num_pixel or wl_max <= wl_min:
            wl_max = self.ta_data.num_pixel - 1
        return wl_min, wl_max

    def plot_spectrum(self, delta_od, wl_min: int = None, wl_max: int = None):
        wl_min, wl_max = self._wl_defaults(wl_min, wl_max)
        x = np.arange(0, wl_max - wl_min)
        spectral_cut = np.mean(delta_od[wl_min:wl_max], axis=1)

        return x, spectral_cut
        # plt.figure()
        # plt.plot(x, spectral_cut)
        # plt.savefig("spectrum.png")
        # plt.close()

    def plot_transient(self, wl_min: int = None, wl_max: int = None):
        wl_min, wl_max = self._wl_defaults(wl_min, wl_max)
        x = np.arange(0, self.ta_data_processed.shape[1])
        temporal_cut = np.mean(self.ta_data_processed[wl_min:wl_max], axis=0)

        return self.tcspc_data_processed[1:-1], temporal_cut[1:-1]
        # plt.figure()
        # plt.plot(self.tcspc_data_processed[1:-1], temporal_cut[1:-1])
        # plt.savefig("temporal_cut.png")
        # plt.show()
        # plt.close()

    def plot_step_transient(self, step_no: int, wl_min: int = None, wl_max: int = None):
        def tadata_indices(all_delays_list, i):
            """Returns a slice to get spectra based on the index of the delays array."""
            begin = sum(map(len, all_delays_list[:i]))
            end = begin + len(all_delays_list[i])
            return slice(begin, end)

        wl_min, wl_max = self._wl_defaults(wl_min, wl_max)

        sorted_indices = np.argsort(self.all_delays_list[step_no])
        delays_sorted = self.all_delays_list[step_no][sorted_indices]
        unique_delays = np.unique(delays_sorted)

        # print("delays_sorted", delays_sorted)
        # print("len delays_sorted", len(delays_sorted))
        # print("unique_delays", unique_delays)

        tadata_slice = tadata_indices(self.all_delays_list, step_no)
        tadata_cleaned_sorted = self.tadata_cleaned[:, tadata_slice][:, sorted_indices]

        # print("shape tadata_cleaned_sorted", tadata_cleaned_sorted.shape)

        tadata_cleaned_sorted_reduced = np.array([])
        tadata_cleaned_sorted_reduced.shape = (0, self.ta_data.num_pixel)
        for dt in unique_delays:
            dt_indices = np.where(delays_sorted == dt)
            tadata_dt_mean = np.mean(tadata_cleaned_sorted[:, dt_indices[0]], axis=1)
            tadata_cleaned_sorted_reduced = np.vstack(
                (tadata_cleaned_sorted_reduced, tadata_dt_mean)
            )
        tadata_cleaned_sorted_reduced = tadata_cleaned_sorted_reduced.T

        temporal_cut = np.mean(tadata_cleaned_sorted_reduced[wl_min:wl_max], axis=0)

        return unique_delays, temporal_cut
