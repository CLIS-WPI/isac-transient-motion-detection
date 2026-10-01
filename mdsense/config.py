"""All tunable constants in one place. Everything the go/no-go decision depends on
(budgets, deadline, false-alarm limit, go region, criteria) is fixed here, before
any test episode is generated."""
from dataclasses import dataclass, field
import numpy as np
from scipy.constants import speed_of_light as C


@dataclass(frozen=True)
class RadioConfig:
    fc: float = 3.5e9
    slot_duration: float = 0.5e-3          # 30 kHz SCS -> 0.5 ms slot; at most one probe per slot
    num_re: int = 32                        # sensing REs per probe, comb across the band
    bandwidth: float = 100e6
    radar_position: tuple = (0.0, 0.0, 2.5)
    snr_ref_db: float = 5.0                 # per-RE SNR of a broadside torso at snr_ref_range
    snr_ref_range: float = 5.0

    @property
    def wavelength(self):
        return C / self.fc

    @property
    def re_offsets(self):
        """Baseband offsets of the sensing REs [Hz], f_k = k*df - B/2."""
        df = self.bandwidth / self.num_re
        return np.arange(self.num_re) * df - self.bandwidth / 2

    @property
    def range_bin(self):
        return C / (2 * self.bandwidth)

    @property
    def noise_var(self):
        """Per-RE complex noise variance, fixed by the radio (not by the episode)."""
        sigma_torso = np.pi * 0.30 ** 2     # broadside ellipsoid torso, half-length 0.30 m
        p_ref = sigma_torso * self.wavelength ** 2 / ((4 * np.pi) ** 3 * self.snr_ref_range ** 4)
        return p_ref / 10 ** (self.snr_ref_db / 10)


@dataclass(frozen=True)
class ScenarioConfig:
    duration: float = 5.0
    warmup: float = 1.0                     # no events, no evaluation
    p_event: float = 0.5                    # fraction of episodes containing one target event
    range_xy: tuple = (3.0, 7.0)            # horizontal distance radar -> person [m]
    azimuth_deg: tuple = (-40.0, 40.0)
    facing_deg: tuple = (-60.0, 60.0)       # 0 = facing the radar
    height: tuple = (1.55, 1.95)
    # Target event = fast limb motion (definition fixed in advance)
    event_duration: tuple = (0.3, 0.8)
    event_peak_speed: tuple = (1.5, 4.5)    # peak hand/foot speed [m/s]
    p_arm_event: float = 0.7                # else leg
    # Negative class realism
    breathing_amp: tuple = (0.002, 0.006)
    breathing_freq: tuple = (0.2, 0.35)
    sway_amp: tuple = (0.003, 0.008)
    jitter_amp_deg: float = 1.5
    nuisance_rate: float = 1.2              # mean nuisance motions per episode (Poisson)
    # Environment
    num_clutter: int = 6
    clutter_range: tuple = (1.0, 10.0)
    clutter_rcs: tuple = (0.3, 3.0)
    drift_amp: float = 2e-3                 # slow common gain drift -> imperfect clutter removal
    drift_phase: float = 2e-3               # [rad]
    comm_busy_prob: float = 0.3             # slots unavailable for sensing (irregular OFDM opportunities)


@dataclass(frozen=True)
class DUConfig:
    zone: tuple = (2.0, 9.0)                # monitored range zone [m] (deployment config, not truth)
    clutter_tau: float = 1.0                # EMA time constant for background tracking [s]
    doppler_max: float = 250.0              # evaluation grid, further capped by the window's Nyquist
    doppler_step: float = 5.0
    f_cut: float = 15.0                     # "fast" Doppler boundary [Hz]
    min_samples: int = 4
    # Dropped/busy probes are not reconstructed. Nyquist of a window is
    # 1/(2 * median successive interval of delivered samples), then capped by doppler_max.


@dataclass(frozen=True)
class EnvConfig:
    tick_slots: int = 4                     # policy is invoked every 2 ms
    ctrl_delay_slots: int = 1               # request -> scheduler (local DU path; no E2 here)
    lead_slots: int = 1                     # scheduler needs >= 1 slot notice
    proc_delay_slots: int = 2               # acquisition -> delivery to the policy


@dataclass(frozen=True)
class DetectionConfig:
    deadline: float = 0.20                  # detect within 200 ms and before the event ends
    fa_grace: float = 0.5                   # alarms up to 0.5 s after an event are not FAs
    refractory: float = 1.0
    service_duration: float = 0.2           # dense sensing after every alarm (counted as cost)
    llr_clip: float = 8.0


@dataclass(frozen=True)
class PilotConfig:
    budgets: tuple = (0.025, 0.05, 0.10)    # monitoring probes per slot
    fa_max_per_min: float = 1.0
    n_train: int = 200
    n_val: int = 200
    n_test: int = 300
    uniform_nw_grid: tuple = (8, 16, 32)
    look_grid: tuple = ((8, 4), (16, 4), (16, 8), (32, 4))   # (probes per look, spacing in slots)
    decusum_h_grid: tuple = (2.0, 5.0, 10.0)
    # Pre-registered go region and criteria (do not change after seeing test results)
    go_budgets: tuple = (0.025, 0.05)
    go_max_event_duration: float = 0.6
    go_min_abs_gain: float = 0.10           # P_fail(burst) - P_fail(DE-CuSum) >= 0.10
    go_fa_slack: float = 1.25               # test FA of DE-CuSum <= 1.25 * limit
    go_cost_ratio: tuple = (0.9, 1.1)       # realised monitoring cost vs burst
    # --- registered scientific decisions (lock before the one test evaluation) ---
    go_cost_basis: str = "monitor"          # GO matches monitoring REs/s, not cost_total
    n_neg: int = 200                        # event-free episodes for FA (not used to retune A)
    seed_neg: int = 50_000
    go_fa_neg_slack: float = 1.25           # FA on neg split: DE-CuSum <= slack * fa_max
    ablation_on_val: bool = True            # energy / Doppler / both; val only, not a GO gate
    seed_train: int = 10_000
    seed_val: int = 20_000
    seed_test: int = 40_000                  # 30_000-30_079 were seen during the smoke run; never reuse


@dataclass(frozen=True)
class Config:
    radio: RadioConfig = field(default_factory=RadioConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    du: DUConfig = field(default_factory=DUConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    det: DetectionConfig = field(default_factory=DetectionConfig)
    pilot: PilotConfig = field(default_factory=PilotConfig)


# Locked before the single test split. Copied into preregistration.json.
SCIENTIFIC_DECISIONS = {
    "deadline_s": 0.20,
    "fa_max_per_min": 1.0,
    "snr_ref_db": 5.0,
    "go_budgets": (0.025, 0.05),
    "go_cost_basis": "monitor",
    "cost_total": "reported_not_gating",
    "dropped_samples": "no_reconstruction_nyquist_from_median_delivered_interval",
    "fa_long_negative": "n_neg=200 seed_neg=50000 force_event=False; does not retune A",
    "ablation": "energy vs doppler vs both on validation only; primary detector uses both",
    "test_seeds": "40000+ ; smoke 30000-30079 never reused",
}
