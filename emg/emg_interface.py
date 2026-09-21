"""
emg/emg_interface.py — Isolated Electromyography (EMG) Interface

STATUS: DISABLED BY DEFAULT (config.EMG_ENABLED = False)

IMPORTANT CLINICAL & SYSTEM DESIGN NOTICE:
------------------------------------------
1. NO EMG HARDWARE IS CURRENTLY REQUIRED OR CONNECTED.
2. EMG_ENABLED = False is the default configuration. The entire application runs
   identically and smoothly using purely optical/camera hand tracking.
3. DO NOT FAKE EMG MEASUREMENTS:
   This module strictly avoids generating synthetic, pseudo-random, or fake
   EMG values when hardware is absent or disabled. When disabled/disconnected,
   functions return None (or appropriate disabled status strings) rather than
   fabricating fake sensor metrics (such as fake RMS, peak, or activation levels).
4. HARDWARE ISOLATION GUARANTEE:
   No physical serial ports, bluetooth sockets, or third-party hardware libraries
   (e.g., bitalino, myo, pylsl) are imported or initialized when EMG_ENABLED is False.

=============================================================================
  SYSTEM ARCHITECTURE
=============================================================================

  CURRENT ACTIVE PIPELINE (Camera-only tracking):
  ┌─────────────────────────┐
  │         CAMERA          │  (Webcam input stream)
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │  OpenCV HAND TRACKING   │  (Index fingertip / wrist landmark detection)
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │     PLAYER POSITION     │  (Smoothed 2D coordinates: px, py)
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │        MAZE GAME        │  (Corridor navigation & continuous collision engine)
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │   PERFORMANCE METRICS   │  (Accuracy, Efficiency, Smoothness, Collisions)
  └─────────────────────────┘

  FUTURE PIPELINE (Camera Tracking + EMG Sensor Fusion):
  ┌─────────────────────────┐
  │       EMG SENSOR        │  (Surface electrodes: Biceps / Forearm / Extensors)
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │     EMG PROCESSING      │  (Bandpass 20-450Hz, Notch 50/60Hz, Rectification)
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │ MUSCLE ACTIVATION DATA  │  (RMS, Peak, MVC Normalized Activation [0.0 - 1.0])
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │    PERFORMANCE DATA     │  (Correlate muscle effort vs kinematic smoothness)
  └─────────────────────────┘

=============================================================================
  FUTURE INTEGRATION POINTS (Where to enable when hardware arrives)
=============================================================================
  1. config.py:
     - Set EMG_ENABLED = True
     - Configure EMG_PORT ("COM3", "/dev/ttyUSB0"), EMG_DEVICE ("bitalino", "myo"),
       EMG_SAMPLE_RATE (e.g. 1000 Hz), and EMG_CHANNELS (e.g. (0, 1)).
  2. main.py:
     - In RehabGame.__init__(): call initialize_emg()
     - In RehabGame._game_loop(): call read_emg() & process_emg() each frame
     - In RehabGame._cleanup(): call close_emg()
  3. game/game_engine.py:
     - Record muscle activation concurrently with trajectory points (x, y, t, activation).
  4. metrics/session_recorder.py:
     - Save mean/peak activation metrics in session CSV and JSON records.
"""

from __future__ import annotations
import math
import logging
from typing import Optional, List, Dict, Sequence, Any, Tuple

try:
    import config
except ImportError:
    config = None  # type: ignore

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  EMG Interface Class
# ─────────────────────────────────────────────────────────────────────────────

class EMGInterface:
    """
    Isolated abstraction layer over EMG hardware.

    All methods are safe no-ops that return None or empty collections when
    EMG_ENABLED is False or hardware is not connected.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        device: Optional[str] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[Tuple[int, ...]] = None,
        mvc_reference: Optional[float] = None,
    ) -> None:
        self.port: str = port or getattr(config, "EMG_PORT", "COM3")
        self.device: str = device or getattr(config, "EMG_DEVICE", "bitalino")
        self.sample_rate: int = sample_rate or getattr(config, "EMG_SAMPLE_RATE", 1000)
        self.channels: Tuple[int, ...] = channels or getattr(config, "EMG_CHANNELS", (0,))
        self.mvc_reference: float = mvc_reference or getattr(config, "EMG_MVC_REFERENCE", 512.0)

        self._connected: bool = False
        self._device_handle: Any = None
        self._latest_raw: Optional[List[float]] = None
        self._latest_processed: Optional[Dict[str, float]] = None

    # ── Connection Lifecycle ─────────────────────────────────────────────────

    def initialize_emg(
        self,
        port: Optional[str] = None,
        device: Optional[str] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[Tuple[int, ...]] = None,
    ) -> bool:
        """
        Initialize and connect to the EMG sensor hardware.

        When EMG_ENABLED is False (default), hardware initialization is bypassed
        safely and False is returned. No hardware drivers are invoked.

        Returns:
            bool: True if physically connected, False otherwise.
        """
        if port is not None:
            self.port = port
        if device is not None:
            self.device = device
        if sample_rate is not None:
            self.sample_rate = sample_rate
        if channels is not None:
            self.channels = channels

        # Hardware connection guard
        emg_enabled = getattr(config, "EMG_ENABLED", False) if config else False
        if not emg_enabled:
            log.info("EMG hardware disabled (config.EMG_ENABLED=False). Camera hand tracking active.")
            self._connected = False
            return False

        # ── FUTURE HARDWARE INITIALIZATION DRIVER BLOCK (UNCOMMENT WHEN SENSOR PRESENT) ──
        #
        # try:
        #     if self.device.lower() == "bitalino":
        #         # Install: pip install bitalino
        #         from bitalino import BITalino
        #         self._device_handle = BITalino(self.port)
        #         self._device_handle.start(self.sample_rate, list(self.channels))
        #         self._connected = True
        #         log.info(f"BITalino EMG connected on {self.port} at {self.sample_rate} Hz.")
        #         return True
        #
        #     elif self.device.lower() == "myo":
        #         # Install: pip install myo-python
        #         import myo
        #         myo.init()
        #         hub = myo.Hub()
        #         # Set up custom Myo listener...
        #         self._device_handle = hub
        #         self._connected = True
        #         log.info("Myo Armband EMG connected successfully.")
        #         return True
        #
        #     elif self.device.lower() == "custom_serial":
        #         # Install: pip install pyserial
        #         import serial
        #         self._device_handle = serial.Serial(self.port, baudrate=115200, timeout=0.1)
        #         self._connected = True
        #         log.info(f"Custom serial ADC connected on {self.port}.")
        #         return True
        #
        #     elif self.device.lower() == "lsl":
        #         # Install: pip install pylsl
        #         from pylsl import StreamInlet, resolve_byprop
        #         streams = resolve_byprop('type', 'EMG', timeout=2)
        #         if streams:
        #             self._device_handle = StreamInlet(streams[0])
        #             self._connected = True
        #             log.info("Lab Streaming Layer (LSL) EMG stream connected.")
        #             return True
        #         log.warning("No LSL EMG stream detected.")
        #         return False
        #
        # except Exception as err:
        #     log.error(f"Failed to connect to EMG sensor ({self.device} on {self.port}): {err}")
        #     self._connected = False
        #     return False
        # ───────────────────────────────────────────────────────────────────────────────

        return False

    def close_emg(self) -> None:
        """
        Safely shut down the EMG sensor connection and release hardware resources.
        """
        if not self._connected:
            self._device_handle = None
            self._latest_raw = None
            self._latest_processed = None
            return

        # ── FUTURE HARDWARE DISCONNECT DRIVER BLOCK (UNCOMMENT WHEN SENSOR PRESENT) ──
        #
        # try:
        #     if self.device.lower() == "bitalino" and self._device_handle is not None:
        #         self._device_handle.stop()
        #         self._device_handle.close()
        #     elif self.device.lower() == "custom_serial" and self._device_handle is not None:
        #         self._device_handle.close()
        #     elif self.device.lower() == "myo" and self._device_handle is not None:
        #         self._device_handle.stop()
        #     log.info("EMG hardware interface disconnected.")
        # except Exception as err:
        #     log.warning(f"Error during EMG disconnection (ignored): {err}")
        # ───────────────────────────────────────────────────────────────────────────────

        self._connected = False
        self._device_handle = None
        self._latest_raw = None
        self._latest_processed = None

    # ── Signal Acquisition (Do NOT Fake Measurements) ─────────────────────────

    def read_emg(self) -> Optional[List[float]]:
        """
        Acquire the latest raw EMG frame from physical hardware.

        CRITICAL INTEGRITY PRINCIPLE:
        Do NOT generate fake EMG values and present them as real sensor measurements.
        When hardware is absent or disabled, this returns None.

        Returns:
            List[float] of raw sensor voltages/ADC units per channel, or None if disabled/disconnected.
        """
        if not self.is_ready():
            return None

        # ── FUTURE HARDWARE STREAM READING (UNCOMMENT WHEN SENSOR PRESENT) ────────────
        #
        # try:
        #     if self.device.lower() == "bitalino":
        #         # Read n samples from buffer
        #         data = self._device_handle.read(1)
        #         # Columns 5+ contain analog channel samples
        #         samples = [float(data[0, 5 + ch]) for ch in range(len(self.channels))]
        #         self._latest_raw = samples
        #         return samples
        #
        #     elif self.device.lower() == "custom_serial":
        #         line = self._device_handle.readline().decode("utf-8", errors="ignore").strip()
        #         if line:
        #             samples = [float(x) for x in line.split(",") if x.strip()]
        #             if len(samples) == len(self.channels):
        #                 self._latest_raw = samples
        #                 return samples
        #
        #     elif self.device.lower() == "lsl":
        #         sample, _ = self._device_handle.pull_sample(timeout=0.01)
        #         if sample:
        #             samples = [float(v) for v in sample[:len(self.channels)]]
        #             self._latest_raw = samples
        #             return samples
        #
        # except Exception as err:
        #     log.warning(f"Error reading physical EMG frame: {err}")
        #     return None
        # ───────────────────────────────────────────────────────────────────────────────

        return None

    # ── Signal Processing & Metrics Calculation ───────────────────────────────

    def process_emg(self, raw_samples: Optional[Sequence[float]] = None) -> Optional[Dict[str, float]]:
        """
        Process a window of raw EMG samples into clinically meaningful metrics:
          - RMS (Root Mean Square amplitude in microvolts or ADC units)
          - Peak amplitude
          - MVC Normalized Activation Level [0.0 - 1.0]
          - Mean Absolute Value (MAV)

        Args:
            raw_samples: Optional sequence of raw float voltage/ADC values.
                         If None, uses the latest frame read via read_emg().

        Returns:
            Dictionary with processed metrics, or None if no physical signal is available.
            Does NOT invent fake values.
        """
        samples = raw_samples if raw_samples is not None else self._latest_raw
        if samples is None or len(samples) == 0:
            return None

        # Digital Signal Processing Pipeline:
        # 1. Zero-mean baseline centering
        mean_val = sum(samples) / float(len(samples))
        centered = [x - mean_val for x in samples]

        # 2. Rectification (absolute values)
        abs_vals = [abs(x) for x in centered]
        mav = sum(abs_vals) / float(len(abs_vals))
        peak = max(abs_vals) if abs_vals else 0.0

        # 3. Root Mean Square (RMS) calculation
        sum_sq = sum(x * x for x in centered)
        rms = math.sqrt(sum_sq / float(len(centered)))

        # 4. Voluntary muscle activation normalized against MVC reference [0.0, 1.0]
        mvc = max(self.mvc_reference, 1.0)
        activation = min(max(rms / mvc, 0.0), 1.0)

        processed = {
            "rms": rms,
            "peak": peak,
            "mean_abs": mav,
            "activation_level": activation,
        }
        self._latest_processed = processed
        return processed

    def get_activation_level(self) -> Optional[float]:
        """
        Return the latest normalized muscle activation level in [0.0, 1.0].

        Returns None if EMG is disabled or disconnected.
        Does NOT return fake or arbitrary measurements.
        """
        if not self.is_ready():
            return None
        if self._latest_processed is not None:
            return self._latest_processed.get("activation_level")
        return None

    def get_rms(self) -> Optional[float]:
        """
        Return the Root Mean Square (RMS) of recent EMG window.

        Returns None if EMG is disabled or disconnected.
        Does NOT return fake or arbitrary measurements.
        """
        if not self.is_ready():
            return None
        if self._latest_processed is not None:
            return self._latest_processed.get("rms")
        return None

    def get_peak(self) -> Optional[float]:
        """
        Return the peak amplitude of recent EMG window.

        Returns None if EMG is disabled or disconnected.
        Does NOT return fake or arbitrary measurements.
        """
        if not self.is_ready():
            return None
        if self._latest_processed is not None:
            return self._latest_processed.get("peak")
        return None

    # ── Status and Diagnostic Properties ─────────────────────────────────────

    def is_ready(self) -> bool:
        """True only if EMG_ENABLED is True AND physical hardware is connected."""
        emg_enabled = getattr(config, "EMG_ENABLED", False) if config else False
        return emg_enabled and self._connected

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_status_string(self) -> str:
        """
        Return user-facing status string for UI display.
        Guaranteed: If disabled, returns exactly 'EMG: Not Connected / Disabled'.
        """
        emg_enabled = getattr(config, "EMG_ENABLED", False) if config else False
        if not emg_enabled:
            return "EMG: Not Connected / Disabled"
        if not self._connected:
            return "EMG: Sensor Disconnected"
        return f"EMG: Connected ({self.device})"


# ─────────────────────────────────────────────────────────────────────────────
#  Global Module-Level API (Convenient Functional Interface)
# ─────────────────────────────────────────────────────────────────────────────

_default_emg_interface = EMGInterface()


def initialize_emg(
    port: Optional[str] = None,
    device: Optional[str] = None,
    sample_rate: Optional[int] = None,
    channels: Optional[Tuple[int, ...]] = None,
) -> bool:
    """Initialize and connect to EMG hardware (disabled when config.EMG_ENABLED=False)."""
    return _default_emg_interface.initialize_emg(
        port=port, device=device, sample_rate=sample_rate, channels=channels
    )


def read_emg() -> Optional[List[float]]:
    """Read raw sample frame from physical sensor, or None if disabled/absent."""
    return _default_emg_interface.read_emg()


def process_emg(raw_samples: Optional[Sequence[float]] = None) -> Optional[Dict[str, float]]:
    """Calculate RMS, peak, and activation metrics from real data, or None if disabled."""
    return _default_emg_interface.process_emg(raw_samples=raw_samples)


def get_activation_level() -> Optional[float]:
    """Get muscle activation level [0.0, 1.0], or None if disabled. No fake values."""
    return _default_emg_interface.get_activation_level()


def get_rms() -> Optional[float]:
    """Get RMS of recent signal, or None if disabled. No fake values."""
    return _default_emg_interface.get_rms()


def get_peak() -> Optional[float]:
    """Get peak amplitude of recent signal, or None if disabled. No fake values."""
    return _default_emg_interface.get_peak()


def close_emg() -> None:
    """Safely shut down EMG sensor hardware."""
    _default_emg_interface.close_emg()


def get_emg_status_string() -> str:
    """
    Return UI status indicator string.
    Returns 'EMG: Not Connected / Disabled' when EMG_ENABLED=False.
    """
    return _default_emg_interface.get_status_string()


def is_emg_enabled() -> bool:
    """Return True if config.EMG_ENABLED is True, False otherwise."""
    return bool(getattr(config, "EMG_ENABLED", False)) if config else False


def is_emg_connected() -> bool:
    """Return True if physical EMG hardware is actively connected."""
    return _default_emg_interface.is_connected
