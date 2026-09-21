"""
emg_interface.py — EMG (Electromyography) Hardware Interface

STATUS: DISABLED  (config.EMG_ENABLED = False)

This module is intentionally preserved so that EMG functionality can be
re-enabled in the future without rewriting any code.

─────────────────────────────────────────────────────────────────────
HOW TO RE-ENABLE:
  1. Install the required library for your device:
       pip install pyserial          # BITalino, generic serial
       pip install MyoPy             # Myo Armband
  2. Set in config.py:
         EMG_ENABLED = True
         EMG_PORT    = "COM3"        # or "/dev/ttyUSB0" on Linux
         EMG_DEVICE  = "bitalino"    # or "myo"
  3. Uncomment the implementation blocks inside each method below.
─────────────────────────────────────────────────────────────────────

Supported devices (planned):
  - BITalino (r)evolution kit  (serial / bluetooth)
  - Myo Armband                (USB dongle, MyoPy)
  - Generic analog ADC via serial (CSV stream: "ch0,ch1\\n")
"""

from __future__ import annotations
import logging
from typing import Optional, List

log = logging.getLogger(__name__)


class EMGInterface:
    """
    Abstraction layer over EMG hardware.

    When EMG_ENABLED is False, all methods are no-ops that return safe
    default values (None / 0.0 / []).  No hardware libraries are imported.
    """

    def __init__(self, port: str, device: str, sample_rate: int,
                 channels: tuple[int, ...]) -> None:
        self.port = port
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self._connected = False
        self._raw_buffer: List[List[float]] = []   # ring buffer of recent samples

        log.info("EMGInterface created (disabled stub). "
                 "Set EMG_ENABLED=True in config.py to activate.")

    # ──────────────────────────────────────────────
    #  Connection lifecycle
    # ──────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Open the connection to the EMG device.
        Returns True on success, False on failure.
        """
        log.info("EMG connect() called — hardware disabled, skipping.")
        return False

        # ── UNCOMMENT BELOW WHEN HARDWARE IS CONNECTED ──────────────────────
        #
        # if self.device == "bitalino":
        #     try:
        #         from bitalino import BITalino
        #         self._device_obj = BITalino(self.port)
        #         self._device_obj.start(self.sample_rate, list(self.channels))
        #         self._connected = True
        #         log.info(f"BITalino connected on {self.port} @ {self.sample_rate} Hz")
        #         return True
        #     except Exception as e:
        #         log.error(f"BITalino connection failed: {e}")
        #         return False
        #
        # elif self.device == "myo":
        #     try:
        #         import myo
        #         myo.init()
        #         self._hub = myo.Hub()
        #         self._listener = myo.DeviceListener()
        #         self._hub.run(1000, self._listener)
        #         self._connected = True
        #         log.info("Myo Armband connected.")
        #         return True
        #     except Exception as e:
        #         log.error(f"Myo connection failed: {e}")
        #         return False
        #
        # elif self.device == "custom":
        #     try:
        #         import serial
        #         self._serial = serial.Serial(self.port, baudrate=115200, timeout=1)
        #         self._connected = True
        #         log.info(f"Custom serial EMG on {self.port}")
        #         return True
        #     except Exception as e:
        #         log.error(f"Custom serial EMG failed: {e}")
        #         return False
        # ─────────────────────────────────────────────────────────────────────

    def disconnect(self) -> None:
        """Cleanly close the device connection."""
        log.info("EMG disconnect() called — hardware disabled, skipping.")
        self._connected = False

        # ── UNCOMMENT BELOW WHEN HARDWARE IS CONNECTED ──────────────────────
        #
        # if not self._connected:
        #     return
        # try:
        #     if self.device == "bitalino":
        #         self._device_obj.stop()
        #         self._device_obj.close()
        #     elif self.device == "myo":
        #         self._hub.stop()
        #         self._hub.shutdown()
        #     elif self.device == "custom":
        #         self._serial.close()
        #     self._connected = False
        #     log.info("EMG device disconnected.")
        # except Exception as e:
        #     log.warning(f"EMG disconnect error (ignored): {e}")
        # ─────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────
    #  Data acquisition
    # ──────────────────────────────────────────────

    def read_signal(self) -> Optional[List[float]]:
        """
        Read one frame of raw EMG samples (one value per channel).

        Returns:
            List[float] of length len(channels), or None if unavailable.
        """
        # Stub: no hardware
        return None

        # ── UNCOMMENT BELOW WHEN HARDWARE IS CONNECTED ──────────────────────
        #
        # if not self._connected:
        #     return None
        # try:
        #     if self.device == "bitalino":
        #         data = self._device_obj.read(1)   # 1 sample
        #         return [float(data[0, 5 + ch]) for ch in range(len(self.channels))]
        #
        #     elif self.device == "myo":
        #         emg = self._listener.emg
        #         return list(map(float, emg)) if emg else None
        #
        #     elif self.device == "custom":
        #         line = self._serial.readline().decode("utf-8", errors="ignore").strip()
        #         parts = line.split(",")
        #         return [float(p) for p in parts] if len(parts) == len(self.channels) else None
        #
        # except Exception as e:
        #     log.warning(f"EMG read error: {e}")
        #     return None
        # ─────────────────────────────────────────────────────────────────────

    def get_muscle_activation(self) -> float:
        """
        Compute a normalised muscle-activation level in [0.0, 1.0].

        Returns:
            float activation (0.0 = resting, 1.0 = maximum voluntary contraction)
            Returns 0.0 when hardware is disabled.
        """
        # Stub: no hardware
        return 0.0

        # ── UNCOMMENT BELOW WHEN HARDWARE IS CONNECTED ──────────────────────
        #
        # sample = self.read_signal()
        # if sample is None:
        #     return 0.0
        # # Simple RMS over all channels
        # import math
        # rms = math.sqrt(sum(v**2 for v in sample) / len(sample))
        # # Normalise against a known MVC reference (calibrate per patient)
        # MVC_REFERENCE = 512.0   # ADC units at maximum voluntary contraction
        # return min(rms / MVC_REFERENCE, 1.0)
        # ─────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────
    #  Future integration hooks (planned)
    # ──────────────────────────────────────────────

    # TODO: map EMG activation → player movement boost or unlock gate
    # TODO: add MVC calibration routine (3-second max-effort recording)
    # TODO: add fatigue index (median frequency shift of power spectrum)
    # TODO: integrate with metrics.py to log muscle activation alongside
    #       jerk / path-efficiency scores for clinical reporting

    @property
    def is_connected(self) -> bool:
        return self._connected
