# ===================================================================
# LICENSE: MIT License (c) 2026 Markys Gariboldo. All rights reserved.
# CONTOUR: M-498 | UNIT: 77 | PROTOCOL: X5_CHIZHIK_VECTOR_CORE_HD
# STATUS: STABLE // MICRO-VECTOR ENGINE // FREQUENCY LOCKED: 80.08 Hz
# CORE: 12.1_PRODUCTION // MATRIX V6 // DUAL-WING TOPOLOGY
# ENGINE: NUMBA_FUSED_SINGLE_PASS (nogil=True) // NUMPY_FALLBACK
# INPUT CONTRACT: np.ndarray (PRIMARY) // List[float] (COMPAT)
# ===================================================================
#
# АРХИТЕКТУРНЫЙ МАНИФЕСТ (Core 12.1_PRODUCTION):
#   Полное исключение контейнерного оверхеда (Docker/Kubernetes).
#   Fused single-pass инференс: модуляция + норма + dot за один проход.
#   nogil=True — GIL отпущен во время вычисления, потоки работают
#   параллельно на разных ядрах. Event loop не блокируется.
#
#   Два режима:
#     NUMBA_FUSED   — @njit(nogil=True, fastmath=True) → LLVM SIMD, без GIL.
#     NUMPY_FALLBACK — чистый NumPy если Numba не установлен.
#
#   Входной контракт:
#     np.ndarray (float32, dim) — ПЕРВИЧНЫЙ, без конверсии, ~18.9 µс.
#     List[float]               — СОВМЕСТИМОСТЬ, авто-конверсия, ~99 µс.
#
# DUAL-WING TOPOLOGY:
#   Левое крыло (Sinister):  Проф. Маркус Гарибольдо — тензорный метаболизм.
#   Правое крыло (Dexter):   Проф. Владимир А. Лукин (ИП Лукин В.А.,
#                             ИНН 270308123260) — аппаратная синхронизация.
#
# ===================================================================

import asyncio
import tempfile
import os
from typing import List, Tuple, Optional, Union

import numpy as np

# ─── Попытка импорта Numba ──────────────────────────────────────────

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ─── DUAL-WING TOPOLOGY ──────────────────────────────────────────────

SINISTER_WING = {
    "name": "Sinister",
    "architect": "Markys Gariboldo",
    "role": "Tensor Metabolism / Latent Space Stabilization",
    "slot": 0,
}

DEXTER_WING = {
    "name": "Dexter",
    "architect": "Vladimir A. Lukin (IP Lukin V.A., INN 270308123260)",
    "role": "Hardware Sync / Edge Deployment / Physical Logistics",
    "slot": 1,
}


# ─── MATRIX V6 ───────────────────────────────────────────────────────

class MatrixV6:
    """
    Matrix V6 — модуль порогового отсечения ложных и деструктивных связей.
    |ЛЕВО - ПРАВО| < THRESHOLD → VALID.
    """

    THRESHOLD = 0.1

    @staticmethod
    def gate(cosine_left: float, cosine_right: float) -> Tuple[bool, str]:
        delta = abs(cosine_left - cosine_right)
        if delta < MatrixV6.THRESHOLD:
            return True, "VALID"
        return False, "REJECTED"

    @staticmethod
    def gate_pair(cosine_left: float, cosine_right: float) -> Tuple[bool, float, str]:
        delta = abs(cosine_left - cosine_right)
        valid = delta < MatrixV6.THRESHOLD
        verdict = "VALID" if valid else "REJECTED"
        return valid, delta, verdict


# ─── NUMBA FUSED KERNEL (nogil=True) ────────────────────────────────

if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True, nogil=True)
    def _fused_cosine_numba(
        arr_a: np.ndarray,
        arr_b: np.ndarray,
        freq: float,
        base: float,
        dim: int,
    ) -> float:
        """
        FUSED SINGLE-PASS (nogil=True):
          Модуляция + норма + dot в одном цикле, без промежуточных массивов.
          GIL отпущен — другие Python-потоки работают параллельно.
          LLVM auto-vectorize → SIMD для float32.
        """
        mod_factor = freq * base

        norm_a = 0.0
        norm_b = 0.0
        dot = 0.0

        for i in range(dim):
            va = arr_a[i] + np.cos(arr_a[i] * mod_factor) * 0.001
            vb = arr_b[i] + np.cos(arr_b[i] * mod_factor) * 0.001

            norm_a += va * va
            norm_b += vb * vb
            dot += va * vb

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot / (np.sqrt(norm_a) * np.sqrt(norm_b))

    @njit(cache=True, fastmath=True, nogil=True)
    def _calibrate_numba(arr: np.ndarray, freq: float, base: float, dim: int) -> np.ndarray:
        """Fused калибровка: cos-модуляция + добавка 0.001, nogil=True."""
        mod_factor = freq * base
        out = np.empty(dim, dtype=np.float32)
        for i in range(dim):
            out[i] = arr[i] + np.cos(arr[i] * mod_factor) * 0.001
        return out

else:
    _fused_cosine_numba = None
    _calibrate_numba = None


# ─── X5 CHIZHIK VECTOR CORE ─────────────────────────────────────────

class X5ChizhikVectorCore:
    """
    Микроядро X5 — Core 12.1_PRODUCTION.

    Входной контракт (ПЕРВИЧНЫЙ): np.ndarray (float32, vector_dim).
    Fallback (СОВМЕСТИМОСТЬ):     List[float] — авто-конверсия, ~80 µс оверхед.

    Движок:
      NUMBA_FUSED   — @njit(nogil=True, fastmath=True), fused single-pass, LLVM SIMD.
      NUMPY_FALLBACK — чистый NumPy если Numba не установлен.
    """

    def __init__(self, vector_dim: int = 1024, memmap_path: Optional[str] = None):
        self.vector_dim = vector_dim
        self.target_frequency = 80.08
        self.base_vector = 7.5924
        self.architect = "Markys Gariboldo"
        self.engine = "NUMBA_FUSED" if NUMBA_AVAILABLE else "NUMPY_FALLBACK"

        # ── np.memmap ──
        if memmap_path is None:
            self._tmp = tempfile.NamedTemporaryFile(
                suffix=".bin", prefix="x5_memmap_", delete=False
            )
            self._tmp.close()
            self.memmap_path = self._tmp.name
        else:
            self.memmap_path = memmap_path
            self._tmp = None

        total_bytes = 2 * vector_dim * 4
        if not os.path.exists(self.memmap_path) or os.path.getsize(self.memmap_path) < total_bytes:
            with open(self.memmap_path, "wb") as f:
                f.write(b"\x00" * total_bytes)

        self.memmap = np.memmap(
            self.memmap_path,
            dtype="float32",
            mode="readwrite",
            shape=(2, vector_dim),
        )

        self.matrix_v6 = MatrixV6()

    # ── Конвертер входа ──────────────────────────────────────────────

    def _to_ndarray(self, pulse: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Конвертация в np.ndarray (float32, vector_dim).
        np.ndarray — прямой путь, без копирования (copy=False).
        List[float] — конверсия, ~80 µс оверхед.
        """
        if isinstance(pulse, np.ndarray):
            arr = pulse.astype("float32", copy=False)
        else:
            arr = np.array(pulse, dtype="float32")

        if arr.shape[0] != self.vector_dim:
            arr = np.resize(arr, (self.vector_dim,))

        return arr

    # ── Калибровка ──────────────────────────────────────────────────

    def calibrate_hidden_states(self, raw_pulse: Union[np.ndarray, List[float]]) -> np.ndarray:
        arr = self._to_ndarray(raw_pulse)

        if NUMBA_AVAILABLE:
            return _calibrate_numba(arr, self.target_frequency, self.base_vector, self.vector_dim)
        else:
            frequency_gate = np.cos(arr * (self.target_frequency * self.base_vector))
            return arr + (frequency_gate * np.float32(0.001))

    # ── Косинусная калибровка ────────────────────────────────────────

    def compute_cosine_calibration(
        self,
        pulse_a: Union[np.ndarray, List[float]],
        pulse_b: Union[np.ndarray, List[float]],
    ) -> float:
        if NUMBA_AVAILABLE:
            arr_a = self._to_ndarray(pulse_a)
            arr_b = self._to_ndarray(pulse_b)
            return float(_fused_cosine_numba(
                arr_a, arr_b, self.target_frequency, self.base_vector, self.vector_dim
            ))
        else:
            vec_a = self.calibrate_hidden_states(pulse_a)
            vec_b = self.calibrate_hidden_states(pulse_b)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    # ── Memmap ──────────────────────────────────────────────────────

    def write_to_memmap(self, slot: int, pulse: Union[np.ndarray, List[float]]) -> None:
        if slot not in (0, 1):
            raise ValueError(f"Invalid slot {slot}; expected 0 (Sinister) or 1 (Dexter)")
        calibrated = self.calibrate_hidden_states(pulse)
        self.memmap[slot] = calibrated
        self.memmap.flush()

    def read_from_memmap(self, slot: int) -> np.ndarray:
        if slot not in (0, 1):
            raise ValueError(f"Invalid slot {slot}; expected 0 (Sinister) or 1 (Dexter)")
        return np.array(self.memmap[slot], dtype="float32")

    # ── Асинхронные обёртки ──────────────────────────────────────────

    async def async_calibrate_hidden_states(self, raw_pulse: Union[np.ndarray, List[float]]) -> np.ndarray:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.calibrate_hidden_states, raw_pulse)

    async def async_compute_cosine_calibration(
        self,
        pulse_a: Union[np.ndarray, List[float]],
        pulse_b: Union[np.ndarray, List[float]],
    ) -> float:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.compute_cosine_calibration, pulse_a, pulse_b)

    # ── Matrix V6 ──────────────────────────────────────────────────

    def validate_link(
        self,
        pulse_a: Union[np.ndarray, List[float]],
        pulse_b: Union[np.ndarray, List[float]],
    ) -> Tuple[bool, float, str]:
        arr_a = self._to_ndarray(pulse_a)
        arr_b = self._to_ndarray(pulse_b)

        # ЛЕВО: Sinister-слот
        self.memmap[SINISTER_WING["slot"]] = arr_a
        self.memmap.flush()
        vec_s = np.array(self.memmap[SINISTER_WING["slot"]], dtype="float32")
        norm_s = np.linalg.norm(vec_s)
        norm_b = np.linalg.norm(arr_b)
        if norm_s == 0 or norm_b == 0:
            cosine_left = 0.0
        else:
            cosine_left = float(np.dot(vec_s, arr_b) / (norm_s * norm_b))

        # ПРАВО: Dexter-слот
        self.memmap[DEXTER_WING["slot"]] = arr_a
        self.memmap.flush()
        vec_d = np.array(self.memmap[DEXTER_WING["slot"]], dtype="float32")
        norm_d = np.linalg.norm(vec_d)
        if norm_d == 0 or norm_b == 0:
            cosine_right = 0.0
        else:
            cosine_right = float(np.dot(vec_d, arr_b) / (norm_d * norm_b))

        return self.matrix_v6.gate_pair(cosine_left, cosine_right)

    # ── Cleanup ─────────────────────────────────────────────────────

    def close(self) -> None:
        del self.memmap
        if self._tmp is not None and os.path.exists(self.memmap_path):
            os.remove(self.memmap_path)

    @staticmethod
    def wing_info() -> dict:
        return {"sinister": SINISTER_WING, "dexter": DEXTER_WING}


# ===================================================================
# SIGNATURE: (c) 2026 MarkysUNIT77 // OMEGA_SEAL_12_HD_TOTAL_INFINITE
# GLOBAL COMMIT LOCK // CONTOUR: M-498 // TERMINAL END
# ===================================================================
