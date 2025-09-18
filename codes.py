# qgledger/codes.py
"""
qgledger.codes
==============
Classical error-correcting codes for **ledger writes** in the IR Quantum Gravity
(ledger/NPR) framework.

Context
-------
A minimal reversible write records four classical bits on the ledger:
    (b_r, X, Y, Z),
where b_r is the branch bit and (X,Y,Z) are sign bits extracted from principal
shear/twist data on the 2D screen. These are **classical** metadata; quantum
transport across the ledger is forbidden between writes by NPR.

This module provides:
- Bit-packing/unpacking helpers linking geometric signs to bits.
- A standard Hamming(7,4) linear block code (single-bit error correction).
- Simple bit-flip channel and repetition helpers.
- Area budgeting: minimal area per write is ΔA_min = 4 A_bit (from qgledger.constants).

Why classical?
--------------
The ledger’s minimal reversible record is **classical** (4 bits). Robust storage
against local defects benefits from a classical ECC. If/when a quantum network
is layered over larger regions (e.g., to protect entanglement across many writes),
that would live in a separate quantum-code module; here we stick to the minimal,
local classical record.

Units
-----
This module manipulates bits only. When area budgeting is relevant we import
`A_bit` in SI units [m^2] from :mod:`qgledger.constants`.
"""

from __future__ import annotations

from typing import Tuple, List
import numpy as np

from .constants import A_bit

# -----------------------------------------------------------------------------
# Register packing: (b_r, X, Y, Z) ↔ 4-bit vector
# -----------------------------------------------------------------------------

def sign_to_bit(s: int) -> int:
    """
    Map a geometric sign ±1 to a bit: +1 → 0, −1 → 1.

    Parameters
    ----------
    s : int
        Either +1 or −1.

    Returns
    -------
    int
        0 for +1, 1 for −1.
    """
    if s not in (+1, -1):
        raise ValueError("sign must be +1 or -1")
    return 0 if s == +1 else 1


def bit_to_sign(b: int) -> int:
    """
    Map a bit to a geometric sign: 0 → +1, 1 → −1.
    """
    if b not in (0, 1):
        raise ValueError("bit must be 0 or 1")
    return +1 if b == 0 else -1


def pack_register(b_r: int, sX: int, sY: int, sZ: int) -> np.ndarray:
    """
    Pack (b_r, X, Y, Z) where X,Y,Z are signs ±1 into a 4-bit vector [b_r, x, y, z].

    Parameters
    ----------
    b_r : int
        Branch bit (0/1).
    sX, sY, sZ : int
        Signs ±1 derived from geometry.

    Returns
    -------
    ndarray
        uint8 vector of length 4: [b_r, x, y, z] with x=sign_to_bit(sX), etc.
    """
    if b_r not in (0, 1):
        raise ValueError("b_r must be 0 or 1")
    return np.array([b_r, sign_to_bit(sX), sign_to_bit(sY), sign_to_bit(sZ)], dtype=np.uint8)


def unpack_register(bits4: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Unpack a length-4 bit vector [b_r, x, y, z] into (b_r, sX, sY, sZ) with signs ±1.
    """
    b = np.asarray(bits4, dtype=np.uint8)
    if b.shape != (4,):
        raise ValueError("bits4 must have shape (4,)")
    b_r, x, y, z = [int(v) for v in b]
    return b_r, bit_to_sign(x), bit_to_sign(y), bit_to_sign(z)


# -----------------------------------------------------------------------------
# Hamming(7,4) code
# -----------------------------------------------------------------------------

def hamming74_matrices() -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (G, H) for the Hamming(7,4) linear block code over GF(2).

    Conventions
    -----------
    - Codeword c = u G  (u is 1×4 row vector of data bits, c is 1×7).
    - Parity-check satisfies H c^T = 0 (mod 2).
    - We choose the common systematic form with codeword ordering [d1 d2 d3 p1 d4 p2 p3].

    Returns
    -------
    (G, H) : (ndarray, ndarray)
        G: 4×7 generator, H: 3×7 parity-check, dtype uint8.
    """
    # Systematic generator (one standard choice):
    # Rows are data bits d1..d4; columns order: [d1 d2 d3 p1 d4 p2 p3]
    G = np.array([
        [1, 0, 0, 1, 0, 1, 1],  # d1
        [0, 1, 0, 1, 0, 0, 1],  # d2
        [0, 0, 1, 1, 0, 1, 0],  # d3
        [0, 0, 0, 0, 1, 1, 1],  # d4
    ], dtype=np.uint8)
    # Parity-check H such that H G^T = 0
    H = np.array([
        [1, 1, 1, 1, 0, 0, 0],  # checks {d1,d2,d3,p1}
        [1, 0, 1, 0, 1, 1, 0],  # checks {d1,d3,d4,p2}
        [1, 1, 0, 0, 1, 0, 1],  # checks {d1,d2,d4,p3}
    ], dtype=np.uint8)
    return G, H


def hamming74_encode(data4: np.ndarray) -> np.ndarray:
    """
    Encode 4 data bits (uint8) into a 7-bit Hamming(7,4) codeword.

    Parameters
    ----------
    data4 : ndarray
        Length-4 vector of bits {0,1}.

    Returns
    -------
    ndarray
        Length-7 codeword (uint8), ordering [d1 d2 d3 p1 d4 p2 p3].
    """
    u = np.asarray(data4, dtype=np.uint8).reshape(1, -1)
    if u.shape != (1, 4):
        raise ValueError("data4 must have length 4")
    G, _ = hamming74_matrices()
    c = (u @ G) % 2
    return c.astype(np.uint8).ravel()


def hamming74_syndrome(code7: np.ndarray) -> np.ndarray:
    """
    Compute the 3-bit syndrome s = H c^T (mod 2).

    Parameters
    ----------
    code7 : ndarray
        Length-7 codeword (uint8).

    Returns
    -------
    ndarray
        Length-3 syndrome bits (uint8). s=0 indicates a valid codeword.
    """
    c = np.asarray(code7, dtype=np.uint8).reshape(7, 1)
    _, H = hamming74_matrices()
    s = (H @ c) % 2
    return s.astype(np.uint8).ravel()


def _syndrome_to_error_index(s: np.ndarray) -> int | None:
    """
    Map a 3-bit syndrome to a single-bit error index 0..6 (or None if s==0).

    Returns
    -------
    int or None
        Index of the flipped bit to correct, or None if no single-bit error is indicated.
    """
    # Build a lookup from columns of H to indices
    _, H = hamming74_matrices()
    cols = [tuple(H[:, i] % 2) for i in range(7)]
    st = tuple(int(x) for x in (s % 2).ravel())
    if st == (0, 0, 0):
        return None
    try:
        return cols.index(st)
    except ValueError:
        # Not a single-bit pattern (e.g., multiple errors)
        return None


def hamming74_decode(code7: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool]:
    """
    Decode a 7-bit word with single-bit error correction.

    Parameters
    ----------
    code7 : ndarray
        Length-7 codeword (uint8).

    Returns
    -------
    (data4, corrected, corrected_flag, detected_multi) :
        data4 : length-4 data bits (uint8) after correction (if any).
        corrected : corrected 7-bit codeword (uint8).
        corrected_flag : True if a single-bit error was detected and corrected.
        detected_multi : True if a non-code syndrome remained after correction
                         (suggests multiple-bit error).

    Notes
    -----
    We extract data in the order [d1, d2, d3, d4] from the (possibly corrected) code.
    """
    c = np.asarray(code7, dtype=np.uint8).copy().ravel()
    if c.shape != (7,):
        raise ValueError("code7 must have length 7")
    s = hamming74_syndrome(c)
    idx = _syndrome_to_error_index(s)
    corrected_flag = False
    if idx is not None:
        c[idx] ^= 1
        corrected_flag = True
    # Recompute syndrome to see if errors persist
    s2 = hamming74_syndrome(c)
    detected_multi = not np.all(s2 == 0)
    # Extract data [d1,d2,d3,d4] from positions [0,1,2,4]
    data4 = c[[0, 1, 2, 4]].astype(np.uint8)
    return data4, c.astype(np.uint8), corrected_flag, bool(detected_multi)


# -----------------------------------------------------------------------------
# Channel and repetition helpers
# -----------------------------------------------------------------------------

def bitflip_channel(x: np.ndarray, p: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Apply an i.i.d. bit-flip channel with flip probability p to a bit array.

    Parameters
    ----------
    x : ndarray
        Bit array (uint8). Arbitrary shape.
    p : float
        Flip probability in [0,1].
    rng : np.random.Generator, optional
        Random generator; if None, uses default.

    Returns
    -------
    ndarray
        Flipped bits (uint8), same shape as input.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1]")
    if rng is None:
        rng = np.random.default_rng()
    flips = rng.random(size=x.shape) < p
    y = (x.astype(np.uint8) ^ flips.astype(np.uint8))
    return y.astype(np.uint8)


def repetition_encode(bit: int, n: int) -> np.ndarray:
    """
    Encode a single bit with an n-fold repetition code.

    Parameters
    ----------
    bit : int
        0 or 1.
    n : int
        Repetition length ≥ 1.

    Returns
    -------
    ndarray
        Length-n vector filled with `bit` (uint8).
    """
    if bit not in (0, 1):
        raise ValueError("bit must be 0 or 1")
    if n < 1:
        raise ValueError("n must be ≥ 1")
    return np.full(n, bit, dtype=np.uint8)


def repetition_decode_majority(block: np.ndarray) -> int:
    """
    Majority-vote decoder for a repetition block.

    Parameters
    ----------
    block : ndarray
        Vector of bits (uint8).

    Returns
    -------
    int
        Decoded bit (0 or 1).
    """
    b = np.asarray(block, dtype=np.uint8).ravel()
    ones = int(b.sum())
    zeros = b.size - ones
    return 1 if ones > zeros else 0


# -----------------------------------------------------------------------------
# Area budgeting
# -----------------------------------------------------------------------------

def area_for_n_registers(n: int) -> float:
    """
    Minimal ledger area to store n independent 4-bit registers:

        A_min(n) = n * (4 A_bit).

    Parameters
    ----------
    n : int
        Number of registers.

    Returns
    -------
    float
        Area in m^2.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    return float(n) * 4.0 * A_bit


__all__ = [
    "sign_to_bit", "bit_to_sign", "pack_register", "unpack_register",
    "hamming74_matrices", "hamming74_encode", "hamming74_syndrome", "hamming74_decode",
    "bitflip_channel", "repetition_encode", "repetition_decode_majority",
    "area_for_n_registers",
]
