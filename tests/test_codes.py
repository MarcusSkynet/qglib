# tests/test_codes.py
import numpy as np
import qgledger as qg

def test_hamming_roundtrip():
    reg = qg.pack_register(1, +1, -1, +1)  # [1,0,1,0]
    code = qg.hamming74_encode(reg)
    # flip bit 3
    code_f = code.copy(); code_f[3] ^= 1
    data_out, code_corr, corrected, multi = qg.hamming74_decode(code_f)
    assert corrected and not multi
    assert np.all(data_out == reg)
