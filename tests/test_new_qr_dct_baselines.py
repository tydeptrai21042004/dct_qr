"""Publication guards for the two newly integrated baseline adapters."""
import json
from pathlib import Path

import numpy as np
import pytest

from qr64_certified.baselines import get_baseline_spec, embed_baseline, extract_baseline
from qr64_certified.baselines.implementations.transform.nha_qr2022 import _embedding_r11_target
from qr64_certified.baselines.implementations.dct.ariatmanto_dct2019 import (
    FIVE, _paper_rx, _paper_ry
)
from qr64_certified.common.io import load_host_rgb, load_watermark_binary, list_image_files
from qr64_certified.common.metrics import nc, ber

ROOT = Path(__file__).resolve().parents[1]
QR = 'qr_nha2022_blind'
DCT = 'dct_adaptive_ariatmanto2019_keyassisted'


def test_nha_qr_modulo_centers_are_paper_ones():
    r = np.array([50., 52., 55., 57., 59., 63., 77.], dtype=np.float64)
    for bit, center in ((0,2.5),(1,7.5)):
        new, _ = _embedding_r11_target(r, np.full_like(r,bit),10.)
        np.testing.assert_allclose(np.mod(new,10),center,atol=1e-12)
        assert np.all(new>0)


def test_published_dct_five_positions_and_indexed_statistics():
    assert FIVE.tolist() == [[4,2],[3,3],[2,4],[1,5],[2,5]]
    x = np.array([[1., -2., 3., 0., -4.]])
    np.testing.assert_allclose(_paper_rx(x),abs(x)*(x.sum()/64.0))
    np.testing.assert_allclose(_paper_ry(x),x*(x.sum()/64.0))


def test_registry_metadata_declares_adaptations_and_side_information():
    qr = get_baseline_spec(QR)
    dct = get_baseline_spec(DCT)
    assert qr.blindness_tier == 'blind'
    assert dct.blindness_tier == 'key_assisted_blind'
    assert qr.common_4096_payload and dct.common_4096_payload
    assert not qr.disclosure['paper_reproduction_eligible']
    assert not dct.disclosure['paper_reproduction_eligible']
    assert not dct.requires_original_host
    assert qr.primary_blind_eligible and not dct.primary_blind_eligible


def test_clean_embedding_on_actual_non_lenna_host_and_two_watermarks():
    host = load_host_rgb(ROOT / 'data/host/tiffany.bmp')
    for name in ('wm.png', 'Cc.logo.circle.png'):
        wm = load_watermark_binary(ROOT/'data/watermark'/name)
        for mid in (QR, DCT):
            marked, key = embed_baseline(mid,host,wm)
            extracted = extract_baseline(marked,key)
            assert marked.shape == host.shape and marked.dtype == np.uint8
            assert extracted.shape == wm.shape and extracted.dtype == np.uint8
            assert nc(wm,extracted) > 0.99
            assert ber(wm,extracted) < 0.01
            if mid == DCT:
                assert key.native_key.selected_block_ids.size == 820
                assert key.native_key.mode == 'adapt'
            else:
                assert key.native_key.block_size == 4
                assert key.native_key.repeats == (2,2)


def test_literal_dct_mode_is_separate_and_not_silently_fixed():
    host = load_host_rgb(ROOT / 'data/host/pepper.bmp')
    wm = load_watermark_binary(ROOT / 'data/watermark/wm.png')
    marked,key = embed_baseline(DCT,host,wm,method_params={'mode':'literal'})
    extracted = extract_baseline(marked,key)
    assert extracted.shape == wm.shape
    assert key.native_key.mode == 'literal'
    # The printed conditional rules may not recover the entire payload.
    assert 0.0 <= nc(wm,extracted) <= 1.0


def test_new_protocol_uses_12_non_lenna_hosts_and_60_attacks():
    d = json.loads((ROOT/'configs/benchmark/two_new_baselines_20x3.json').read_text())
    assert set(d['methods']) == {QR,DCT}
    assert len(d['host_files']) == 12
    assert all('lena' not in s.lower() and 'lenna' not in s.lower() for s in d['host_files'])
    assert len(list_image_files(ROOT/'data/host')) == 12
    assert d['attack_suite'] == 'revision_20x3_variance'
