"""Regression tests for the reviewer-requested comparison protocol."""
import json
import numpy as np
from pathlib import Path
from qr64_certified.attacks import get_attack_suite, apply_attack, AttackConfig
from qr64_certified.attacks.operations import true_crop_only
from qr64_certified.benchmark.types import BenchmarkProtocol

ROOT = Path(__file__).resolve().parents[1]


def test_revision_suites_are_exactly_20_by_3():
    for mode in ('sigma', 'variance'):
        attacks = get_attack_suite(f'revision_20x3_{mode}')
        assert len(attacks) == 60
        assert len({a.attack_id for a in attacks}) == 60
        assert {a.severity for a in attacks} == {'mild', 'moderate', 'strong'}
        assert len([a for a in attacks if a.group == 'rotation']) == 3
        assert [a.params['degrees'] for a in attacks if a.group == 'rotation'] == [5, 10, 15]
        assert not any(a.group in ('rotation_back', 'crop_resize') for a in attacks)
        assert len([a for a in attacks if a.group == 'crop_zero_pad']) == 3
        assert len([a for a in attacks if 'gaussian' in a.attack_id and a.category=='noise']) == 3
        cfg = json.loads((ROOT / f'configs/benchmark/revision_20x3_{mode}.json').read_text())
        protocol = BenchmarkProtocol.from_mapping(cfg)
        assert len(protocol.host_files) == 12
        assert 'lenna.bmp' not in protocol.host_files
        assert protocol.seeds == (2026,)


def test_gaussian_sigma_and_variance_have_correct_8bit_units():
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    sigma = apply_attack(img, AttackConfig('s5', 'gaussian_noise', {'sigma': 5, 'seed': 123}))
    var = apply_attack(img, AttackConfig('v3', 'gaussian_noise_var', {'variance': 0.003, 'seed': 123}))
    sd_sigma = np.std(sigma.astype(float) - img.astype(float))
    sd_var = np.std(var.astype(float) - img.astype(float))
    assert 4.8 < sd_sigma < 5.2
    assert 13.5 < sd_var < 14.5
    assert sd_var > sd_sigma * 2.5
    assert not np.array_equal(sigma, img)


def test_crop_is_zero_padded_without_resampling():
    img = np.arange(100 * 100 * 3, dtype=np.uint32).reshape(100, 100, 3).astype(np.uint8)
    out = apply_attack(img, AttackConfig('c90', 'crop_zero_pad', {'keep': .9}))
    assert out.shape == img.shape
    assert np.array_equal(out[5:95, 5:95], img[5:95, 5:95])
    assert np.all(out[:5] == 0) and np.all(out[-5:] == 0)
    exact = true_crop_only(img, keep=.9)
    assert exact.shape == (90, 90, 3)
    assert np.array_equal(exact, img[5:95, 5:95])


def test_lenna_is_excluded_from_all_host_enumeration_and_direct_loading():
    import pytest
    from qr64_certified.common.io import list_image_files, load_host_rgb
    hosts = list_image_files(ROOT / 'data/host')
    assert len(hosts) == 12
    assert {x.stem.casefold() for x in hosts}.isdisjoint({'lena','lenna'})
    for spelling in ('lenna.bmp', 'lena.bmp'):
        with pytest.raises(ValueError, match='Excluded host'):
            load_host_rgb(ROOT / 'data/host' / spelling)


def test_jpeg2000_revision_never_silently_falls_back_to_jpeg():
    for mode in ('sigma','variance'):
        rows = [a for a in get_attack_suite(f'revision_20x3_{mode}') if 'jpeg2000' in a.attack_id]
        assert len(rows) == 3
        assert all(row.params.get('strict') is True for row in rows)
