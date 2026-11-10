"""Guard the supervisor-approved final method roster and attack/analysis identity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from qr64_certified.attacks import get_attack_suite
from qr64_certified.benchmark.registry import resolve_methods

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from export_revision_fig_tables import write_report, prepare
from export_revision_extended_experiments import export, paired_bootstrap
from run_revision_ablation_grid import _safe_result, VARIANTS

EXPECTED = (
    'dwt_entropy_kumar2021_nonblind',
    'dwt_svd_roy2018_semiblind',
    'dct_ca_qim_mao2024_semiblind',
    'dwt_wht_svd_kumar2024_semiblind',
    'dwt_hessenberg_svd_paper2025_semiblind',
    'dwt_qr_fa_guo2017_keyassisted',
    'iwt_svd_qim_zhu2021_blind',
    'hessenberg_nha2023_blind',
    'qr_nha2022_blind',
    'dct_adaptive_ariatmanto2019_keyassisted',
    'dct_qr_theory',
)


def test_table10_profiles_exactly_match_approved_roster():
    for mode in ('sigma','variance'):
        p=json.loads((ROOT/f'configs/benchmark/revision_20x3_{mode}.json').read_text())
        assert tuple(p['methods']) == EXPECTED
        specs=resolve_methods(p['methods'])
        assert tuple(s.method_id for s in specs)==EXPECTED
        assert all(s.common_4096_payload for s in specs)
        assert len(p['host_files'])==12
        assert not any('lenna' in x.lower() or 'lena' in x.lower() for x in p['host_files'])
        assert len(p['watermarks']) == 2
        assert len(get_attack_suite(p['attack_suite'])) == 60
        assert p['seeds'] == [2026]
        assert p['protocol_id'].endswith('v2_table10')
        tiffany=json.loads((ROOT/f'configs/benchmark/tiffany_fig4_{mode}.json').read_text())
        assert tiffany['attack_suite'] == p['attack_suite']
        assert tiffany['host_files'] == ['tiffany.bmp']
        assert tiffany['methods'] == ['dct_qr_theory']
        assert tiffany['watermarks'] == ['data/watermark/Cc.logo.circle.png']


def _dummy_complete_run(tmp_path):
    from pathlib import Path
    run=tmp_path/'run'
    (run/'trials').mkdir(parents=True)
    ids=['qr_nha2022_blind','dct_qr_theory']
    modes=['airplane.bmp','tiffany.bmp']
    wms=['wm.png','Cc.logo.circle.png']
    attacks=get_attack_suite('revision_20x3_variance')
    manifest={'protocol':{'protocol_id':'synthetic_test','attack_suite':'revision_20x3_variance','seeds':[2026]},
              'experiment_fingerprint':'test_only',
              'hosts':[{'path':'data/host/'+n} for n in modes],
              'watermark_files':[{'path':'data/watermark/'+n} for n in wms],
              'methods':[{'method_id':m,'display_name':m,'blindness_tier':'blind',
                          'fidelity_tier':'test'} for m in ids]}
    (run/'manifest.json').write_text(json.dumps(manifest))
    for m in ids:
        for h in modes:
            for wm in wms:
                is_pro=m=='dct_qr_theory'
                nc=0.9 if is_pro else 0.75
                ber=0.05 if is_pro else 0.2
                records=[{'attack_id':a.attack_id,'attack_severity':a.severity,
                          'status':'ok','attacked_nc':nc,'attacked_ber':ber,
                          'embedding_psnr_db':52.0,'embedding_ssim':0.997}
                         for a in attacks]
                t={'protocol_id':'synthetic_test','experiment_fingerprint':'test_only',
                   'status':'ok','method_id':m,'host_id':h,'watermark_id':wm,'seed':2026,
                   'embedding_metrics':{'psnr_db':52.0,'ssim':0.997},
                   'clean_watermark_metrics':{'nc':1.,'ber':0.},'rows':records}
                (run/'trials'/f'{m}_{h}_{wm}.json').write_text(json.dumps(t))
    return run


def test_cross_method_exports_use_same_complete_results(tmp_path):
    run=_dummy_complete_run(tmp_path)
    out=tmp_path/'out'
    write_report(run,out)
    assert (out/'table10_equal_payload_comparison.csv').exists()
    import csv
    with (out/'table10_equal_payload_comparison.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    assert [x['method_id'] for x in rows]==['qr_nha2022_blind','dct_qr_theory']
    assert all(int(r['case_count'])==4*60 for r in rows)
    assert (out/'fig5_moderate_comparison.pdf').exists()
    ext=export(run,tmp_path/'extra',n_bootstrap=120)
    assert ext['host_count']==2 and len(ext['watermarks'])==2
    with (tmp_path/'extra'/'paired_comparisons_vs_proposed.csv').open(newline='') as f:
        paired=list(csv.DictReader(f))
    assert len(paired)==1 and paired[0]['baseline']=='qr_nha2022_blind'
    assert np.isclose(float(paired[0]['mean_nc_difference_proposed_minus_baseline']),0.15)
    assert (tmp_path/'extra'/'gaussian_sensitivity.csv').exists()


def test_incomplete_trials_are_rejected_not_filled_with_zeros(tmp_path):
    run=_dummy_complete_run(tmp_path)
    first=next((run/'trials').glob('*.json'))
    first.unlink()
    import pytest
    with pytest.raises(ValueError,match='missing trials'):
        prepare(run)


def test_full_ablation_validator_requires_every_attack(tmp_path):
    p=tmp_path/'ablation.json'
    ids={a.attack_id for a in get_attack_suite('revision_20x3_variance')}
    example={'variant':'full', 'attacks':[{'attack':a,'nc':.99,'ber':.01} for a in sorted(ids)]}
    p.write_text(json.dumps([example]))
    assert not _safe_result(p,expected=ids)  # Missing six variants cannot pass
    assert len(VARIANTS)==7
    a,b=paired_bootstrap(np.array([0.1,0.2,0.3]),n_bootstrap=120,seed=2026)
    assert 0.09 <= a <= b <= 0.31
