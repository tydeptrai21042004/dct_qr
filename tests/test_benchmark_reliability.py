"""Integrity checks for long-running and checkpointed experimental evidence."""
from __future__ import annotations
import importlib.util, json, zipfile
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

def load_script(name):
    spec=importlib.util.spec_from_file_location(f'test_script_{name}',ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_strict_trial_rejects_duplicate_nonfinite_and_incomplete_rows():
    m=load_script('run_unified_benchmark')
    t={'protocol_id':'p','experiment_fingerprint':'f','method_id':'dct_qr_theory','host_id':'a',
       'watermark_id':'wm','seed':2026,'status':'ok',
       'embedding_metrics':{'psnr_db':53.0,'ssim':0.998},
       'clean_watermark_metrics':{'nc':1.0,'ber':0.0},
       'rows':[{'attack_id':x,'method_id':'dct_qr_theory','seed':2026,
                'status':'ok','attacked_nc':0.9,'attacked_ber':0.1,
                'attacked_bit_accuracy':0.9,'embedding_ssim':0.998,'embedding_psnr_db':53.0}
               for x in ('jpeg_q90','noise') ]}
    kwargs={'protocol_id':'p','fingerprint':'f','attack_ids':('jpeg_q90','noise'),
            'identity':('dct_qr_theory','a','wm',2026)}
    assert m._trial_issues(t,**kwargs)==[]
    t['rows'][1]['attack_id']='jpeg_q90'
    assert 'missing_duplicate_or_misordered_attacks' in m._trial_issues(t,**kwargs)
    t['rows'][1]['attack_id']='noise';t['rows'][1]['attacked_nc']=float('nan')
    assert 'nonfinite_or_missing_attack_metrics' in m._trial_issues(t,**kwargs)
    t['rows'].pop()
    assert 'wrong_attack_count' in m._trial_issues(t,**kwargs)


def test_validated_archive_restore_and_traversal_rejection(tmp_path):
    m=load_script('run_kaggle_experiments')
    out=tmp_path/'results';(out/'a').mkdir(parents=True)
    (out/'a'/'result.json').write_text('{"ok":true}')
    archive=tmp_path/'ckpt.zip';m._snapshot(out,archive)
    target=tmp_path/'restored';m.restore(archive,target)
    assert (target/'a'/'result.json').read_text()=='{"ok":true}'
    m.restore(archive,target)  # no-op allowed if bytes identical
    (target/'a'/'result.json').write_text('different')
    with pytest.raises(ValueError,match='conflicts'):m.restore(archive,target)
    malicious=tmp_path/'evil.zip'
    with zipfile.ZipFile(malicious,'w') as z:z.writestr('dct_qr_results/../../bad','oops')
    with pytest.raises(ValueError,match='Unsafe'):m.restore(malicious,tmp_path/'other')


def test_audit_detects_missing_source_and_failed_trial(tmp_path):
    m=load_script('audit_benchmark')
    base=tmp_path/'b';(base/'trials').mkdir(parents=True)
    manifest={'experiment_fingerprint':'f','protocol':{'protocol_id':'p','seeds':[2026]},
              'methods':[{'method_id':'a'}], 'hosts':[{'path':'data/host/Girl.bmp','sha256':'x'}],
              'watermark_files':[{'path':'data/watermark/wm.png','sha256':'x'}],
              'attacks':[{'attack_id':'jpeg'}], 'source_sha256':{}}
    (base/'manifest.json').write_text(json.dumps(manifest))
    result=m.audit(base,verify_sources=False)
    assert result['expected_trials']==1 and result['missing_trial_count']==1 and not result['all_complete']
    (base/'trials'/'x.json').write_text(json.dumps({'protocol_id':'p','experiment_fingerprint':'f',
      'status':'error','method_id':'a','host_id':'Girl.bmp','watermark_id':'wm.png','seed':2026,'rows':[]}))
    result=m.audit(base,verify_sources=False)
    assert result['missing_trial_count']==0 and result['failed_or_invalid_trial_count']==1
