from pathlib import Path
import json
import numpy as np
from qr64_certified.attacks import apply_attack, get_attack_suite
from qr64_certified.attacks.types import AttackConfig


def test_reviewer_suites_are_complete_and_nonduplicated():
    attacks=get_attack_suite('reviewer_strength')
    assert len(attacks)>=35
    ids=[a.attack_id for a in attacks]
    assert len(ids)==len(set(ids))
    assert any(a.group=='carrier_bandstop' for a in attacks)
    assert any(a.group=='subpixel_translation' for a in attacks)
    assert any(a.group=='gaussian_noise_var' and a.params['variance']==0.05 for a in attacks)
    assert any(a.group=='jpeg' and a.params['quality']==30 for a in attacks)


def test_fractional_translation_deterministic():
    im=np.arange(32*32*3,dtype=np.uint8).reshape(32,32,3)
    a=AttackConfig('x','subpixel_translation',{'shift_x':0.5,'shift_y':0.5})
    out=apply_attack(im,a)
    assert out.shape==im.shape and out.dtype==np.uint8
    assert np.array_equal(out,apply_attack(im,a))
    assert not np.array_equal(out,im)


def test_public_bandstop_is_not_noop():
    im=np.random.default_rng(3).integers(0,256,(32,32,3),dtype=np.uint8)
    out=apply_attack(im,AttackConfig('zero','carrier_bandstop',{'attenuation':0.0}))
    assert out.shape==im.shape and out.dtype==np.uint8
    assert not np.array_equal(im,out)


def test_skip_failed_trials_for_resume(tmp_path):
    import importlib.util
    path=Path(__file__).parents[1]/'scripts'/'run_unified_benchmark.py'
    spec=importlib.util.spec_from_file_location('unified_bench_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    trial=tmp_path/'trial.json'
    payload={'protocol_id':'x','experiment_fingerprint':'hash','status':'embed_or_clean_error','rows':[]}
    trial.write_text(json.dumps(payload))
    assert not module._valid_trial(trial,protocol_id='x',attack_ids=('jpeg',),fingerprint='hash')
    payload.update(status='ok',rows=[{'attack_id':'jpeg','status':'error'}]);trial.write_text(json.dumps(payload))
    assert not module._valid_trial(trial,protocol_id='x',attack_ids=('jpeg',),fingerprint='hash')
    payload['rows'][0]['status']='ok';trial.write_text(json.dumps(payload))
    assert module._valid_trial(trial,protocol_id='x',attack_ids=('jpeg',),fingerprint='hash')
    assert not module._valid_trial(trial,protocol_id='x',attack_ids=('jpeg',),fingerprint='different')
