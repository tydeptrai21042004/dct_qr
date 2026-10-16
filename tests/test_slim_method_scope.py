from qr64_certified.benchmark.registry import PROPOSAL_IDS, resolve_methods
from qr64_certified.baselines import ALL_BASELINE_IDS
from qr64_certified import DCT_QR_THEORY, embed_proposal
import numpy as np
import pytest

def test_one_proposed_method_and_sixteen_baselines():
    assert PROPOSAL_IDS == (DCT_QR_THEORY,)
    assert len(ALL_BASELINE_IDS) == 16
    methods = resolve_methods('all')
    assert len(methods) == 17
    assert [m.method_id for m in methods if m.method_kind == 'proposal'] == ['dct_qr_theory']

def test_original_dct_qr_is_unavailable():
    with pytest.raises((KeyError,ValueError)):
        resolve_methods('dct_qr')
    with pytest.raises((KeyError,ValueError)):
        embed_proposal('dct_qr', np.zeros((512,512,3),dtype=np.uint8),np.zeros((64,64),dtype=np.uint8))
