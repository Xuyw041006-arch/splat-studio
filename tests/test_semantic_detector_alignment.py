import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.semantics import _aligned_detector_labels


def test_box_query_mapping_preserves_rejections_without_shifting_masks(monkeypatch):
    logits=torch.tensor([[[-3.,-3.],[2.,0.],[.3,.2],[0.,3.]]])
    scores=logits[0].sigmoid().amax(-1)[1:]
    out=SimpleNamespace(logits=logits)
    batch=SimpleNamespace(input_ids=torch.tensor([[1,2]]),attention_mask=torch.ones(1,2))
    detected={'boxes':torch.zeros(3,4),'scores':scores,'text_labels':['coffee mug coffee','fragment','mug']}
    def align(*a,**kw):
        assert kw['min_score']==.20 and kw['min_margin']==.025
        return {'query_labels':[None,'coffee',None,'coffee mug'],'best_scores':[0,.8,.2,.9],'margins':[0,.5,.01,.6]}
    monkeypatch.setattr('backend.semantic_labeling.align_grounding_dino_queries',align)
    labels,meta,diag=_aligned_detector_labels(None,batch,out,detected,'coffee. coffee mug.',['coffee','coffee mug'],.5)
    assert labels==['coffee',None,'coffee mug']
    assert [m['detector_query_index'] for m in meta]==[1,2,3]
    assert meta[2]['raw_detector_label']=='mug'
    assert diag['rejected_ambiguous']==1 and diag['missing_candidate_labels']==[]
    detected['scores']=scores.flip(0)
    with pytest.raises(ValueError,match='query order'):
        _aligned_detector_labels(None,batch,out,detected,'coffee.',['coffee'],.5)
