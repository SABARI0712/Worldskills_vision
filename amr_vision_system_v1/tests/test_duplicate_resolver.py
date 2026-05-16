import pytest
from fusion.duplicate_resolver import DuplicateResolver

def test_aruco_priority():
    resolver = DuplicateResolver()
    dets = [
        {"label":"marker","confidence":0.9,"bbox_xyxy":[10,10,50,50],"source":"yolo"},
        {"label":"marker","confidence":0.8,"bbox_xyxy":[12,12,48,48],"source":"aruco"},
    ]
    out = resolver.resolve(dets)
    assert len(out) == 1
    assert out[0]["source"] == "aruco"
