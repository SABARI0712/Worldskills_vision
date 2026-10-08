from tracking.centroid_tracker import CentroidTracker


def test_velocity_computation():
    tracker = CentroidTracker()
    det1 = {"bbox_xyxy": [0, 0, 10, 10], "label": "obj"}
    out1 = tracker.update([det1], timestamp_ms=1000)
    assert len(out1) == 1
    obj_id = out1[0].get("id")
    assert obj_id is not None

    # move 10 pixels in x over 0.1s -> vx ~100 px/s
    det2 = {"bbox_xyxy": [10, 0, 20, 10], "label": "obj"}
    out2 = tracker.update([det2], timestamp_ms=1100)
    assert len(out2) == 1
    v = out2[0].get("meta", {}).get("velocity")
    assert v is not None
    vx, vy = v
    assert abs(vx - 100.0) < 20.0
