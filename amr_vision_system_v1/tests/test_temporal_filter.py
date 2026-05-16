from tracking.temporal_filter import TemporalFilter


def test_state_persistence_with_alive_ids():
    tf = TemporalFilter(alpha=0.5)
    # initial detection with id 1
    det = {"id": 1, "bbox_xyxy": [0, 0, 10, 10], "confidence": 0.9}
    tf.update([det], alive_ids={1})
    # next frame has no detections but tracker reports id 1 still alive
    out = tf.update([], alive_ids={1})
    # no detections returned but state should still contain the id
    assert 1 in tf.states
