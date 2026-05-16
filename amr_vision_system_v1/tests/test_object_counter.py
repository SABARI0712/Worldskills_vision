from perception.counter import ObjectCounter


def test_per_frame_counts_snapshot():
    counter = ObjectCounter()
    dets_frame1 = [{"label": "servo_power_block", "id": 1}]
    dets_frame2 = [{"label": "servo_power_block", "id": 1}, {"label": "servo_power_block", "id": 2}]

    summary1 = counter.update(dets_frame1)
    assert summary1["total_objects"] == 1
    assert summary1["unique_tracked"] == 1

    summary2 = counter.update(dets_frame2)
    assert summary2["total_objects"] == 2
    assert summary2["unique_tracked"] == 2

    # Frame 1 counts should not carry over
    assert summary1["total_objects"] == 1
    assert summary1["unique_tracked"] == 1
