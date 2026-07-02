from snookerhelp.recognition.color import BallColorClassifier


def classify(h: int, s: int, v: int, lab: tuple[int, int, int]) -> str:
    return BallColorClassifier._classify_values(h, s, v, *lab)[0]


def test_supplied_camera_color_rules() -> None:
    assert classify(5, 210, 240, (140, 194, 180)) == "red"
    assert classify(28, 23, 243, (242, 125, 138)) == "white"
    assert classify(3, 20, 243, (231, 134, 131)) == "pink"
    assert classify(4, 125, 245, (179, 164, 154)) == "pink"
    assert classify(108, 102, 58, (43, 128, 118)) == "black"
    assert classify(95, 163, 236, (200, 102, 105)) == "blue"
    assert classify(87, 255, 193, (179, 84, 127)) == "green"
    assert classify(27, 220, 255, (230, 120, 200)) == "yellow"
    assert classify(14, 160, 227, (177, 149, 174)) == "brown"
