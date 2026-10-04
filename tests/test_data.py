from jeopardybench.data import eligible_subjects


def test_filters_images(questions):
    assert "fximg" not in questions.index
    assert len(questions) == 13 * 7 + 3


def test_eligible_subjects(questions):
    subjects = eligible_subjects(questions)
    assert len(subjects) == 13
    assert "Tiny Subject" not in subjects
