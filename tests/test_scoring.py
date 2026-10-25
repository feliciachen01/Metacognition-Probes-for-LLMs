from kwik.scoring import clean_prediction, contains_match, exact_match, normalize, same_answer, score


def test_normalize():
    assert normalize("The Beatles!") == "beatles"
    assert normalize("  Paris,  France ") == "paris france"


def test_clean_prediction():
    assert clean_prediction("Answer: Paris.\nExtra line") == "Paris"
    assert clean_prediction("\n\n  Mount Everest  ") == "Mount Everest"


def test_exact_and_contains():
    golds = ["George Washington", "Washington"]
    assert exact_match("george washington", golds)
    assert contains_match("It was George Washington", golds)
    assert not contains_match("Washingtonian", golds)          # whole words only
    long = "word " * 20 + "Washington"
    assert not contains_match(long, golds)                     # rambling answers get no credit
    assert score("Lincoln", golds) == {"em": False, "contains": False, "correct": False}


def test_same_answer():
    assert same_answer("Paris", "paris, france")
    assert not same_answer("Paris", "London")
    assert not same_answer("", "Paris")
