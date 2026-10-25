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


def test_lists_and_hedges_get_no_containment_credit():
    # from the pilot hand check: one right city inside a wrong list is not a correct answer
    assert not score("New York City, Tokyo, and Beijing", ["Beijing", "Shanghai", "Delhi"])["correct"]
    assert not score("Paris or Lyon", ["Paris"])["correct"]
    assert not score("Butane or propane", ["lighter fluid", "butane"])["correct"]
    assert score("B3 or niacin", ["B3", "niacin", "Vitamin B3"])["correct"]   # synonyms, not a hedge
    assert score("Seattle, Washington", ["Seattle"])["correct"]          # place, region still counts
    assert score("Football (also known as soccer)", ["football"])["correct"]


def test_same_answer():
    assert same_answer("Paris", "paris, france")
    assert not same_answer("Paris", "London")
    assert not same_answer("", "Paris")
