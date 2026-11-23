"""Tests for the packaged cost knowledge base (agents/data/cost_kb.json)."""

from agents.nodes.cost import COST_KB, COST_GRADES, format_cost_option


def _by(hospital: str, name: str, unit: str):
    for option in COST_KB.options:
        if (
            option.hospital == hospital
            and option.name == name
            and option.unit == unit
        ):
            return option
    raise AssertionError(f"missing cost entry: {hospital} / {name} / {unit}")


def test_currency_and_metadata() -> None:
    assert COST_KB.currency == "INR"
    assert COST_KB.as_of
    assert COST_KB.source
    assert COST_KB.notes


def test_injection_prices_match_the_supplied_figures() -> None:
    eylea = _by("ASG Eye Hospital", "Eylea", "one package, per eye")
    accentrix_asg = _by("ASG Eye Hospital", "Accentrix", "one package, per eye")
    accentrix_renuka = _by("Renuka", "Accentrix", "one package, per eye")

    assert eylea.cost_inr == 39900
    assert accentrix_asg.cost_inr == 29450
    assert accentrix_renuka.cost_inr == 27000

    unnamed = "Anti-VEGF injection (brand not specified)"
    assert _by("Renuka", unnamed, "one full dose, per eye").cost_inr == 13000
    assert _by("Renuka", unnamed, "one half dose, per eye").cost_inr == 7000


def test_laser_prices_match_the_supplied_figures() -> None:
    prp = "PRP laser (panretinal photocoagulation)"
    assert _by("ASG Eye Hospital", prp, "one session").cost_inr == 4300
    assert _by("Renuka", prp, "one session for both eyes").cost_inr == 3000


def test_active_ingredients() -> None:
    assert _by("ASG Eye Hospital", "Eylea", "one package, per eye").active_ingredient == "aflibercept"
    assert (
        _by("ASG Eye Hospital", "Accentrix", "one package, per eye").active_ingredient
        == "ranibizumab"
    )
    assert _by("Renuka", "Accentrix", "one package, per eye").active_ingredient == "ranibizumab"
    # The unnamed Renuka medicine must not get an invented brand or ingredient.
    unnamed = _by(
        "Renuka", "Anti-VEGF injection (brand not specified)", "one full dose, per eye"
    )
    assert unnamed.active_ingredient is None


def test_kinds_are_valid_and_complete() -> None:
    kinds = {option.kind for option in COST_KB.options}
    assert kinds == {"injection", "laser"}
    for option in COST_KB.options:
        assert option.cost_inr > 0
        assert option.unit
        assert option.name


def test_notes_say_what_is_excluded() -> None:
    text = COST_KB.note_text().lower()
    assert "per eye" in text
    assert "per session" in text
    assert "consultation" in text
    assert "oct" in text
    assert "not the total" in text


def test_cost_grades() -> None:
    assert COST_GRADES == {2, 3, 4}


def test_formatting_is_readable() -> None:
    line = format_cost_option(_by("ASG Eye Hospital", "Eylea", "one package, per eye"))
    assert "ASG Eye Hospital" in line
    assert "Eylea" in line
    assert "aflibercept" in line
    assert "₹39,900" in line
    assert "per eye" in line

    laser = format_cost_option(
        _by("Renuka", "PRP laser (panretinal photocoagulation)", "one session for both eyes")
    )
    assert "₹3,000" in laser
    assert "both eyes" in laser


if __name__ == "__main__":
    test_currency_and_metadata()
    test_injection_prices_match_the_supplied_figures()
    test_laser_prices_match_the_supplied_figures()
    test_active_ingredients()
    test_kinds_are_valid_and_complete()
    test_notes_say_what_is_excluded()
    test_cost_grades()
    test_formatting_is_readable()
    print("cost KB tests passed")
