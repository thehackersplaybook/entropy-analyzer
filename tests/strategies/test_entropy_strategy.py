import pytest
from entropy_analyzer.strategies import EntropyFactory


@pytest.fixture
def entropy_calculators():
    return {
        "text": EntropyFactory.get_entropy_calculator("text"),
        "numerical": EntropyFactory.get_entropy_calculator("numerical"),
        "search": EntropyFactory.get_entropy_calculator("search"),
        "contextual": EntropyFactory.get_entropy_calculator("contextual"),
        "time": EntropyFactory.get_entropy_calculator("time"),
    }


def test_text_entropy(entropy_calculators):
    calc = entropy_calculators["text"]
    assert calc.compute_entropy("") == 0.0
    assert 0 <= calc.compute_entropy("hello") <= 1
    assert calc.compute_entropy("aaaa") < calc.compute_entropy("abcd")


def test_numerical_entropy(entropy_calculators):
    calc = entropy_calculators["numerical"]
    assert calc.compute_entropy([]) == 0.0
    assert 0 <= calc.compute_entropy([1, 2, 3, 4, 5]) <= 1
    assert calc.compute_entropy([1, 1, 1]) < calc.compute_entropy([1, 2, 3])


def test_search_entropy(entropy_calculators):
    calc = entropy_calculators["search"]
    assert calc.compute_entropy([]) == 0.0
    results = ["python programming", "snake species", "python web framework"]
    assert 0 <= calc.compute_entropy(results) <= 1


def test_contextual_entropy(entropy_calculators):
    calc = entropy_calculators["contextual"]
    assert calc.compute_entropy("") == 0.0
    assert 0 <= calc.compute_entropy("Simple predictable text") <= 1


def test_time_entropy(entropy_calculators):
    calc = entropy_calculators["time"]
    assert calc.compute_entropy([]) == 0.0
    timestamps = ["2023-01-01T00:00:00", "2023-01-02T00:00:00"]
    assert 0 <= calc.compute_entropy(timestamps) <= 1


def test_factory_invalid_type():
    with pytest.raises(ValueError, match="Invalid Strategy Type: invalid"):
        EntropyFactory.get_entropy_calculator("invalid")

    with pytest.raises(ValueError, match="Strategy type must be a string"):
        EntropyFactory.get_entropy_calculator(None)


def test_text_entropy_invalid_input(entropy_calculators):
    calc = entropy_calculators["text"]
    with pytest.raises(ValueError, match="Input must be a string or None"):
        calc.compute_entropy(123)
    with pytest.raises(ValueError, match="Input must be a string or None"):
        calc.compute_entropy([1, 2, 3])


def test_text_entropy_special_cases(entropy_calculators):
    calc = entropy_calculators["text"]
    assert calc.compute_entropy(None) == 0.0
    assert calc.compute_entropy("🎉🌟") > 0  # Unicode/emoji
    assert calc.compute_entropy("a" * 1000) < calc.compute_entropy(
        "a" * 500 + "b" * 500
    )
    assert calc.compute_entropy("\n\t\r ") > 0  # Whitespace chars


def test_numerical_entropy_invalid_input(entropy_calculators):
    calc = entropy_calculators["numerical"]
    with pytest.raises(ValueError, match="Input must be a list of numbers or None"):
        calc.compute_entropy("123")
    with pytest.raises(ValueError, match="Input contains non-finite values"):
        calc.compute_entropy([1, float("inf"), 3])
    with pytest.raises(ValueError):
        calc.compute_entropy([1, "2", 3])


def test_numerical_entropy_special_cases(entropy_calculators):
    calc = entropy_calculators["numerical"]
    assert calc.compute_entropy(None) == 0.0
    assert calc.compute_entropy([1.5, -1.5, 0.0]) > 0
    assert calc.compute_entropy([1e9, 2e9, 3e9]) > 0  # Large numbers
    assert calc.compute_entropy([0.0000001, 0.0000002]) > 0  # Small numbers


def test_search_entropy_invalid_input(entropy_calculators):
    calc = entropy_calculators["search"]
    with pytest.raises(ValueError, match="Input must be a list of strings or None"):
        calc.compute_entropy("not a list")
    with pytest.raises(ValueError):
        calc.compute_entropy([1, 2, 3])


def test_search_entropy_special_cases(entropy_calculators):
    calc = entropy_calculators["search"]
    assert calc.compute_entropy(None) == 0.0
    # Test identical results
    assert calc.compute_entropy(["same", "same", "same"]) < calc.compute_entropy(
        ["different", "totally different", "completely different"]
    )
    # Test with special characters
    assert calc.compute_entropy(["hello!", "hello?", "hello..."]) > 0


def test_contextual_entropy_invalid_input(entropy_calculators):
    calc = entropy_calculators["contextual"]
    with pytest.raises(ValueError, match="Input must be a string or None"):
        calc.compute_entropy(123)
    with pytest.raises(ValueError, match="Input must be a string or None"):
        calc.compute_entropy(["list"])


def test_time_entropy_invalid_input(entropy_calculators):
    calc = entropy_calculators["time"]
    with pytest.raises(
        ValueError, match="Input must be a list of timestamp strings or None"
    ):
        calc.compute_entropy("2023-01-01")
    with pytest.raises(ValueError):
        calc.compute_entropy(["invalid date", "2023-01-01"])
    with pytest.raises(ValueError):
        calc.compute_entropy(["2023-13-01"])  # Invalid month


def test_time_entropy_special_cases(entropy_calculators):
    calc = entropy_calculators["time"]
    assert calc.compute_entropy(None) == 0.0
    # Test regular intervals vs irregular intervals
    regular = ["2023-01-01T00:00:00", "2023-01-01T01:00:00", "2023-01-01T02:00:00"]
    irregular = ["2023-01-01T00:00:00", "2023-01-01T00:01:00", "2023-01-01T10:00:00"]
    reg_entropy = calc.compute_entropy(regular)
    irreg_entropy = calc.compute_entropy(irregular)
    assert (
        reg_entropy <= irreg_entropy
    ), f"Regular {reg_entropy} should be <= Irregular {irreg_entropy}"


def test_factory_edge_cases():

    # Test various invalid types return TextEntropy
    invalid_types = ["", "invalid", "unknown", "TEST"]
    for invalid_type in invalid_types:
        with pytest.raises(ValueError, match=f"Invalid Strategy Type: {invalid_type}"):
            EntropyFactory.get_entropy_calculator(invalid_type)


def test_cross_strategy_comparisons(entropy_calculators):
    """Test that different strategies handle similar input patterns consistently"""
    text_calc = entropy_calculators["text"]
    numerical_calc = entropy_calculators["numerical"]

    # Both should show higher entropy for diverse input
    assert text_calc.compute_entropy("aaaa") < text_calc.compute_entropy("abcd")
    assert numerical_calc.compute_entropy(
        [1, 1, 1, 1]
    ) < numerical_calc.compute_entropy([1, 2, 3, 4])


# Performance test for large inputs
def test_large_inputs(entropy_calculators):
    large_text = "abc" * 10000
    large_numbers = list(range(10000))
    large_timestamps = [
        f"2023-{str(i//12 + 1).zfill(2)}-{str(i%12 + 1).zfill(2)}T00:00:00"
        for i in range(99)
    ]

    for calc_type, calc in entropy_calculators.items():
        try:
            if calc_type == "text":
                assert 0 <= calc.compute_entropy(large_text) <= 1
            elif calc_type == "numerical":
                assert 0 <= calc.compute_entropy(large_numbers) <= 1
            elif calc_type == "time":
                assert 0 <= calc.compute_entropy(large_timestamps) <= 1
            elif calc_type == "search":
                results = [f"Result {i}" for i in range(100)]
                assert 0 <= calc.compute_entropy(results) <= 1
        except Exception as e:
            pytest.fail(f"Large input failed for {calc_type}: {str(e)}")
