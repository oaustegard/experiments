from scripts.fx import titlecase, chunk, parse_kv, median, dedupe, days_in_month


# titlecase: "Capitalize the first letter of each space-separated word and lowercase the rest."

def test_titlecase_capitalizes_each_word():
    assert titlecase("hello world") == "Hello World"


def test_titlecase_lowercases_rest_of_each_word():
    assert titlecase("hELLO wORLD") == "Hello World"


def test_titlecase_hyphen_is_not_a_word_separator():
    assert titlecase("well-known fact") == "Well-known Fact"


def test_titlecase_empty_string():
    assert titlecase("") == ""


# chunk: "Split a list into consecutive lists of length n. The last list may be shorter."

def test_chunk_splits_into_consecutive_lists_with_short_tail():
    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_chunk_exact_multiple_has_no_short_tail():
    assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_chunk_n_larger_than_list_returns_one_list():
    assert chunk([1, 2], 5) == [[1, 2]]


def test_chunk_empty_list_returns_empty_list():
    assert chunk([], 3) == []


# parse_kv: "Parse 'key=value' into a (key, value) tuple, splitting on the first '='."

def test_parse_kv_basic_pair():
    assert parse_kv("name=oskar") == ("name", "oskar")


def test_parse_kv_splits_on_first_equals_only():
    assert parse_kv("expr=a=b") == ("expr", "a=b")


def test_parse_kv_empty_value():
    assert parse_kv("key=") == ("key", "")


# median: "Median of a non-empty list. For an even count, the mean of the two middle values."

def test_median_odd_count_returns_middle_value():
    assert median([3, 1, 2]) == 2


def test_median_single_value():
    assert median([7]) == 7


def test_median_even_count_is_mean_of_two_middle_values():
    assert median([1, 2, 3, 4]) == 2.5


# dedupe: "Remove duplicates from a list, keeping the first occurrence of each item in order."

def test_dedupe_keeps_first_occurrence_in_order():
    assert dedupe([1, 2, 1, 3, 2]) == [1, 2, 3]


def test_dedupe_does_not_sort_first_occurrences():
    assert dedupe(["b", "a", "b"]) == ["b", "a"]


def test_dedupe_no_duplicates_returns_same_order():
    assert dedupe([3, 1, 2]) == [3, 1, 2]


def test_dedupe_empty_list():
    assert dedupe([]) == []


# days_in_month: "Number of days in the given month, accounting for leap years."

def test_days_in_month_thirty_day_months():
    for m in (4, 6, 9, 11):
        assert days_in_month(2023, m) == 30


def test_days_in_month_thirty_one_day_months():
    for m in (1, 3, 5, 7, 8, 10, 12):
        assert days_in_month(2023, m) == 31


def test_days_in_month_february_common_year():
    assert days_in_month(2023, 2) == 28


def test_days_in_month_february_leap_year():
    assert days_in_month(2024, 2) == 29
