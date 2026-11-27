from scripts.fx import titlecase, chunk, parse_kv, median, dedupe, days_in_month


# titlecase: "Capitalize the first letter of each space-separated word and lowercase the rest."

def test_titlecase_capitalizes_each_word():
    assert titlecase("hello world") == "Hello World"


def test_titlecase_lowercases_rest_of_word():
    assert titlecase("hELLO wORLD") == "Hello World"


def test_titlecase_empty_string():
    assert titlecase("") == ""


def test_titlecase_keeps_empty_words_from_extra_spaces():
    assert titlecase("a  b") == "A  B"


# chunk: "Split a list into consecutive lists of length n. The last list may be shorter."

def test_chunk_even_split():
    assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_chunk_last_list_shorter():
    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_chunk_empty_list():
    assert chunk([], 3) == []


def test_chunk_n_larger_than_list():
    assert chunk([1, 2], 5) == [[1, 2]]


# parse_kv: "Parse 'key=value' into a (key, value) tuple, splitting on the first '='."

def test_parse_kv_basic_pair():
    assert parse_kv("a=b") == ("a", "b")


def test_parse_kv_splits_on_first_equals_only():
    assert parse_kv("a=b=c") == ("a", "b=c")


def test_parse_kv_empty_value():
    assert parse_kv("a=") == ("a", "")


# median: "Median of a non-empty list. For an even count, the mean of the two middle values."

def test_median_odd_count():
    assert median([3, 1, 2]) == 2


def test_median_single_value():
    assert median([7]) == 7


def test_median_even_count_is_mean_of_middle_pair():
    assert median([1, 2, 3, 4]) == 2.5


# dedupe: "Remove duplicates from a list, keeping the first occurrence of each item in order."

def test_dedupe_keeps_first_occurrence_order():
    assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]


def test_dedupe_no_duplicates_unchanged():
    assert dedupe([1, 2, 3]) == [1, 2, 3]


def test_dedupe_empty_list():
    assert dedupe([]) == []


def test_dedupe_all_same_item():
    assert dedupe(["a", "a", "a"]) == ["a"]


# days_in_month: "Number of days in the given month, accounting for leap years."

def test_days_in_month_31_day_months():
    for m in (1, 3, 5, 7, 8, 10, 12):
        assert days_in_month(2023, m) == 31


def test_days_in_month_30_day_months():
    for m in (4, 6, 9, 11):
        assert days_in_month(2023, m) == 30


def test_days_in_month_february_leap_year():
    assert days_in_month(2024, 2) == 29


def test_days_in_month_february_common_year():
    assert days_in_month(2023, 2) == 28
