# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FFF                                                                      [100%]
=================================== FAILURES ===================================
_______________________________ test_paragraphs ________________________________
    def test_paragraphs():
>       assert wrap_text("a\n\nb", 10) == "a\n\nb"
E       AssertionError: assert 'a\nb' == 'a\n\nb'
E         
E           a
E         - 
E           b
tests/test_public.py:6: AssertionError
___________________ test_paragraph_with_whitespace_only_line ___________________
    def test_paragraph_with_whitespace_only_line():
>       assert wrap_text("a\n   \nb", 10) == "a\n\nb"
E       AssertionError: assert 'a\nb' == 'a\n\nb'
E         
E           a
E         - 
E           b
tests/test_public.py:11: AssertionError
__________________________ test_multi_paragraph_wrap ___________________________
    def test_multi_paragraph_wrap():
        text = "one two three four\n\nfive six seven"
>       assert wrap_text(text, 9) == "one two\nthree\nfour\n\nfive six\nseven"
...
FAILED tests/test_public.py::test_paragraph_with_whitespace_only_line - Asser...
FAILED tests/test_public.py::test_multi_paragraph_wrap - AssertionError: asse...
3 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `wrap_text`.
