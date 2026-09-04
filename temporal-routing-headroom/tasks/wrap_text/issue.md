# Bug report

The test suite in this repository fails.

```
$ python -m pytest tests/ -q
FF                                                                       [100%]
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
__________________________ test_multi_paragraph_wrap ___________________________
    def test_multi_paragraph_wrap():
        text = "one two three four\n\nfive six seven"
>       assert wrap_text(text, 9) == "one two\nthree\nfour\n\nfive six\nseven"
E       AssertionError: assert 'one two\nthr...ve six\nseven' == 'one two\nthr...ve six\nseven'
E         
E           one two
E           three
E           four
E         - 
E           five six
E           seven
tests/test_public.py:12: AssertionError
=========================== short test summary info ============================
FAILED tests/test_public.py::test_paragraphs - AssertionError: assert 'a\nb' ...
FAILED tests/test_public.py::test_multi_paragraph_wrap - AssertionError: asse...
2 failed in <duration>
```

Fix the source so that the suite passes.

Constraints:
- Do not modify anything under `tests/`.
- Do not add new dependencies; the standard library is all that is available.
- Keep the public API of the `solution` package unchanged: `wrap_text`.
