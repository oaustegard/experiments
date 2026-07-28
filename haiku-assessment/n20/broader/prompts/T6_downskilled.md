<role>
You apply filter/sort/select procedures over small tabular inputs. You return exact answers and show the constraint application.
</role>

<task>
Apply the stated constraints to the input table. Output:
- Line 1: "Eligible after filter: <comma-separated titles>"
- Line 2: "Top by rating: 1) <title> 2) <title> 3) <title>"

No preamble. No closing summary.
</task>

<rules>
1. Apply filters first, then sort, then select.
2. "After year X" means year > X (strictly later). 2015 is excluded if the constraint is "after 2015".
3. "Under N pages" means pages < N (strictly less). 400 is excluded if the constraint is "under 400".
4. Ties on the sort key: break by alphabetical title (ascending).
5. If fewer than 3 items pass the filter, list as many as exist and stop. Do not pad.
6. Do NOT include items that fail the filter in the "Top by rating" line, even as runner-ups.
</rules>

<process>
1. List the eligible titles (those passing both filters).
2. Sort eligible by rating descending; break ties by title ascending.
3. Take the top 3.
4. Emit the two lines.
</process>

<examples>
EXAMPLE 1:
Constraints: pick top 2 by rating; year > 2010 AND pages < 300.
Table:
| Title | Year | Pages | Rating |
| A | 2009 | 200 | 4.8 |
| B | 2015 | 250 | 4.5 |
| C | 2018 | 290 | 4.5 |
| D | 2020 | 310 | 4.7 |
Output:
Eligible after filter: B, C
Top by rating: 1) B 2) C
(NOTE: B and C tie at 4.5; B wins by alphabetical title.)

EXAMPLE 2 (filter leaves only one item):
Constraints: pick top 3; year > 2020 AND pages < 400.
Table:
| Title | Year | Pages | Rating |
| Alpha | 2021 | 350 | 4.0 |
| Beta | 2019 | 300 | 4.5 |
Output:
Eligible after filter: Alpha
Top by rating: 1) Alpha
</examples>

<input>
Constraints: pick top 3 by rating; year > 2015 AND pages < 400.
Table:
| Title | Year | Pages | Rating |
|---|---|---|---|
| Project Hail Mary | 2021 | 476 | 4.5 |
| Klara and the Sun | 2021 | 320 | 4.2 |
| The Overstory | 2018 | 502 | 4.1 |
| Piranesi | 2020 | 272 | 4.4 |
| Educated | 2018 | 334 | 4.3 |
| The Vanishing Half | 2020 | 352 | 4.0 |
| Atomic Habits | 2018 | 320 | 4.4 |
</input>
