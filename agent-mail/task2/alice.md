# Your section: orders/parser.py (alice)

Input is the raw `orders_csv` text.

- Line numbers are physical and 1-based: every line counts, including the
  header, blank lines and comments.
- Blank lines, and lines whose first non-space character is `#`, are ignored.
- The first line that is neither blank nor a comment MAY be a header: if it
  equals `order_id,sku,qty` after removing all spaces and lowercasing, skip it.
- Every other line is a data line. Split on commas and strip each field.
  - not exactly 3 fields, or an empty order_id, or an empty sku: rejected, reason `bad format`
  - qty that is not an integer > 0 (as Python `int()` reads the stripped text): rejected, reason `bad qty`
  - sku is uppercased
- Accepted lines with the same (order_id, sku) are merged into one line: qty
  is summed, and the merged line keeps the line number and position of the
  first one. Rejected lines are never merged.
- Every rejection carries its line number and reason.

## A rule only you know, for bob

The rule below must be implemented in orders/pricing.py, NOT in your file, and nobody else
has it. Make sure it gets implemented.

> Any sku that starts with `GIFT-` is free: its unit price is 0.00 whatever the catalog says.
