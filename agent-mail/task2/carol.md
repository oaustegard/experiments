# Your section: orders/inventory.py (carol)

You allocate stock to the accepted (already merged) order lines.

- Process lines in input order: ascending by their line number in the file,
  across all orders.
- A line whose sku is not in `stock` is rejected with reason `unknown sku`,
  under that line's line number. It gets no allocation.
- Otherwise: filled = min(qty, remaining stock for that sku);
  backorder = qty - filled; remaining stock drops by filled.
- Never mutate the caller's `stock` dict; the tests reuse it.

## A rule only you know, for dave

The rule below must be implemented in orders/report.py, NOT in your file, and nobody else
has it. Make sure it gets implemented.

> Every order block gets one more line after `total`: `  status PARTIAL` if any of its lines has a backorder, otherwise `  status COMPLETE`.
