# Your section: orders/report.py (dave)

`process(orders_csv, catalog, stock)` runs the pipeline and returns the invoice text.

Exact format, two-space indents, money always with 2 decimals:

    ORDER A1
      WIDGET Blue Widget x3 @ 2.50 = 7.50
      GADGET Gadget x5 @ 10.00 = 50.00 (backorder 2)
      subtotal 57.50
      tax 4.74
      shipping 0.00
      total 62.24

    ORDER B7
      ...

    REJECTED
      line 4: bad qty
      line 9: unknown sku

- Line: `<SKU> <catalog name> x<filled> @ <unit price> = <line total>`, plus
  ` (backorder N)` only when N > 0. A line with filled 0 still prints.
- Orders sorted by order_id (plain string sort); lines in input order.
- An order is printed only if it has at least one line that was not rejected.
- One blank line between blocks. The REJECTED block comes last, only if
  anything was rejected, sorted by line number.
- Nothing at all to print: return `NO ORDERS`.
- No trailing newline.
