# Your section: orders/pricing.py (bob)

All money is exact decimal (never float), rounded ROUND_HALF_UP to cents.

- unit price = `catalog[sku]["price"]`.
- line total = filled * unit price; if filled >= 10, multiply by 0.90.
  Round the line total to cents.
- order subtotal = sum of its line totals.
- tax = subtotal * 0.0825, rounded to cents.
- shipping = 5.00 if 0 < subtotal < 50.00, else 0.00.
- total = subtotal + tax + shipping.
- Pricing uses FILLED quantity, not ordered quantity.

## A rule only you know, for carol

The rule below must be implemented in orders/inventory.py, NOT in your file, and nobody else
has it. Make sure it gets implemented.

> Keep 2 units of every sku in reserve: only (stock - 2), never below 0, can be allocated.
