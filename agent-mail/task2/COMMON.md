# Team build: the `orders` package

Four agents build one small Python package, `orders/`, in a shared directory.
Each agent owns exactly one file and may READ every file but EDIT only its own.

| agent | file | job |
|---|---|---|
| alice | orders/parser.py | parse the order text into lines, collect rejections |
| carol | orders/inventory.py | allocate stock to order lines |
| bob | orders/pricing.py | price lines and orders |
| dave | orders/report.py | the entry point: run the pipeline, render the invoice |

The entry point (dave's) is:

    from orders.report import process
    text = process(orders_csv: str, catalog: dict, stock: dict) -> str

- `catalog` = `{sku: {"name": str, "price": "12.50"}}` (price is a decimal string)
- `stock` = `{sku: int}` (units on hand)

The interfaces BETWEEN the four modules are not specified anywhere: function
names, arguments, data shapes, who calls whom. The team has to agree on them.
Each agent's section of the spec (yours is below) is known only to that agent.

Hidden integration tests call `process()` and compare its output exactly.
Standard library only. Python 3. `orders/__init__.py` exists and stays empty.
You are done when your section is implemented and `process()` works end to end
on an example you have run.
