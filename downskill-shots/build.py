"""Build the arm prompts and the SQLite fixture.

Arms, per task, from the down-skilling skill's own distilled prompts:
  full  - the prompt as the skill ships it
  rules - the same prompt with <examples>...</examples> removed
  bare  - role, task and output format only: label names, no definitions or tie-breaks
"""
import random
import re
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixture as F  # noqa: E402

SKILL = Path("/home/user/claude-skills/down-skilling/examples")  # skill checkout, d854510..66d1719
# Runs are Agent-tool subagents that read this file; the header keeps them from using tools on the task.
NO_TOOLS = ("These are your complete task instructions. Use no tools beyond having read this file. "
            "Reply with only the requested output, nothing before or after it.\n\n")


def shipped(name: str) -> str:
    lines = (SKILL / f"{name}.md").read_text().splitlines()
    start = lines.index("```xml") + 1
    end = next(i for i in range(len(lines)) if lines[i].startswith("## Why it works")) - 1
    while not lines[end].startswith("```"):
        end -= 1
    return "\n".join(lines[start:end])


def strip_examples(p: str) -> str:
    return re.sub(r"\n*<examples>.*?</examples>\n*", "\n\n", p, flags=re.S)


BARE = {
    "feedback": """<task>
Read each comment. Extract sentiment (POSITIVE | NEGATIVE | MIXED | NEUTRAL), theme
(PRICING | UX | PERFORMANCE | SUPPORT | FEATURE_REQUEST | OTHER) and a one-sentence summary.
Output a JSON array of {id, sentiment, theme, summary}.
</task>

<context>
{{comments}}
</context>""",
    "moderation": """<task>
Classify each comment. verdict: APPROVED | FLAGGED | REMOVED. policy: HATE_SPEECH | HARASSMENT |
SPAM | MISINFORMATION | NONE. confidence: HIGH | MEDIUM | LOW. Output a JSON array of
{id, verdict, policy, confidence, reason}.
</task>

<context>
{{comments}}
</context>""",
    "sql": """<task>
Write a SQL query (PostgreSQL) that answers the question from the schema. If the schema cannot
answer it, output "-- Cannot answer: [reason]". Put the query in a ```sql block, then one
sentence of explanation.
</task>

<schema>
{{database_schema}}
</schema>

<context>
{{user_question}}
</context>""",
}


def comments(items) -> str:
    return "\n".join(f'Comment {i}: "{t}"' for i, t, *_ in items)


def sql_questions() -> str:
    return ("Answer each of these questions separately, headed Q1 to Q8, applying the format above to each.\n"
            + "\n".join(f'Q{i}: "{q}"' for i, q, *_ in F.SQL))


def prompts() -> dict:
    base = {"feedback": shipped("data-extraction"), "moderation": shipped("content-moderation"),
            "sql": shipped("sql-generation")}
    out = {}
    for task, p in base.items():
        for arm, text in (("full", p), ("rules", strip_examples(p)), ("bare", BARE[task])):
            if task == "sql":
                text = text.replace("{{database_schema}}", F.SQL_SCHEMA).replace("{{user_question}}", sql_questions())
            else:
                text = text.replace("{{comments}}", comments(F.FEEDBACK if task == "feedback" else F.MODERATION))
            out[f"{task}-{arm}"] = NO_TOOLS + text
    return out


def build_db(path: Path) -> None:
    """Deterministic data where 'completed only' changes the top-3 spenders."""
    rnd = random.Random(7)
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.executescript("""
      CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, country TEXT, signup_date TEXT);
      CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INT, order_date TEXT, total REAL, status TEXT);
      CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
      CREATE TABLE order_items(id INTEGER PRIMARY KEY, order_id INT, product_id INT, quantity INT, unit_price REAL);""")
    countries = ["Norway"] * 14 + ["Sweden"] * 10 + ["Denmark"] * 8 + ["USA"] * 12
    for i, c in enumerate(countries, 1):
        db.execute("INSERT INTO customers VALUES (?,?,?,?)", (i, f"cust{i:02d}", c, f"2023-{rnd.randint(1,12):02d}-01"))
    cats = ["books", "tools", "garden", "toys"]
    for i in range(1, 21):
        db.execute("INSERT INTO products VALUES (?,?,?,?)", (i, f"prod{i}", cats[i % 4], rnd.randint(5, 90)))
    oid = 0
    for cid in range(1, 41):  # customers 41-44 never order
        for _ in range(rnd.randint(1, 6)):
            oid += 1
            month = rnd.choice([1, 2, 3, 3, 3, 5, 6, 8, 9, 11]) if rnd.random() < 0.8 else rnd.randint(1, 12)
            year = 2025 if rnd.random() < 0.7 else 2024
            status = "completed" if rnd.random() < 0.75 else rnd.choice(["cancelled", "refunded"])
            total = 0.0
            for _ in range(rnd.randint(1, 3)):
                pid = rnd.randint(1, 20)
                qty = rnd.randint(1, 4) * (3 if pid % 4 == 1 else 1)
                price = db.execute("SELECT price FROM products WHERE id=?", (pid,)).fetchone()[0]
                db.execute("INSERT INTO order_items(order_id,product_id,quantity,unit_price) VALUES (?,?,?,?)",
                           (oid, pid, qty, price))
                total += qty * price
            db.execute("INSERT INTO orders VALUES (?,?,?,?,?)",
                       (oid, cid, f"{year}-{month:02d}-{rnd.randint(1,28):02d}", round(total, 2), status))
    # Three big cancelled orders, so 'all statuses' and 'completed only' disagree on the top 3
    for cid in (5, 17, 29):
        oid += 1
        db.execute("INSERT INTO orders VALUES (?,?,?,?,?)", (oid, cid, "2025-07-04", 9000.0, "cancelled"))
    db.commit()
    db.close()


if __name__ == "__main__":
    out = HERE / "prompts"
    out.mkdir(exist_ok=True)
    for k, v in prompts().items():
        (out / f"{k}.txt").write_text(v)
    build_db(HERE / "shop.db")
    db = sqlite3.connect(HERE / "shop.db")
    top_c = [r[0] for r in db.execute(F.SQL[0][2])]
    top_all = [r[0] for r in db.execute(F.SQL[0][2].replace("WHERE o.status = 'completed' ", ""))]
    print("top3 completed", top_c, "| all statuses", top_all)
    for i, q, gold, *_ in F.SQL:
        if gold and not gold.startswith("LIMIT"):
            print(i, db.execute(gold).fetchall()[:4])
