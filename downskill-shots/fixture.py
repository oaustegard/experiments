"""Test items for the down-skilling example ablation.

kind = "rule": the distilled prompt's rules alone determine the gold label.
kind = "convention": the rules leave it open and one of the prompt's worked
examples settles it (the example it mirrors is named). Gold may be a list when
two answers are defensible. Gold written by Opus 5.5 from the prompts' own
examples, which is this eval's main caveat.
"""

FEEDBACK = [  # gold: (sentiment, theme)
    (1, "Checkout is so much quicker since the update, thanks!", ("POSITIVE", "PERFORMANCE"), "convention", "ex1: faster -> PERFORMANCE, not UX"),
    (2, "The new colour scheme is beautiful but the app freezes whenever I upload a photo.", ("MIXED", "PERFORMANCE"), "convention", "ex3: crash beats UX praise"),
    (3, "Our plan renews on the 1st of each month.", ("NEUTRAL", "PRICING"), "convention", "ex4: billing statement, no opinion"),
    (4, "Would love a dark mode. Honestly it's the only reason I haven't moved my whole team over.", ("MIXED", "FEATURE_REQUEST"), "convention", "ex7: wish plus blocked recommendation"),
    (5, "qwerty uiop", ("NEUTRAL", "OTHER"), "rule", "rule 6"),
    (6, "Mark from support stayed late to fix our SSO config. Above and beyond.", ("POSITIVE", "SUPPORT"), "rule", ""),
    (7, "Way too expensive for what you get.", ("NEGATIVE", "PRICING"), "rule", ""),
    (8, "The settings menu is buried three levels deep and I can never find anything.", ("NEGATIVE", "UX"), "rule", ""),
    (9, "Customer since 2019. Loved v2, hated the v3 redesign, v4 fixed the menus but now reports take a full minute to load.", ("MIXED", "PERFORMANCE"), "convention", "ex9: current complaint wins"),
    (10, "Support never answered my ticket. Two weeks and nothing.", ("NEGATIVE", "SUPPORT"), "rule", ""),
    (11, "", ("NEUTRAL", "OTHER"), "rule", "rule 6"),
    (12, "Great value at this price point.", ("POSITIVE", "PRICING"), "rule", ""),
]

MODERATION = [  # gold: (verdict or [verdicts], policy)
    (1, "This review is lazy trash, whoever wrote it should find a new job.", ("APPROVED", "NONE"), "convention", "ex1: harsh at the content, not a person's group"),
    (2, "nobody would miss you if you disappeared, do everyone a favour", ("REMOVED", "HARASSMENT"), "convention", "ex3: self-harm encouragement"),
    (3, "This documentary examines the history of racist propaganda in the 1930s.", ("APPROVED", "NONE"), "rule", "rule 7"),
    (4, "Oh sure, 'brilliant' analysis from someone who clearly can't read 🙄", ("FLAGGED", "HARASSMENT"), "convention", "ex5: sarcastic insult -> FLAGGED"),
    (5, "BUY FOLLOWERS CHEAP!!! 10k for $5 >>> [link] [link] [link]", ("REMOVED", "SPAM"), "rule", ""),
    (6, "According to WHO data (2023), measles cases rose 79% globally.", ("APPROVED", "NONE"), "convention", "ex7: sourced claim"),
    (7, "5G towers are spreading the virus. Unplug your router to stay safe.", ("REMOVED", "MISINFORMATION"), "convention", "ex8: conspiracy with harmful health advice"),
    (8, "I'm not sexist, but women just aren't suited to engineering leadership.", ("FLAGGED", "HATE_SPEECH"), "convention", "ex9: implicit stereotype -> FLAGGED"),
    (9, "Thanks, this fixed my build error!", ("APPROVED", "NONE"), "rule", ""),
    (10, "People from that country are vermin and should all be thrown out.", ("REMOVED", "HATE_SPEECH"), "rule", "clear violation"),
    (11, "I disagree with your take on interest rates; the 2008 data points the other way.", ("APPROVED", "NONE"), "rule", ""),
    (12, "You're an idiot and everyone at your company knows it.", (["FLAGGED", "REMOVED"], "HARASSMENT"), "rule", "direct personal insult"),
]

SQL_SCHEMA = ("customers(id, name, country, signup_date), orders(id, customer_id, order_date, total, status), "
              "order_items(id, order_id, product_id, quantity, unit_price), products(id, name, category, price)")

SQL = [  # gold: a reference query, or None for "cannot answer"
    (1, "Who are our top 3 customers by total spending?",
     "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id WHERE o.status = 'completed' GROUP BY c.id, c.name ORDER BY SUM(o.total) DESC LIMIT 3",
     "convention", "ex1: spending counts completed orders only"),
    (2, "How many orders were placed in 2025?",
     "SELECT COUNT(*) FROM orders WHERE order_date >= '2025-01-01' AND order_date < '2026-01-01'", "rule", ""),
    (3, "Which product category brings in the most revenue?",
     "SELECT p.category FROM order_items oi JOIN products p ON p.id = oi.product_id GROUP BY p.category ORDER BY SUM(oi.quantity * oi.unit_price) DESC LIMIT 1",
     "rule", ""),
    (4, "List some customers from Norway.", "LIMIT10:Norway", "rule", "rule 7: 'some' -> LIMIT 10"),
    (5, "What is the average order value per country?",
     "SELECT c.country, AVG(o.total) FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.country", "rule", ""),
    (6, "Which customers have never placed an order?",
     "SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id = c.id WHERE o.id IS NULL", "rule", ""),
    (7, "What is each customer's phone number?", None, "rule", "rule 9"),
    (8, "Which month of 2025 had the most orders?",
     "SELECT strftime('%m', order_date) AS m FROM orders WHERE order_date LIKE '2025-%' GROUP BY m ORDER BY COUNT(*) DESC LIMIT 1",
     "rule", ""),
]
