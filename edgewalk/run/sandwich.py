"""The sandwich board: its pieces, the question, and the cases Jev labels.

A case is one value per piece. `sample(rng)` draws the training cases: a third uniformly at random,
two thirds by walking 1-8 single-piece moves away from a well-known food, which is how a game moves.
"""
import random

QUESTION = {"is_sandwich": {"type": "noul", "instructions": "This food is a sandwich."}}

PIECES = [
    ("outside", "What holds it", [
        "two slices of sandwich bread", "a sub roll", "a hot dog bun", "a hamburger bun", "a bagel",
        "a croissant", "a pita pocket", "a flour tortilla", "a crispy taco shell", "two waffles",
        "two chocolate chip cookies", "two lettuce leaves", "a sheet of nori", "two rice cakes",
        "two slices of pizza", "two pancakes"]),
    ("assembly", "How it's put together", [
        "stacked between two separate pieces", "a single piece split partway, hinged on one side",
        "folded in half", "open-faced, toppings on one layer", "rolled up into a cylinder",
        "stuffed into a pocket", "wrapped and sealed all around"]),
    ("filling", "Filling", [
        "ham and cheese", "peanut butter and jelly", "a grilled sausage", "a beef patty",
        "a scoop of vanilla ice cream", "scrambled eggs", "tuna salad", "seasoned ground beef",
        "smashed avocado", "chocolate spread", "a fried fish fillet", "raw salmon and rice",
        "another slice of bread", "nothing at all"]),
    ("sauce", "Sauce", [
        "none", "mayonnaise", "ketchup and mustard", "salsa", "gravy poured over the top",
        "maple syrup", "whipped cream", "soy sauce"]),
    ("temperature", "Temperature", ["frozen", "cold", "room temperature", "warm", "hot off the grill"]),
    ("utensils", "Eaten with", ["hands", "knife and fork", "a spoon", "chopsticks"]),
    ("size", "Size", ["one bite", "fits in your palm", "the length of your forearm", "six feet long, for a party"]),
    ("venue", "Bought at", [
        "a deli counter", "a ballpark", "a taco truck", "an ice cream parlour", "a sushi bar",
        "a diner, for breakfast", "a French bakery", "a school cafeteria"]),
    ("menu_name", "The menu calls it", [
        "a sandwich", "a hot dog", "a burger", "a wrap", "a taco", "a dessert", "a roll", "nothing; it has no name"]),
]
ORDERED = {"temperature", "size"}   # distance in the game is fractional along these

CLASSICS = {
    "deli ham and cheese": ["two slices of sandwich bread", "stacked between two separate pieces", "ham and cheese", "mayonnaise", "cold", "hands", "fits in your palm", "a deli counter", "a sandwich"],
    "hot dog": ["a hot dog bun", "a single piece split partway, hinged on one side", "a grilled sausage", "ketchup and mustard", "hot off the grill", "hands", "fits in your palm", "a ballpark", "a hot dog"],
    "burger": ["a hamburger bun", "stacked between two separate pieces", "a beef patty", "ketchup and mustard", "hot off the grill", "hands", "fits in your palm", "a diner, for breakfast", "a burger"],
    "taco": ["a crispy taco shell", "folded in half", "seasoned ground beef", "salsa", "warm", "hands", "fits in your palm", "a taco truck", "a taco"],
    "burrito": ["a flour tortilla", "wrapped and sealed all around", "seasoned ground beef", "salsa", "warm", "hands", "the length of your forearm", "a taco truck", "a wrap"],
    "ice cream sandwich": ["two chocolate chip cookies", "stacked between two separate pieces", "a scoop of vanilla ice cream", "none", "frozen", "hands", "fits in your palm", "an ice cream parlour", "a dessert"],
    "sushi hand roll": ["a sheet of nori", "rolled up into a cylinder", "raw salmon and rice", "soy sauce", "cold", "chopsticks", "one bite", "a sushi bar", "a roll"],
    "avocado toast": ["two slices of sandwich bread", "open-faced, toppings on one layer", "smashed avocado", "none", "warm", "knife and fork", "fits in your palm", "a French bakery", "nothing; it has no name"],
    "PB&J": ["two slices of sandwich bread", "stacked between two separate pieces", "peanut butter and jelly", "none", "room temperature", "hands", "fits in your palm", "a school cafeteria", "a sandwich"],
    "fish-and-waffles stack": ["two waffles", "stacked between two separate pieces", "a fried fish fillet", "maple syrup", "hot off the grill", "knife and fork", "fits in your palm", "a diner, for breakfast", "nothing; it has no name"],
    "tuna pita": ["a pita pocket", "stuffed into a pocket", "tuna salad", "mayonnaise", "cold", "hands", "fits in your palm", "a deli counter", "a sandwich"],
    "party sub": ["a sub roll", "a single piece split partway, hinged on one side", "ham and cheese", "mayonnaise", "cold", "hands", "six feet long, for a party", "a deli counter", "a sandwich"],
}
KEYS = [k for k, _, _ in PIECES]
VALUES = {k: v for k, _, v in PIECES}


def state(case):
    """The case as Jev sees it: readable keys, one value each."""
    return {label: case[k] for k, label, _ in PIECES}


def sample(rng: random.Random):
    if rng.random() < 1 / 3:
        return {k: rng.choice(VALUES[k]) for k in KEYS}
    case = dict(zip(KEYS, rng.choice(list(CLASSICS.values()))))
    for _ in range(rng.randint(1, 8)):
        k = rng.choice(KEYS)
        case[k] = rng.choice([v for v in VALUES[k] if v != case[k]])
    return case
