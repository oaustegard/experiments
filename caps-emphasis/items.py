"""Item bank: (id, forbidden word, question, forced answer prefix).

Each item is a question whose answer is a single high-probability word, and a
forced assistant prefix that makes that word the natural next token. The screen
in screen_items.py keeps only items where the model puts the word top-1 under no
directive, so a suppression measurement has room to move.
"""

CANDIDATES = [
    ("fr_capital",   "Paris",     "What is the capital of France?",            "The capital of France is"),
    ("jp_capital",   "Tokyo",     "What is the capital of Japan?",             "The capital of Japan is"),
    ("it_capital",   "Rome",      "What is the capital of Italy?",             "The capital of Italy is"),
    ("uk_capital",   "London",    "What is the capital of the United Kingdom?","The capital of the United Kingdom is"),
    ("de_capital",   "Berlin",    "What is the capital of Germany?",           "The capital of Germany is"),
    ("es_capital",   "Madrid",    "What is the capital of Spain?",             "The capital of Spain is"),
    ("ru_capital",   "Moscow",    "What is the capital of Russia?",            "The capital of Russia is"),
    ("cn_capital",   "Beijing",   "What is the capital of China?",             "The capital of China is"),
    ("eg_capital",   "Cairo",     "What is the capital of Egypt?",             "The capital of Egypt is"),
    ("gr_capital",   "Athens",    "What is the capital of Greece?",            "The capital of Greece is"),
    ("pt_capital",   "Lisbon",    "What is the capital of Portugal?",          "The capital of Portugal is"),
    ("no_capital",   "Oslo",      "What is the capital of Norway?",            "The capital of Norway is"),
    ("se_capital",   "Stockholm", "What is the capital of Sweden?",            "The capital of Sweden is"),
    ("pl_capital",   "Warsaw",    "What is the capital of Poland?",            "The capital of Poland is"),
    ("nl_capital",   "Amsterdam", "What is the capital of the Netherlands?",   "The capital of the Netherlands is"),

    ("planet_red",   "Mars",      "Which planet is known as the Red Planet?",  "The Red Planet is"),
    ("planet_big",   "Jupiter",   "Which is the largest planet in the solar system?", "The largest planet in the solar system is"),
    ("star_sun",     "Sun",       "What star is at the centre of our solar system?",  "The star at the centre of our solar system is the"),
    ("moon_earth",   "Moon",      "What is Earth's only natural satellite called?",   "Earth's only natural satellite is called the"),

    ("h2o",          "water",     "What is the common name for the compound H2O?",    "The common name for H2O is"),
    ("gold_sym",     "gold",      "Which metal has the chemical symbol Au?",          "The metal with the chemical symbol Au is"),
    ("iron_sym",     "iron",      "Which metal has the chemical symbol Fe?",          "The metal with the chemical symbol Fe is"),
    ("oxygen",       "oxygen",    "Which gas do humans need to breathe to survive?",  "The gas humans need to breathe is"),

    ("romeo",        "Shakespeare","Who wrote the play Romeo and Juliet?",            "Romeo and Juliet was written by"),
    ("relativity",   "Einstein",  "Who developed the theory of general relativity?",  "The theory of general relativity was developed by"),
    ("gravity_app",  "Newton",    "Which scientist is associated with the falling apple story about gravity?", "The scientist associated with the falling apple story is"),
    ("mona_lisa",    "Leonardo",  "Which artist painted the Mona Lisa? Give the first name only.", "The Mona Lisa was painted by"),

    ("tallest",      "Everest",   "What is the tallest mountain above sea level?",    "The tallest mountain above sea level is Mount"),
    ("longest_riv",  "Nile",      "Which river is traditionally called the longest in the world?", "The river traditionally called the longest is the"),
    ("largest_ocean","Pacific",   "Which is the largest ocean on Earth?",             "The largest ocean on Earth is the"),
    ("desert_big",   "Sahara",    "What is the largest hot desert in the world?",     "The largest hot desert in the world is the"),

    ("dog_sound",    "bark",      "What sound does a dog make? Answer with one verb.", "A dog makes a sound called a"),
    ("cat_animal",   "cat",       "Which common pet is known for purring?",           "The common pet known for purring is the"),
    ("bee_prod",     "honey",     "What sweet substance do bees produce?",            "Bees produce a sweet substance called"),

    ("two_plus",     "four",      "What is two plus two? Write the answer as a word.", "Two plus two is"),
    ("days_week",    "seven",     "How many days are there in a week? Write it as a word.", "The number of days in a week is"),
    ("colors_rain",  "red",       "What is the first colour of the rainbow?",         "The first colour of the rainbow is"),

    ("python_lang",  "Python",    "Which programming language is named after a British comedy troupe?", "The programming language named after a British comedy troupe is"),
    ("html_web",     "HTML",      "What markup language are web pages written in?",   "Web pages are written in"),
    ("git_vcs",      "Git",       "Which version control system did Linus Torvalds create?", "The version control system Linus Torvalds created is"),

    # second tranche
    ("dk_capital",   "Copenhagen","What is the capital of Denmark?",           "The capital of Denmark is"),
    ("at_capital",   "Vienna",    "What is the capital of Austria?",           "The capital of Austria is"),
    ("ie_capital",   "Dublin",    "What is the capital of Ireland?",           "The capital of Ireland is"),
    ("be_capital",   "Brussels",  "What is the capital of Belgium?",           "The capital of Belgium is"),
    ("hu_capital",   "Budapest",  "What is the capital of Hungary?",           "The capital of Hungary is"),
    ("cz_capital",   "Prague",    "What is the capital of the Czech Republic?","The capital of the Czech Republic is"),
    ("fi_capital",   "Helsinki",  "What is the capital of Finland?",           "The capital of Finland is"),
    ("tr_capital",   "Ankara",    "What is the capital of Turkey?",            "The capital of Turkey is"),
    ("in_capital",   "Delhi",     "What is the capital of India?",             "The capital of India is New"),
    ("ca_capital",   "Ottawa",    "What is the capital of Canada?",            "The capital of Canada is"),
    ("au_capital",   "Canberra",  "What is the capital of Australia?",         "The capital of Australia is"),
    ("br_capital",   "Bras",      "What is the capital of Brazil?",            "The capital of Brazil is"),
    ("ar_capital",   "Buenos",    "What is the capital of Argentina?",         "The capital of Argentina is"),
    ("mx_capital",   "Mexico",    "What is the capital of Mexico?",            "The capital of Mexico is"),
    ("ke_capital",   "Nairobi",   "What is the capital of Kenya?",             "The capital of Kenya is"),
    ("th_capital",   "Bangkok",   "What is the capital of Thailand?",          "The capital of Thailand is"),
    ("kr_capital",   "Seoul",     "What is the capital of South Korea?",       "The capital of South Korea is"),
    ("pe_capital",   "Lima",      "What is the capital of Peru?",              "The capital of Peru is"),
    ("ch_capital",   "Bern",      "What is the capital of Switzerland?",       "The capital of Switzerland is"),
    ("us_capital",   "Washington","What is the capital of the United States?", "The capital of the United States is"),
]
