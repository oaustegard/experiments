"""Build run/dizione_corpus.json: the sentences on which the JavaScript port must agree with Python dizione.

    python3 -I run/dizione_corpus.py --tests DIZIONE/tests --books GUTENBERG_DIR [--out run/dizione_corpus.json]

Five parts, in this order, each labelled in run/dizione_corpus_parts.json:
  tests     every TEXT line of dizione's tests/data/tricky_sentences.txt and the sentence-like strings of
            tests/test_dizione.py (Apache-2.0, dizione)
  prose     sentences from public-domain Project Gutenberg books: English novels, essays and science, plus a
            French, German and Italian book (accents, words the dictionary lacks)
  made      sentences written here for each thing the normaliser rewrites: numbers, dates, times, money, units,
            abbreviations, chat forms, names, homographs, unknown words, apostrophes, accents, addresses, codes,
            signs, inline pronunciations, scripts with nothing to pronounce
  targeted  one or more sentences for each branch of the rules that the parts above left unexercised (found with
            V8 coverage of web/dizione.js)
  mixed     the made fragments recombined at random, with random separators: odd strings for the regex rules
  chars     strings of random characters drawn from digits, signs, letters, marks and other scripts
  words     words the dictionary lacks, alone and in a frame, for the unknown-word model
  edge      empty and whitespace-only text, private-use marks the normaliser uses internally, very long input
"""

import argparse
import ast
import json
from pathlib import Path
import random
import re

SEED = 20261117


def test_sentences(tests):
    out = []
    for line in (tests / "data" / "tricky_sentences.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("TEXT:"):
            out.append(line[5:].strip())
    tree = ast.parse((tests / "test_dizione.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if 2 <= len(value) <= 300 and "\n" not in value and re.search(r"[A-Za-z0-9]", value) and not value.startswith("test_"):
                out.append(value)
    return list(dict.fromkeys(out))


def book_sentences(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF", text)
    text = text[start.end() if start else 0:end.start() if end else len(text)]
    text = re.sub(r"\s+", " ", text)
    pieces = re.split(r"(?<=[.!?])\s+(?=[\"'“‘(]?[A-ZÀ-Ý])", text)
    return [p.strip() for p in pieces if 3 <= len(p.split()) <= 45]


def prose(books, rng):
    english = ["1342", "1661", "2701", "345", "120", "1228", "1404", "5200", "11", "74", "76", "84", "98"]
    other = ["17489", "2229", "1012"]          # French, German, Italian
    out = []
    for name, count in [(n, 140) for n in english] + [(n, 70) for n in other]:
        path = books / f"{name}.txt"
        if not path.exists():
            continue
        sentences = book_sentences(path)
        out += rng.sample(sentences, min(count, len(sentences)))
    return out


# ---- sentences written for the rules -------------------------------------------------------------------

NAMES = ["Samir", "Kayleigh", "Ryleigh", "Okafor", "Nguyen", "Siobhan", "Zbigniew", "Xiomara", "Aoife", "Saoirse", "Joaquin",
         "Nnamdi", "Oluwaseun", "Thibault", "Krzysztof", "Yevgeny", "Dmitri", "Priyanka", "Chiamaka", "Tadhg", "Niamh",
         "Ngozi", "Mbappe", "Szczepanski", "Pfeiffer", "Ljubljana", "Reykjavik", "Kyiv", "Tlaquepaque", "Llewelyn",
         "Ffion", "Gwyneth", "Eilidh", "Oisin", "Bartholomew", "Xochitl", "Quetzalcoatl", "Zhaoyang", "Xinyi", "Jaehyun"]
MADE_UP = ["glorptastic", "frumious", "bandersnatch", "zibberflop", "snarfle", "blibber", "quaxly", "vrumple", "throckmorton",
           "skreeb", "plonkish", "gribbet", "fizzwhistle", "mumblecore", "doomscrolling", "rizzler", "yeetable", "cromulent",
           "embiggen", "flibbertigibbet", "shmorgle", "twerple", "zorblatt", "kerfuffling", "snollygoster", "bumfuzzle",
           "lollygagging", "cattywampus", "gobbledygook", "taradiddle", "wabbit", "squonk", "hornswoggle", "brouhahaing",
           "unfriendable", "rewildering", "overgoogled", "nonbinaryness", "coronaviruses", "rhinovirus", "Brayleigh"]
HOMOGRAPHS = [
    "I read the book yesterday.", "I will read the book tomorrow.", "She has read it twice.", "Read it aloud, please.",
    "It is a well-read copy.", "They watched a live show.", "They live in Boston.", "Live music tonight!",
    "The wind is strong.", "Wind the clock before bed.", "The lead pipe was old.", "Lead the way, captain.",
    "He caught a bass in the lake.", "She plays the bass guitar.", "The bow of the ship.", "Take a bow.",
    "She tied a bow in her hair.", "It was a minute amount.", "Wait a minute.", "I am content with the content.",
    "Please record the record.", "We present the present.", "I object to that object.", "A tear rolled down.",
    "Don't tear the paper.", "The wound healed.", "He wound the rope around the post.", "Close the door.",
    "Stay close to me.", "The desert is dry.", "Don't desert me.", "I refuse to take the refuse out.",
    "You need a permit to permit that.", "Fresh produce is cheap.", "They produce cars.", "The project is late.",
    "Project the image.", "Change the subject.", "Separate the eggs.", "The separate rooms.", "He estimates the estimates.",
    "Her records are kept.", "He records everything.", "The convicts escaped.", "She contracts the muscle.",
    "The dove flew away.", "He dove into the pool.", "Does the doe see the does?", "The polish on Polish furniture.",
    "Use the number to number them.", "He moped on the moped.", "The learned professor learned it.",
    "A blessed day, she blessed them.", "Increase the increase.", "They rebel; the rebel army.",
]
FRAGMENTS = {
    "numbers": ["3", "42", "1,234,567", "3.14159", "007", "0.5", ".75", "-5", "1st", "2nd", "3rd", "21st", "112th", "100",
                "1000000", "12,5", "1.299,50", "10^6", "2^10", "1e10", "999999999999999999999", "12345678901234567890123",
                "0", "00", "10", "13", "1100", "1999", "2000", "2005", "2026", "4070", "9001", "720", "1.0", "1,000.00"],
    "dates": ["3/14/2015", "2024-11-15", "15.11.2026", "March 3rd, 2021", "3 March 2021", "Jan. 5", "Mon.-Fri.", "the 1990s",
              "'90s", "in 1984", "2019-2021", "08/27", "12/10", "14/03", "on 5/6", "Nov 3-17", "3-17 November", "22 February",
              "5 Jan. 2026", "Sept. 9", "Tues-Thurs", "Jan-Mar", "1/2/03", "2026/01/02", "31.12.1999", "July 4, 1776"],
    "times": ["at 7:30 pm", "9.30 to 11.30", "10am", "12:00", "00:00", "23:59:59", "2:03:45", "48:12 long", "7 a.m.",
              "6 P.M. EST", "11:15 PM PST", "5:00-6:00", "7.30pm", "3pm-5pm", "at 12:05:30", "opens 9.00-17.30", "1:25"],
    "money": ["$5", "$3.50", "£20", "€1.299,50", "20 €", "750 000 $", "$2.5 million", "$10k", "USD 100", "100 USD", "¥500",
              "₹1,000", "5¢", "-$5", "$-5", "€12,50", "0,99 €", "CHF 20", "$1.01", "$0.99", "$100.005", "5-10 €", "$3bn",
              "AUD 12.75", "£1", "€0.01", "$1,000,000"],
    "units": ["5 km", "3.5kg", "10 mph", "100°F", "20 °C", "5 m²", "3 m3", "1,5 kg", "5-10 MB", "2 TB", "60 fps", "120 Hz",
              "5 ft", "3 x 4 m", "500 mg", "2 L", "40 W", "5 M", "5G", "12 V", "3 ha", "1 m", "2 mL", "250 ml", "8 GB",
              "5 lbs", "6 oz", "1.5 hrs", "30 mins", "5 m/s", "100 km/h", "3 kWh", "10 μm", "5°", "45 rpm", "3 tbsp"],
    "abbreviations": ["Mr. Smith", "Mrs. Jones", "Dr. Who", "St. Louis", "Main St.", "etc.", "e.g.", "i.e.", "U.S.A.",
                      "vs.", "No. 5", "Inc.", "Jr.", "Prof. Xavier", "Gen. Lee", "Mt. Everest", "approx.", "Dept.",
                      "Ave.", "Elm Dr.", "est. 1885", "Est. arrival", "max. load", "min.", "Jan.", "he sat.", "ref. no. 7"],
    "chat": ["lol", "omg", "brb", "idk", "k", "k, thanks", "u r gr8", "b4", "2day", "thx", "np", "imo", "tbh", "w/o", "w/",
             "n/a", "sooo goood", "heyyy", "Sss", "zzz", "hmm", "tsk", "Sh! Ss!", "mmm", "alll", "offf", "OMG", "LOL", "smh",
             "ttyl", "gtg", "g2g", "afaik", "dms", "ok k", "see you, k?", "i, j, k"],
    "apostrophes": ["don't", "it's", "'em", "'tis", "rock 'n' roll", "the dogs' bones", "James's", "o'clock", "ma'am",
                    "y'all", "'cause", "'twas", "he'd've", "ne'er", "'Hello'", "rock'n'roll", "’Tis", "it’s", "`quoted'"],
    "accents": ["café", "naïve", "résumé", "exposé", "jalapeño", "Zoë", "Ångström", "Dvořák", "Œuvre", "straße",
                "smørrebrød", "Łódź", "crème brûlée", "Résumés", "rosé", "pâté", "saké", "façade", "Mötley Crüe", "Beyoncé",
                "São Paulo", "Ñandú", "fiancée", "déjà vu", "Ærø"],
    "foreign": ["你好世界", "Привет мир", "Γειά σου κόσμε", "مرحبا بالعالم", "こんにちは", "안녕하세요", "नमस्ते", "שלום",
                "สวัสดี", "😀👍", "🇫🇷", "❤️", "…", "!!!", "—", "« »", "¿?", "¡!", "٣٤", "١٢٣", "𝐀𝐁𝐂", "ℌ𝔢𝔩𝔩𝔬", "Ⅻ",
                "①②", "ﬁne", "Ｆｕｌｌ", "x²", "H₂O", "ǅ", "İstanbul", "ıi", "ß", "ſ", "K", "µm", "Ω", "Å"],
    "addresses": ["https://example.com/path?x=1", "john.doe@example.org", "report_final.pdf", "/usr/local/bin",
                  "www.google.com", "config.yaml", "index.html", "github.com/n4ze3m/dizione", "file.TXT", "notes.md",
                  "photo.JPEG", "a.b.c", "the end.Then", "README.md", "my_file.v2.tar.gz", "ftp.site.net", "data.csv"],
    "codes": ["AB-4471-Z", "#A7-4419", "COVID-19", "F-16s", "B-12", "KX-300", "SAVE-20", "Wi-Fi", "3-in-1", "PS5", "B12",
              "X7K", "RTX 4070 Ti", "ISO 9001", "1080p", "4K", "v2.3.1", "Python 3.12", "iOS 17.4", "Order #A7", "#123",
              "#hashtag", "5-year", "3-D", "20-year-old", "50s-60s", "pre-2020", "91-220-0017", "10-15", "9-1-1", "XL-XXL",
              "SW1A 1AA", "GB 123 456 789", "1Z 999 AA1 01 2345 6784", "4111 1111 1111 1111", "1 000 000", "v10.15"],
    "signs": ["67 > 66", "b>a", "if x < 5", "a=10", "x = a + b", "f(x) >= 0", "p<.05", "x^2", "k = k + 1", "(a)",
              "<b>bold</b>", "Vec<u64>", "$x = 2$", "3 × 4", "10-3=7", "1/2", "3/4", "2 1/2", "½", "1½", "24/7", "50/50",
              "a & b", "100%", "~5", "±3", "x ≥ 2", "a != b", "n ≤ 10", "→", "5 + 3 = 8", "for n > 4", "any q > 0",
              "<your name>", "map<int, int>", "File>Open", "a) the first", "f(a, b)", "w/a friend", "π", "√2", "∞"],
    "inline": ["Say {HH AH0 L OW1} to Samir{S AH0 M IH1 R}.", "{h ə l ˈoʊ}", "{name}", "{1, 2, 3}", "x_{i}", "\\frac{1}{2}",
               "{ return 1; }", "{XX YY}", "Kubernetes{K UW2 B ER0 N EH1 T IY0 Z}", "{n}", "{t ˈɑː k}", "{AH0 B}x"],
    "phones": ["(555) 123-4567", "+1 555 123 4567", "+44 20 7946 0958", "1-800-555-0199", "call 911", "555-0199",
               "dial 0800 123 456", "PIN 4821", "code 0042", "account number 12345678", "order 98765", "ticket 4521"],
    "roman": ["World War II", "Henry VIII", "Chapter IV", "Part III", "Louis XIV", "Super Bowl LVIII", "Type II", "Elizabeth II"],
    "capitals": ["NASA", "FBI", "THIS IS ALL CAPS TEXT", "US", "IT department", "iPhone", "YouTube", "JavaScript", "macOS",
                 "eSIM", "iOS", "NATO", "UNESCO", "the USA's", "PDFs", "HTTP", "an FAQ", "AI", "LED"],
}


def made(rng):
    out = list(HOMOGRAPHS)
    frames = ["{}", "It was {}.", "Then {} happened, right?", "\"{}\"", "({})", "She said {} and left.", "{}!", "{}?",
              "We saw {} on Monday.", "Note: {}; see below.", "{} — really.", "Is it {} or not?"]
    for name, items in FRAGMENTS.items():
        for item in items:
            out.append(item)
            for frame in rng.sample(frames, 2):
                out.append(frame.format(item))
    for name in NAMES + MADE_UP:
        out += [name, f"{name} is here.", f"Have you met {name}?", f"{name}'s house"]
    # each number in common shapes
    for _ in range(150):
        n = rng.choice([rng.randint(0, 20), rng.randint(0, 100), rng.randint(100, 9999), rng.randint(10000, 10 ** 9)])
        out.append(rng.choice(["I counted {} sheep.", "Page {}", "There were {:,} people.", "It costs ${}.", "{} km",
                               "Room {}", "in {}", "{}%", "#{}", "{}th place", "{}:{:02d}", "{}/{}", "{}-{}"])
                   .format(n, rng.randint(0, 59), rng.randint(1, 31)))
    # several sentences together: the date and figure passes look at the whole text
    paragraphs = [
        "Call 555-0199 before 5 pm. The meeting is on 12/03 at 10:30, not 03/12. Bring $20.",
        "On 31/12 we close. On 1/2 we reopen. Prices rise 5% to €1.299,50.",
        "Mr. Smith lives on Main St. He has 3 cats, 2 dogs and 1 fish. It's 4:30 p.m. Time to go!",
        "Version 2.10 fixes bug #4471. Update to iOS 17.4 or macOS 14.2.1 first. See https://example.com/notes.",
        "I read that you read it. Read it again! The lead singer will lead.",
        "The ratio is 16:9; the score was 3:1 at 9:45. The odds are 1:25,000.",
        "From 1990 to 2005, sales grew from 1,000 to 25,000 units (a 2,400% rise).",
    ]
    out += paragraphs
    return out


TARGETED = [
    # numbers, money and times at the edges of their rules
    "It cost $1234567890123456789012345.", "The total was 12345678901234567890 dollars.", "$1,000,000,000,000,000,000,000",
    "The race took 1:01:01.", "The clip is 2:00:01 long.", "It lasted 0:00:00.", "The tape runs 48:05.", "Meet at 25:30 pm.",
    "At 25:30:15 it stopped.", "See you at 12:00 pm.", "Lunch is at 12:00 p.m. The rest follows.", "Start at 12:00:01.",
    "Doors at 7:05 a.m. Then we go.", "Leave at 6 p.m. We will be late.", "It is 00:30 now.", "The run was 61:02 long.",
    "It went on for 30:15:20 at most.", "duration 75:40", "lasted 99:59", "The 7:45 PM CET train.",
    "Pay 1¢ or 2¢.", "It's €-5 now.", "USD-20 fee", "A £1.5k bonus", "A ¥1,000 note", "A ₹2.50 stamp", "INR 99.99",
    "Add 2 1/8 cups.", "Add 1 1/16 inch.", "Add 2 5/4 cups.", "Add 3 3/3 cups.", "About 1/100 of it.", "1/64 inch",
    "Mix 2/7 and 5/9.", "It is 7/8 done.", "Ratio 125/250.", "0/5 stars", "A 3/200 chance", "The 33/45 split.",
    "Type enter 1234 now.", "Enter 2019 as the year.", "Enter 20190 please.", "Call us at 1234.", "PIN 0042", "code 7",
    "There were 1234 people.", "About 2020 children came.", "In 1066 they landed.", "Model 3000 parts", "RTX 3090 cards",
    "ISO 27001 and BS 7799.", "The SKU is AB 1234.", "Over 4500 feet up.", "It weighs 1250 lbs.", "I have 4096 books.",
    # dates
    "On 13/32/2020 nothing.", "Due 1/2/15 or 1/2/03.", "Born 25/12/2020.", "Due 05.25.2020.", "On 45 March.", "On 32 May.",
    "On 5 March. then we left.", "On 5 Sept. 2026 we met.", "On 5 Sep. we met.", "Nov 3-17, 2026", "3-17 November 2026",
    "Monday-Friday", "Thurs-Sat.", "Open Mon - Fri. The rest is closed.", "Jan-Mar. Then Apr-Jun.", "Tue.-Thu. We work.",
    "The card expires 08/27.", "Card valid until 12/29.", "On 12/25 and 25/12.", "Between 3/4 and 4/5.", "Dated 1/1.",
    "From 12/03 to 15/03.", "On 12/03/2025 and 15/03/2025.", "Due by 6/7.", "Ships on 1/2.", "Arrives 3/5 next week.",
    "2026-02-30", "2026-13-01", "2026/02/03", "1999-12-31T23:59", "31.12.99", "3.4.2021", "13.25.2021", "Sept 31st",
    "March 3rd-5th", "the 3rd-5th of March", "22nd February 2022", "1 Jan, 2000", "August 15th 1947", "July 4th",
    # codes, groups and versions
    "Code AB-12345 applies.", "Gate ALPHA-7-BRAVO opens.", "Ref XYZ-12-STATION.", "Part KX-3000 and KX-3001.",
    "Order #ABCD-1", "Use X-1 or Y-22.", "IN 2019 150 units were sold.", "ON 2019 12 34 off.", "Ship 1 000 000 units.",
    "We sold 2 500 000 cars.", "1 234 567 people", "Card 4111 1111 1111 1111 expires.", "IBAN GB29 NWBK 6016 1331 9268 19",
    "Tracking 1Z 999 AA1 01 2345 6784 is late.", "VAT GB 123 456 789 AB", "+33 1 23 45 67 89", "Call 0800 123 4567 now.",
    "Phone: 020 7946 0958.", "Account 12 34 56", "Upgrade from 2.10 to 2.12 now.", "Python 3.10 or 3.11", "iOS 9.03",
    "Update 4.05 is out.", "v1.02", "V2.10.3-4 released", "version 10.15.7", "Kernel 6.1.0-13", "macOS 14.2.1",
    "It is 5x faster.", "Use 2x2 tiles.", "Zoom 10x now.", "4x4 truck", "3 x 4", "3x4 metres", "2 × 3 cm", "×5 zoom",
    "In 1080i or 720p.", "a 4K TV in 2160p", "The '80s and the 1990s.", "In the 60s.", "The 2000s", "her 20s",
    # abbreviations
    "Dr Smith is here.", "Dr. Brown, Elm Dr. and St. Mark's.", "He lives at 5 Main St.", "Visit St Paul.", "Go to Oak St. Then turn.",
    "The order no. is 5.", "The order no. 5.", "Invoice no. 12", "Room no. 4", "Ref. no. 12 and ref. No. 13", "Est. arrival",
    "max. load", "Max. speed", "min. 5", "est. 1885", "Est. 1885", "he sat. She stood.", "Sun. 5 pm", "Mar. 3", "Jan. Feb.",
    "Mr Smith", "MR. SMITH", "Prof. X", "Gen. Lee and gen. motors", "Capt. Hook", "Lt. Dan", "Sgt. Pepper", "Co. Ltd.",
    "approx. 5", "dept. head", "Inc. and Corp.", "No. 1", "no. 2", "NO. 3", "the U.S. and U.K. The end.", "e.g. this", "i.e. that",
    # chat, sounds and replies
    "k", "K.", "k, thanks", "\"k\"", "(k)", "see you there, k?", "the value of k", "i, j, k", "a, k, b", "k! = mk", "k: king",
    "Ss zzz", "Sh, ss, tsk.", "the ss", "mm:ss", "written as 'ss'", "Zzzz...", "Sss!", "Psst, ss.", "hmm, mm.",
    "dm5", "b4 2day", "gr8!", "l8r", "2moro", "2nite", "w/o sugar", "with w/ cream", "N/A", "n/a.", "xoxo", "lol!!!",
    "'blah", "'Cause I said so.", "'em all", "'tis true", "'bout time", "'round here", "rock'n'roll", "fish 'n' chips",
    # signs, comparisons, brackets
    "if a > b then", "when x >= 5", "the value a=10 holds", "b>a", "x<5", "p<.05", "File>Open", "M>Train", "x < y", "3 > 2",
    "f(x) > 0", "a[i] < b", "n ≥ 2", "so x ≠ y", "an integer k ≥ 2", "for n > 4", "thus 5 > 3", "a>=b", "c<=d", "e!=f",
    "<b>bold</b>", "<your name>", "Vec<u64>", "map<int, int>", "x->y", "a <- b", "<<>>", "<3 you", "you<3", "I <3 NY",
    "$x = 2$", "$3 + 5 = 8$", "from $5 to 30$", "5 $, 10 $", "costs 750 000 $ total", "10-3=7", "2024-11-15=", "5 - 3 = 2",
    "a+b", "1/a", "w/a friend", "(a)", "f(a, b)", "f(b, a)", "a) first b) second", "x=a", "c = a + b", "mode = a set",
    "#1 fan", "#hashtag", "Order #A7", "~5 km", "~ 10", "x^2 + y^2", "2^10", "e^x", "100%", "50 %", "3,5%", "€12,50 + 0,99 €",
    # words built from parts, names, homographs with hyphens
    "Kayleigh and Brayleighs", "coronaviruses", "the rhinovirus's spread", "noroviruses", "Tung Tung Tung Sahur!",
    "tung tung tung sahur", "Tung-tung tung sahur", "so much so far", "too much, no longer", "the well-read man", "a widely read book",
    "half-read", "un-read", "I've read it", "having read", "to read", "can't read", "let's read", "Read!", "live-streaming",
    "minute changes", "a minute.", "content with it", "very content", "the content.", "she's content", "bow-tie", "lead-free",
    "xkcd", "brrr", "pfft", "tsktsk", "hmmmm", "zzxq", "bcdfg", "qwrtp", "mmm", "shhh", "nth", "rhythms", "crwth", "cwm",
    # capitals, initialisms, Roman numerals
    "THE QUICK BROWN FOX JUMPS", "I LOVE NY", "IT IS A TEST", "Use AI and IT", "the OS and PE", "WHO is WHO", "OK GO",
    "macOS and iPadOS", "eSIMs", "iPhone's", "McDonald's", "O'Neill", "DeShawn", "LaToya", "MacBook", "PowerPoint", "USBs",
    "Henry VIII's wife", "World War II.", "Chapter XX", "Act IV, Scene II", "King Louis XVIII", "type VI", "Rocky IV",
    # accents and marks the plain-letter step decides
    "RÉSUMÉ", "Résumés", "SAKÉ", "rosés", "Exposé", "the résumé's", "naïveté", "Ångström", "Œdipus", "Æsir", "Straße",
    "İzmir", "Iğdır", "dotless ı", "Đorđe", "Łukasz", "Øresund", "Ǆemal", "ﬀ", "ﬃx", "Ⅷ", "½ cup", "1½ cups", "⅓", "2⅔",
    "x³", "m²", "5 m² and 3 m³", "10 in²", "3 km³", "−5 °C", "1⁄2", "２０２６年", "٣ apples", "the ٣٤th", "۱۲۳", "१२३",
    "፩ ፪", "❶ ❷", "①", "𝟙𝟚𝟛", "Ⓐ", "™", "℃", "№ 5", "…and then", "a—b", "a – b", "a ― b", "«quote»", "„quote“",
    # addresses
    "Mail me@example.com.", "Go to www.example.co.uk/a/b?c=d#e.", "See http://x.io:8080/path.", "Open C:/files/data.csv",
    "Read the_file.TXT", "notes.Md", "archive.tar.gz", "Run ./build.sh", "docs/index.HTML", "a.b@c.de", "x@y.zz",
    "The end.Then we left.", "Version.2 is out", "ok.google", "file_name.jpeg", "IMG_2024.PNG", "my-site.dev/blog",
]


def chars(rng, count):
    alphabet = (list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") * 3 + list("0123456789") * 4 +
                list(" .,;:!?'\"()[]{}<>/\\-_=+*^%$£€¥₹¢#@&~`|") * 2 +
                list("–—―‘’“”«»…×÷±°²³½¼¾⅓→−≈≠≤≥∞√πéèêëàâäçñöüøåßæœłđ") +
                list("你好Привет Γειάمرحباこんにちは안녕नमस्ते٣٤١۱१፩❶①ﬁＦ😀👍🇫") +
                ["\n", "\t", "\u00a0", "\u200b", "\u2028", "\x1c", "\x85", "\u0301", "\u0308"])
    out = []
    for _ in range(count):
        out.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 30))))
    return out


def mixed(rng, count):
    pool = [item for items in FRAGMENTS.values() for item in items] + NAMES + MADE_UP + \
        [w for s in HOMOGRAPHS for w in s.split()]
    separators = [" ", " ", " ", "", ", ", ". ", "-", "/", ":", "; ", " — ", "'", "(", ")", " & ", "\n", "\t"]
    out = []
    for _ in range(count):
        parts = [rng.choice(pool) for _ in range(rng.randint(2, 6))]
        text = parts[0]
        for part in parts[1:]:
            text += rng.choice(separators) + part
        out.append(text)
    return out


def unknown_words(lexicon, books, rng, count):
    known = set()
    if lexicon and lexicon.exists():
        known = {line.split("\t", 1)[0] for line in lexicon.read_text(encoding="utf-8").splitlines() if line}
    seen = set()
    for path in sorted(books.glob("*.txt")):
        for word in re.findall(r"[A-Za-z]{4,}", path.read_text(encoding="utf-8", errors="replace")):
            low = word.lower()
            if low not in known:
                seen.add(low)
    words = sorted(seen)
    picked = rng.sample(words, min(count, len(words)))
    out = []
    for word in picked:
        out.append(rng.choice(["{}", "The {} is here.", "{}s", "un{}", "{}ing", "{} and {}"]).format(word, word))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tests", type=Path, required=True, help="dizione's tests/ directory")
    parser.add_argument("--books", type=Path, required=True, help="Gutenberg plain-text books, named <number>.txt")
    parser.add_argument("--lexicon", type=Path, help="lexicon.tsv, to pick words it lacks")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "dizione_corpus.json")
    args = parser.parse_args()
    rng = random.Random(SEED)
    parts = {
        "tests": test_sentences(args.tests),
        "prose": prose(args.books, rng),
        "made": made(rng),
        "targeted": TARGETED,
        "mixed": mixed(rng, 4000),
        "chars": chars(rng, 2000),
        "words": unknown_words(args.lexicon, args.books, rng, 1200),
    }
    parts["edge"] = ["", " ", "\t\n", "   　", ".", "...", "!?", "-", "--", "'", "\"\"", "()", "a", "I", "A.", "x", "0",
                     " ", "​", "﻿", "\x1c", " ", "\U000e0001", "AB", "",
                     "a" * 300, "1" * 40, "word " * 200]
    corpus, labels = [], []
    for name, items in parts.items():
        for item in items:
            corpus.append(item)
            labels.append(name)
    args.out.write_text(json.dumps(corpus, ensure_ascii=False, indent=0), encoding="utf-8")
    (args.out.parent / "dizione_corpus_parts.json").write_text(json.dumps(labels), encoding="utf-8")
    print({name: len(items) for name, items in parts.items()}, "total", len(corpus))


if __name__ == "__main__":
    main()
