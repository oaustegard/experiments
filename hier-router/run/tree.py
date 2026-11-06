"""The fixture: 69 leaf specialists in a tree of variable depth, their cases, and induced subtrees.

Leaves come from three sources, each with a natural grouping:
  - Amazon MASSIVE en-US (CC BY 4.0): one leaf per intent, grouped by scenario; the iot scenario is
    split once more into lights and plugs, so that branch is one level deeper. Six intents with fewer
    than 10 test utterances are dropped; cooking then has one intent, so cooking_recipe hangs directly
    under the voice router, as do news_query and weather_query (single-intent scenarios).
  - Typed Decisions (adapter-library round 4's 12 families): one leaf per family, grouped by workflow.
  - adapter-library's everyday set: transactions, emails and reviews, one leaf each, directly under
    the root. The root therefore offers two routers and three leaves.

A router node's options are its children's descriptions; a leaf's description is what a router reads
to pick it. Seven leaves are held out of every router's training (HELD_OUT) to test unseen leaves.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from _lib.paths import experiment  # noqa: E402

AL = experiment("adapter-library")
sys.path.insert(0, str(AL / "run"))
from shared import FAM_DESC, FAM_IDS, FAM_WORKFLOW, family_map, load_cases  # noqa: E402

MASSIVE_DIR = Path(__import__("os").environ.get("MASSIVE_DIR", ROOT / "data" / "massive"))
SCENARIOS = ['social', 'transport', 'calendar', 'play', 'news', 'datetime', 'recommendation', 'email', 'iot', 'general', 'audio',
             'lists', 'qa', 'cooking', 'takeaway', 'music', 'alarm', 'weather']
INTENTS = ['datetime_query', 'iot_hue_lightchange', 'transport_ticket', 'takeaway_query', 'qa_stock', 'general_greet', 'recommendation_events',
           'music_dislikeness', 'iot_wemo_off', 'cooking_recipe', 'qa_currency', 'transport_traffic', 'general_quirky', 'weather_query',
           'audio_volume_up', 'email_addcontact', 'takeaway_order', 'email_querycontact', 'iot_hue_lightup', 'recommendation_locations',
           'play_audiobook', 'lists_createoradd', 'news_query', 'alarm_query', 'iot_wemo_on', 'general_joke', 'qa_definition', 'social_query',
           'music_settings', 'audio_volume_other', 'calendar_remove', 'iot_hue_lightdim', 'calendar_query', 'email_sendemail', 'iot_cleaning',
           'audio_volume_down', 'play_radio', 'cooking_query', 'datetime_convert', 'qa_maths', 'iot_hue_lightoff', 'iot_hue_lighton',
           'transport_query', 'music_likeness', 'email_query', 'play_music', 'audio_volume_mute', 'social_post', 'alarm_set', 'qa_factoid',
           'calendar_set', 'play_game', 'alarm_remove', 'lists_remove', 'transport_taxi', 'recommendation_movies', 'iot_coffee', 'music_query',
           'play_podcasts', 'lists_query']
DROPPED = {"general_greet", "iot_hue_lighton", "cooking_query", "music_dislikeness", "music_settings", "audio_volume_other"}

INTENT_DESC = {
    "social_query": "Checking social media: what friends posted or replies to a post.",
    "social_post": "Posting to social media or complaining to a company in public.",
    "transport_ticket": "Booking a train or plane ticket.",
    "transport_traffic": "Traffic on the roads right now.",
    "transport_query": "Train times, routes, directions or travel questions.",
    "transport_taxi": "Ordering a taxi or a ride.",
    "calendar_remove": "Deleting or cancelling a calendar event or reminder.",
    "calendar_query": "Asking what is on the calendar.",
    "calendar_set": "Adding a meeting, event or reminder to the calendar.",
    "play_audiobook": "Playing or resuming an audiobook.",
    "play_radio": "Playing a radio station.",
    "play_music": "Playing a song, artist, album or playlist.",
    "play_game": "Starting a game.",
    "play_podcasts": "Playing a podcast or its next episode.",
    "news_query": "News headlines and stories.",
    "datetime_query": "The current date, time or day of the week somewhere.",
    "datetime_convert": "Converting a time between time zones.",
    "recommendation_events": "Suggesting events or things to do nearby.",
    "recommendation_locations": "Suggesting a restaurant, shop or place to go.",
    "recommendation_movies": "Suggesting a movie to watch.",
    "email_addcontact": "Adding a contact or an email address to the address book.",
    "email_querycontact": "Looking up a contact's details.",
    "email_sendemail": "Writing or sending an email.",
    "email_query": "Checking for new email or reading messages.",
    "iot_hue_lightchange": "Changing the light colour or setting a lighting scene.",
    "iot_wemo_off": "Turning a smart plug or socket off.",
    "iot_hue_lightup": "Making the lights brighter.",
    "iot_wemo_on": "Turning a smart plug or socket on.",
    "iot_hue_lightdim": "Dimming the lights.",
    "iot_cleaning": "Starting the robot vacuum cleaner.",
    "iot_hue_lightoff": "Turning the lights off.",
    "iot_coffee": "Making coffee with the coffee machine.",
    "general_quirky": "Odd remarks, chit-chat and questions that fit nowhere else.",
    "general_joke": "Asking for a joke.",
    "audio_volume_up": "Turning the volume up.",
    "audio_volume_down": "Turning the volume down.",
    "audio_volume_mute": "Muting the sound or asking for quiet.",
    "lists_createoradd": "Making a list or adding an item to one.",
    "lists_remove": "Removing an item from a list or deleting a list.",
    "lists_query": "Reading out what is on a list.",
    "qa_stock": "A stock price or how a company's shares are doing.",
    "qa_currency": "Exchange rates and converting between currencies.",
    "qa_definition": "What a word means or how to spell it.",
    "qa_maths": "A sum or a maths question.",
    "qa_factoid": "A general knowledge question about a fact.",
    "cooking_recipe": "A recipe or how to cook something.",
    "takeaway_query": "Asking about a takeaway order or which places deliver.",
    "takeaway_order": "Ordering takeaway food.",
    "music_likeness": "Saying which music the person likes, to remember it.",
    "music_query": "Asking what song or artist is playing.",
    "alarm_query": "Asking which alarms are set.",
    "alarm_set": "Setting an alarm or a wake-up call.",
    "alarm_remove": "Removing or cancelling an alarm.",
    "weather_query": "The weather or the forecast.",
}
SCEN_DESC = {
    "social": "Social media: posting or checking updates.",
    "transport": "Travel: tickets, taxis, traffic and transport questions.",
    "calendar": "Calendar events, meetings and reminders.",
    "play": "Starting something to play: music, radio, a podcast, an audiobook or a game.",
    "datetime": "The current date or time, or converting between time zones.",
    "recommendation": "Suggestions for places, events or movies.",
    "email": "Email and contacts: reading, sending, adding or looking up a contact.",
    "iot": "Smart-home devices: lights, plugs, the robot vacuum or the coffee machine.",
    "iot_lights": "The smart lights: on, off, brighter, dimmer or a different colour.",
    "iot_plugs": "Smart plugs and sockets: turning one on or off.",
    "general": "Small talk: jokes and odd remarks.",
    "audio": "The device's sound: volume up, down or mute.",
    "lists": "Shopping and to-do lists: making, adding to, reading or removing items.",
    "qa": "General knowledge: facts, definitions, maths, stock prices or currency.",
    "takeaway": "Takeaway food: ordering or asking about an order.",
    "music": "Music preferences and questions about what is playing.",
    "alarm": "Alarms: setting, checking or removing one.",
}
WF_DESC = {
    "customer_service": "Customer support conversation: a ticket thread with the customer's account and orders.",
    "invoice_processing": "Vendor invoice review: an invoice with its purchase order, delivery record and payment terms.",
    "security_incidents": "Security alert triage: an alert with its evidence, the principal involved and context.",
    "agent_trace_observability": "AI agent run trace: an autonomous agent's task, constraints and step summary.",
}
EVERYDAY_DESC = {
    "transactions": "A bank or card transaction with the history of charges from the same merchant.",
    "emails": "An email message in someone's inbox: sender, subject and body.",
    "reviews": "A customer review of a restaurant or business.",
}
TOP_DESC = {
    "voice": "A short spoken command or question to a voice assistant, as plain text.",
    "business": "A JSON record from a business workflow: a support ticket, a vendor invoice, a security alert or an AI agent trace.",
}
HELD_OUT = ["iot_hue_lightdim", "calendar_remove", "qa_currency", "play_podcasts", "transport_taxi", "alarm_remove", "sec_privilege"]
NONE_DESC = "None of these fits this input."
QUESTION = "choice: Which of these should handle this input? Pick the one whose description fits it."


def build_tree():
    """parent -> list of children (node names). Leaves have no entry. Also returns desc and parent maps."""
    kids, desc = {"root": ["voice", "business", "transactions", "emails", "reviews"]}, {}
    desc.update(TOP_DESC)
    desc.update(EVERYDAY_DESC)
    by_scen = {}
    for it in INTENTS:
        if it in DROPPED:
            continue
        by_scen.setdefault(it.split("_")[0], []).append(it)
    kids["voice"] = []
    for sc in SCENARIOS:
        its = by_scen.get(sc, [])
        if len(its) == 1:
            kids["voice"].append(its[0])
        elif sc == "iot":
            kids["voice"].append("iot")
            kids["iot"] = ["iot_lights", "iot_plugs", "iot_cleaning", "iot_coffee"]
            kids["iot_lights"] = [i for i in its if i.startswith("iot_hue")]
            kids["iot_plugs"] = [i for i in its if i.startswith("iot_wemo")]
            desc.update({k: SCEN_DESC[k] for k in ("iot", "iot_lights", "iot_plugs")})
        else:
            kids["voice"].append(sc)
            kids[sc] = its
            desc[sc] = SCEN_DESC[sc]
    desc.update(INTENT_DESC)
    kids["business"] = list(WF_DESC)
    desc.update(WF_DESC)
    for w in WF_DESC:
        kids[w] = [f for f in FAM_IDS if FAM_WORKFLOW[f] == w]
    desc.update(FAM_DESC)
    parent = {c: p for p, cs in kids.items() for c in cs}
    return kids, desc, parent


KIDS, DESC, PARENT = build_tree()
LEAVES = [n for n in PARENT if n not in KIDS]
TRAINED = [x for x in LEAVES if x not in HELD_OUT]


def path_of(leaf):
    """root ... leaf, as a list of nodes."""
    p = [leaf]
    while p[-1] != "root":
        p.append(PARENT[p[-1]])
    return p[::-1]


def induced(leaves):
    """The tree restricted to `leaves`: empty routers removed, a router left with one child replaced by
    that child. Returns parent -> children over the surviving nodes (root always survives)."""
    keep = set(leaves)

    def walk(n):
        if n not in KIDS:
            return n if n in keep else None
        cs = [c for c in (walk(k) for k in KIDS[n]) if c is not None]
        if not cs:
            return None
        if len(cs) == 1 and n != "root":
            return cs[0]
        sub[n] = cs
        return n

    sub = {}
    walk("root")
    return sub


def leaf_cases(cap_train=150, cap_test=30):
    """leaf -> {"train": [...], "test": [...]} with state text; seeded caps per leaf."""
    import pandas as pd
    out = {x: {"train": [], "test": []} for x in LEAVES}
    for split, fn in (("train", "train"), ("test", "test")):
        d = pd.read_parquet(MASSIVE_DIR / f"{fn}.parquet")
        for r in d.itertuples():
            it = INTENTS[r.intent]
            if it in out:
                out[it][split].append({"id": f"massive-{split}-{r.id}", "state": r.utt, "leaf": it})
    fm = family_map()
    for w in WF_DESC:
        for split in ("train", "test"):
            for c in load_cases(w, split):
                out[fm[c["id"]]][split].append({"id": c["id"], "state": c["state"], "leaf": fm[c["id"]]})
    for line in open(AL / "data" / "everyday" / "cases.jsonl"):
        c = json.loads(line)
        out[c["workflow"]][c["split"]].append({"id": c["id"], "state": c["state"], "leaf": c["workflow"]})
    for x, d in out.items():
        for split, cap in (("train", cap_train), ("test", cap_test)):
            d[split].sort(key=lambda c: c["id"])
            random.Random(f"{x}-{split}").shuffle(d[split])
            d[split] = d[split][:cap]
    return out


def router_example(node, options, state, gold, case=""):
    """One routing question at `node`: options are child names plus `none`; gold is a name or None."""
    names = list(options) + ["none"]
    opts = [(o, DESC[o]) for o in options] + [("none", NONE_DESC)]
    g = [1.0 if n == gold else 0.0 for n in names] if gold is not None else [1.0 / len(names)] * len(names)
    return {"case": case, "node": node, "names": names, "type": "choice", "question": QUESTION, "options": opts,
            "state": state, "gold": g}


if __name__ == "__main__":
    depth = {x: len(path_of(x)) - 1 for x in LEAVES}
    from collections import Counter
    print(len(LEAVES), "leaves,", len(TRAINED), "trained,", len(KIDS), "routers; depth", dict(Counter(depth.values())))
    print("root:", KIDS["root"])
    print("voice:", KIDS["voice"])
    sub = induced(random.Random(0).sample(TRAINED, 8))
    print("induced 8:", sub)
