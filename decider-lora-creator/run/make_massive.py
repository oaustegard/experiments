"""The page's main demo: voice-assistant commands from Amazon MASSIVE (en-US, CC BY 4.0).

    python3 -I run/make_massive.py PARQUET_DIR      # train.parquet and test.parquet from the HF parquet export

AmazonScience/massive (FitzGerald et al. 2022, arXiv 2204.08582) labels each utterance with a scenario
(18) and an intent (60), plus slot annotations. No library spike covers spoken commands, so this demo
shows a new decision rather than a held-out library workflow. Three questions, all from MASSIVE's own
labels:
  domain         choice, the 18 scenarios
  request        choice: act / ask / chat, a fixed mapping of the 60 intents (REQUEST below)
  mentions_time  noul: the utterance has a date, time or time-of-day slot in MASSIVE's annotation
Samples 1,200 train and 600 test utterances (seeded), the test ones marked "split": "test", and writes
web/demo/massive.json. The label ids follow the ClassLabel order in the dataset's massive.py.
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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
CHAT = {"general_greet", "general_quirky", "general_joke"}
ACT = {"iot_hue_lightchange", "transport_ticket", "iot_wemo_off", "audio_volume_up", "email_addcontact", "takeaway_order", "iot_hue_lightup",
       "play_audiobook", "lists_createoradd", "iot_wemo_on", "music_settings", "audio_volume_other", "calendar_remove", "iot_hue_lightdim",
       "email_sendemail", "iot_cleaning", "audio_volume_down", "play_radio", "iot_hue_lightoff", "iot_hue_lighton", "audio_volume_mute",
       "social_post", "alarm_set", "calendar_set", "play_game", "alarm_remove", "lists_remove", "transport_taxi", "iot_coffee", "play_music",
       "play_podcasts", "music_likeness", "music_dislikeness"}
REQUEST = {i: "chat" if i in CHAT else "act" if i in ACT else "ask" for i in INTENTS}
TIME_SLOT = re.compile(r"\[(date|time|timeofday) :")

DOMAIN = {
    "alarm": "Alarms: setting, checking or removing one.",
    "audio": "The device's sound: volume up, down or mute.",
    "calendar": "Calendar events, meetings and reminders.",
    "cooking": "Recipes and cooking questions.",
    "datetime": "The current date or time, time zones, or converting between them.",
    "email": "Email and contacts: reading, sending, adding or looking up a contact.",
    "general": "Small talk: greetings, jokes and odd remarks.",
    "iot": "Smart-home devices: lights, plugs, the robot vacuum or the coffee machine.",
    "lists": "Shopping and to-do lists: making, adding to, reading or removing items.",
    "music": "Music preferences and questions about what is playing.",
    "news": "News headlines and stories.",
    "play": "Starting something to play: music, radio, a podcast, an audiobook or a game.",
    "qa": "General knowledge: facts, definitions, maths, stock prices or currency.",
    "recommendation": "Suggestions for places, events or movies.",
    "social": "Social media: posting or checking updates.",
    "takeaway": "Takeaway food: ordering or asking about an order.",
    "transport": "Travel: tickets, taxis, traffic and transport questions.",
    "weather": "The weather or the forecast.",
}
DECISION = {
    "id": "voice_assistant_command",
    "description": "A spoken command to a voice assistant, as text. Decides what area it is about, whether the person wants something done, "
                   "wants information or is just chatting, and whether it names a date or time.",
    "questions": {
        "domain": {"type": "choice", "instructions": "Which area of the assistant does this command belong to?", "criteria": DOMAIN},
        "request": {"type": "choice", "instructions": "What does the person want from the assistant?", "criteria": {
            "act": "Something done or changed: set, add, remove, send, order, play, turn on or off, adjust.",
            "ask": "Information: a question, a lookup, a recommendation or a recipe.",
            "chat": "Neither: a greeting, a joke or small talk."}},
        "mentions_time": {"type": "noul", "instructions": "The command names a date, a time or a time of day.", "criteria": {
            "true": "A date, a clock time or a part of the day is named, such as tomorrow, at nine, tonight.",
            "false": "No date or time is named."}},
    },
}


def rows(path):
    import pandas as pd
    d = pd.read_parquet(path)
    out = []
    for r in d.itertuples():
        sc, it = SCENARIOS[r.scenario], INTENTS[r.intent]
        assert it.split("_")[0] == sc or (sc, it) in {("general", "general_quirky")} or it.startswith(sc), (sc, it)
        out.append({"id": f"massive-{r.partition}-{r.id}", "state": r.utt,
                    "gold": {"domain": sc, "request": REQUEST[it], "mentions_time": "true" if TIME_SLOT.search(r.annot_utt) else "false"},
                    "intent": it})
    return out


def main():
    src = Path(sys.argv[1])
    assert len(SCENARIOS) == 18 and len(INTENTS) == 60 and len(set(INTENTS)) == 60
    tr, te = rows(src / "train.parquet"), rows(src / "test.parquet")
    random.Random("massive-demo").shuffle(tr); random.Random("massive-demo-test").shuffle(te)
    cases = [{k: v for k, v in c.items() if k != "intent"} for c in tr[:1200]] + [{**{k: v for k, v in c.items() if k != "intent"}, "split": "test"} for c in te[:600]]
    out = {"decision": DECISION, "cases": cases, "init": [],
           "source": "Amazon MASSIVE en-US (CC BY 4.0, AmazonScience/massive, FitzGerald et al. 2022): 1,200 train and 600 test utterances, "
                     "labels from its scenario, intent (mapped to act/ask/chat) and date/time slots. decider-lora-creator/run/make_massive.py"}
    (ROOT / "web" / "demo" / "massive.json").write_text(json.dumps(out, separators=(",", ":")))
    from collections import Counter
    for q in DECISION["questions"]:
        print(q, dict(Counter(c["gold"][q] for c in cases if c.get("split") != "test").most_common()))
    print(len(cases), "cases", f"{(ROOT / 'web' / 'demo' / 'massive.json').stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
