import sys, pathlib
S = pathlib.Path(sys.argv[1]); task = S / "task2"
common = (task / "COMMON.md").read_text()
CONTRACT = """
## Contract first

Before writing any code, decide the exact interface your file PROVIDES to
teammates: each function's name, arguments with types, and return value shape
(field names and types). Write it as one compact line, for example
`parse(text:str) -> (lines:list[dict n:int order:str sku:str qty:int], rejected:list[tuple n:int reason:str])`.
Publish it as described below, then code against your teammates' published
contracts exactly. If a contract you need is missing or does not fit, ask its
owner before guessing. If you must change your own, republish it and tell the
teammates it affects in one line. dave owns the call order: if a pipeline
detail is unclear, dave decides.
"""
FINISH = """
## Finishing

When your file is done, tell everyone `done`. Then stay available: keep
answering questions until dave sends `SHIP`, {wait}. dave: send `SHIP` to all
once `process()` works end to end on an example you ran.
"""
CH = {
 "none": """
## Communication

There is no messaging channel. You can read your teammates' files in
`{T}/orders/` as they write them. Work out the interfaces from what you can read.
""",
 "native": """
## Communication

A message from the coordinator listing your teammates' addresses arrives
shortly after you start. Talk to a teammate with the SendMessage tool (load it
with ToolSearch `select:SendMessage`), `to` = that teammate's address.
Messages to you arrive automatically at your next tool call. Sending to a
teammate who has already finished wakes it up. Keep messages short and specific.
{contract_pub}""",
 "mail": """
## Communication

Messaging goes through `{T}/mail`:
- Your FIRST command: `{T}/mail join {name} "<card>"`. It prints the cards of teammates who joined so far.{card_rule}
- Send: `{T}/mail send {name} <to> "<text>"` where `<to>` is a teammate name, a comma list, or `all`.
- Mail to you appears automatically after any tool call, labelled `mail for {name}`, together with new or updated teammate cards. You never need to poll.
- To block until mail arrives: `{T}/mail recv {name} --wait 90`.
Keep messages short and specific.
""",
}
for arm in [f"{a}-{r}" for a in ["none", "native-plain", "mail-plain", "mail-contract"] for r in (1, 2, 3)]:
    T = pathlib.Path("/tmp/ac2") / arm
    for name in ["alice", "bob", "carol", "dave"]:
        kind = arm.split("-")[0]; contract = "contract" in arm
        ch = CH[kind].format(T=T, name=name,
            contract_pub="Publish your contract by sending it to all three teammates as your first message.\n" if contract else "",
            card_rule=" Your card is your contract line; republish by running join again with the new card." if contract else " Your card is one line: who you are and which file you own.")
        parts = [common.replace("a shared directory", f"the shared directory `{T}`"),
                 (task / f"{name}.md").read_text(), CONTRACT if contract else "", ch]
        if kind != "none":
            parts.append(FINISH.format(wait="re-running `" + f"{T}/mail recv {name} --wait 90" + "` until SHIP arrives or a wait returns `(no mail)` twice in a row" if kind == "mail"
                                       else "then finish; a teammate's message will wake you if needed"))
        parts.append(f"\nWork only inside `{T}`. Do not read any other directory except this brief.\n")
        (S / "briefs2" / f"{arm}-{name}.md").write_text(f"You are {name}.\n\n" + "\n".join(p for p in parts if p))
print("ok")
