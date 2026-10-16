import random, json
items=json.load(open("items.json"))
random.seed(2509)
words = ["apple","brick","chair","delta","eagle","flame","grape","house","ivory","joker","knife","lemon","mango","night","ocean","piano","queen","river","stone","tiger","umbra","vivid","whale","xenon","yacht","zebra","amber","blaze","cloud","drift","ember","frost","glove","heart","index","jelly","koala","laser","maple","nerve","olive","pearl","quilt","radar","sheep","torch","ultra","vapor","wheat","yield"]
D={w:random.randint(-99,99) for w in words}
r2=random.Random(500)
dtxt="\n".join(f"{w}: {v}" for w,v in D.items())
for i in range(3):
    keys=[r2.choice(words) for _ in range(500)]
    s=sum(D[k] for k in keys)
    p=(f"Dictionary:\n{dtxt}\n\nStarting from 0, add the value of each of the following 500 keys, in order, and report the final total.\n"
       f"Keys: {', '.join(keys)}\n\nReply with ONLY <answer>N</answer>, nothing else.")
    items.append(dict(K=500,i=i,ans=s,prompt=p))
json.dump(items,open("items.json","w"))
for o in items:
    if o["K"]==10: continue
    fn=f"item_K{o['K']}_{o['i']}.txt"; open(fn,"w").write(o["prompt"]); print(fn,o["ans"])
# check dict identical
assert items[0]["prompt"].split("Starting")[0]==items[-1]["prompt"].split("Starting")[0]
