import random, json
items=json.load(open("items.json"))
random.seed(2509)
words = ["apple","brick","chair","delta","eagle","flame","grape","house","ivory","joker","knife","lemon","mango","night","ocean","piano","queen","river","stone","tiger","umbra","vivid","whale","xenon","yacht","zebra","amber","blaze","cloud","drift","ember","frost","glove","heart","index","jelly","koala","laser","maple","nerve","olive","pearl","quilt","radar","sheep","torch","ultra","vapor","wheat","yield"]
D={w:random.randint(-99,99) for w in words}
dtxt="\n".join(f"{w}: {v}" for w,v in D.items())
for K,seed in ((1000,1000),(2000,2000)):
    r2=random.Random(seed)
    for i in range(3):
        keys=[r2.choice(words) for _ in range(K)]
        s=sum(D[k] for k in keys)
        p=(f"Dictionary:\n{dtxt}\n\nStarting from 0, add the value of each of the following {K} keys, in order, and report the final total.\n"
           f"Keys: {', '.join(keys)}\n\nReply with ONLY <answer>N</answer>, nothing else.")
        items.append(dict(K=K,i=i,ans=s,prompt=p))
        open(f"item_K{K}_{i}.txt","w").write(p); print(K,i,s)
json.dump(items,open("items.json","w"))
