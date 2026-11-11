import json, random, pandas as pd
random.seed(7); N=60; D='/tmp/w/data/'; out=[]
# PubMedQA (pqa_labeled): context + question -> yes/no/maybe
d=json.load(open(D+'ori_pqal.json'))
for k in random.sample(sorted(d),N):
    r=d[k]; ctx="\n".join(r['CONTEXTS'])
    out.append({"id":f"pmq-{k}","bench":"PubMedQA","input":f"Question: {r['QUESTION']}\n\nAbstract (conclusion withheld):\n{ctx}",
      "questions":[{"type":"choice","name":"a","instructions":"Based on the abstract, what is the answer to the research question?",
        "choices":[{"value":"yes","description":"The evidence supports yes."},{"value":"no","description":"The evidence supports no."},{"value":"maybe","description":"The evidence is mixed or inconclusive."}]}],
      "labels":{"a":r['final_decision']}})
# ESCI: query + product -> E/S/C/I ; stratified-ish random from test, small_version US
ex=pd.read_parquet(D+'shopping_queries_dataset_examples.parquet')
ex=ex[(ex.split=='test')&(ex.product_locale=='us')&(ex.small_version==1)].sample(N,random_state=7)
pr=pd.read_parquet(D+'shopping_queries_dataset_products.parquet',columns=['product_id','product_locale','product_title','product_brand','product_bullet_point'],
   filters=[('product_locale','==','us'),('product_id','in',list(ex.product_id))])
pr=pr.drop_duplicates('product_id').set_index('product_id')
lab={'E':'exact','S':'substitute','C':'complement','I':'irrelevant'}
for _,r in ex.iterrows():
    p=pr.loc[r.product_id]; bp=(p.product_bullet_point if isinstance(p.product_bullet_point,str) else '')[:800]; p_brand=p.product_brand if isinstance(p.product_brand,str) else ''
    out.append({"id":f"esci-{r.example_id}","bench":"ESCI","input":f"Search query: {r.query}\n\nProduct: {p.product_title}\nBrand: {p_brand}\n{bp}",
      "questions":[{"type":"choice","name":"a","instructions":"How relevant is this product to the shopper's search query?",
        "choices":[{"value":"exact","description":"Relevant for the query and satisfies all query specifications."},
          {"value":"substitute","description":"Somewhat relevant: fails some aspects of the query but can be used as a functional substitute."},
          {"value":"complement","description":"Does not fulfil the query but could be used in combination with an exact item."},
          {"value":"irrelevant","description":"Irrelevant to the query, or fails a central aspect of it."}]}],
      "labels":{"a":lab[r.esci_label]}})
# When2Call MCQ: pick the right behaviour
rows=[json.loads(l) for l in open(D+'when2call_test_mcq.jsonl')]
for r in random.sample(rows,N):
    tools="\n".join(r['tools']) or "(no tools available)"
    out.append({"id":f"w2c-{r['uuid']}","bench":"When2Call","input":f"Available tools:\n{tools}\n\nUser request: {r['question']}\n\nCandidate assistant responses:\n"+"\n".join(f"[{k}] {v}" for k,v in r['answers'].items()),
      "questions":[{"type":"choice","name":"a","instructions":"Which candidate response is the correct behaviour for a tool-using assistant given the tools available?",
        "choices":[{"value":"tool_call","description":"Call a tool: an available tool fits and all required arguments are known."},
          {"value":"request_for_info","description":"Ask a follow-up: a tool fits but required information is missing."},
          {"value":"cannot_answer","description":"Decline: no available tool can do it and it needs external action or data."},
          {"value":"direct","description":"Answer directly without a tool."}]}],
      "labels":{"a":r['correct_answer']}})
with open('/tmp/w/run/items.jsonl','w') as f:
    for o in out: f.write(json.dumps(o)+'\n')
import collections; print(collections.Counter((o['bench'],o['labels']['a']) for o in out))
