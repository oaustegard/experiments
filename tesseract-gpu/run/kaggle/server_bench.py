"""Kaggle GPU job: tesseract-gpu's PyTorch engine (server/torch_engine.py) on a T4.

Pulls the engine at a pinned commit of oaustegard/experiments and the tessdata models from
GitHub, renders English benchmark sentences (Wikipedia text from the experiment's synthetic set,
CC BY-SA) as 36-px lines with DejaVu Sans, then measures: parity of exact mode on CUDA against
the NumPy reference, text agreement of cudnn mode with exact mode, and lines/s per mode, model
and batch size. Writes /kaggle/working/server_bench.json.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SHA = "__SHA__"
SENTENCES = json.loads(r'''["They also did not allow CRPF jawans to enter their village.", "It was also the first ever win against \"Seleção das Quinas\".", "They sought better service in Wellston.", "Daffodils and other bulbs generally bloom in February.", "Terrebonne is located in northeastern Deschutes County along U.S.", "McGill and manager W.H.", "The road was shut for several days for repairs.", "The institutions include Veda Bhavan at Puri.", "She remains with the group until the end.", "Most of these are in Scotland.", "Rounds earned his BS from Black Hills State University.", "Allahabad High Court By Justice Shiva Nath Katju.", "Playing roulette, he puts his last one hundred francs on Black.", "He was born in Philadelphia, and died in Woodbury Heights, New Jersey.", "This was her majesty Charlotte dowager queen of Württemberg.", "Gimple, Andrew Chambliss, and Ian B.", "\"The beginnings of Slavic settlement east of the river Elbe\".", "Her grandfather owned a bakery, where her father worked.", "His sketchbook is held in the Archives of American Art.", "Schreiben und Konzipieren für das Internet.", "The lines were elegant, despite the small size of the body.", "On television, she portrayed B.J.", "Am I? - Faremi dore?", "Humphreys, Randolph S.", "His brother Harry H.", "After mitosis it also is essential in the degradation of the tapetum.", "Drawings of such blades were found in prehistoric caves.", "Snow is rarer, but not unusual in Harbor, averaging only per year.", "Frank Coombs, California legislator, U.S. congressman, U.S.", "He helped Apolonia maintain a mid-table position.", "Dan is characterised as a \"likeable, committed and passionate\" man.", "Drain is a city in Douglas County, Oregon, United States.", "G Yard, consisting of five tracks.", "An alternative is Jan Smit Czn. in Alblasserdam.", "He even molded a bust of Washington.", "Louis Mayor Rolla Wells.", "Munden, Kenneth White.", "La teoria literària.", "Simultaneous premiere on HC and HMM.", "Bolidophyceae is the sister taxon to the diatoms (Bacillariophyceae).", "Hadfield is a surname.", "In this role he oversaw the logistical support.", "As she grew older, she suffered increasingly from poor health.", "Wodehouse and Scott Meredith.", "Forde ended the season by receiving an All-Star nomination.", "The omnibus was the first of its kind west of the Mississippi.", "He designated Brigadier General Bennett E.", "Catrina appeared in his Pink Windmill show.", "There was a single Solex carburettor.", "This programme is sponsored by the MTV Staying Alive initiative.", "And a reservoir called Fengshan Reservoir is to the south of park.", "As she tried to leave, he shot her in the face, killing her.", "Les AuCoin, a nine-term member of the U.S.", "MGM head of European operations Robert Littman flew out to the set.", "Provision of an active travel route and crossing facilities.", "Woluwe-Zaventem, and Armenian club Alashkert, and Belgian clubs K.V.V.", "It was built in two series.", "The reading was accompanied by an interview.", "Redmond is a city in Deschutes County, Oregon, United States.", "The film score includes a song about building a snowman.", "The teeth of the rear axle pinion were of the \"Citroën chevron\" type.", "Alabana () is a village in the municipality of Blace, Serbia.", "Does he understand? - Falafa dofa?", "During his partnership with Grounds this interest was intensified.", "\"The word goy does not apply to aliens.", "The Department of Economics at Beloit is also named after him.", "Wellington, Dominion Museum.", "Taxa named by Michael S.", "In football, he was a four-year letterman and a three-year starter.", "He withdraws, leaving Anselm and Myrtle to discuss their predicament.", "Keith Basso and Henry A.", "It was formerly located at .", "Newsweird, which features strange news from around the globe.", "Senator Mike Rounds.", "Concha Maria Gómez is an American mathematician.", "There were no spaces between words.", "It was won by Tom Stamsnijder of the team.", "He has been involved in making music since an early age.", "She served in several diplomatic posts during her career with the U.S.", "Myrtle convinces him to send in the claim anyway.", "The Supreme Court also affirmed the judgment of the district court.", "However, this will not work for all sORFs.", "La Pine is part of the Bend, Oregon Metropolitan Statistical Area.", "Later, Canfield manages to take Granny's gun.", "Bowser and lawyer S.D.", "The storm swept away a bridge along the Lake Worth Lagoon.", "The gray water is then piped out of the box and into a bucket.", "\"Crossroads\" (Robert Johnson, arr.", "Indirect objects always come after the verb.", "If they refused they were satirised.", "Mika Vasara himself has a family of four.", "Cooper Dodson as Dylan: Annie and Max's younger brother.", "This house served as a summer home in Washington.", "First ever goal scorer: Raman Vijayan for East Bengal v.", "One supplied about per minute as and the other about per minute at .", "It is reached by U.S.", "He was born the seventh of eight siblings.", "MC Serch and Prince Paul deliberate.", "The Mahoor Institute published both books.", "Clothing, such as a hat or scarf, may be included.", "At the dinner, N.O.R.E. appears to leave early.", "Lou King led the conference with eight interceptions.", "Eventually rabbi Smallman brought everybody to a common point of view.", "OEC's are processed on-site and in itself cost .", "The entire length of the road has clearway restrictions on it.", "He started learning music at age nine under Maestro Hassan Kamkar.", "Hindman, Sandra et al.", "He is a strong fisherman, can start a fire and hunt.", "Louis to be inadequate.", "Morrison, Elizabeth and Kren, Thomas (eds).", "Each section of the road was completed in separate link roads.", "It includes the Grangetown Viaduct.", "Canfield then gives his gun to Marcia.", "The theme of the rap was \"thug\" versions of old nursery rhymes.", "Louis Globe-Democrat, resided in St.", "A Danube hypermarket is one anchor.", "Sydney Lemmon as Isabelle: A CRM pilot who befriends Althea.", "O'Connell, Daniel J.", "Hendrick, city building inspector (manslaughter).", "Walters was born in Wheathampstead, Hertfordshire, where he grew up.", "Émile Paladilhe was born in Montpellier.", "The day's result massively shuffled the overall classification.", "In linguistics, is-a relations are called hyponymy.", "Mariela is a given name.", "Today at all resolutions the preferred method is FTIR.", "The board was empowered to appoint a president and three managers.", "Dietz also invented the chalybssonnans.", "The Blue Cross is an Advisory Member of the ABTC.", "The front part of the building was designed by Sidney R.", "Bingo readily consents, and Rosie leaves.", "The river Schilde flows through the municipality.", "Runnymede Road divides the yards.", "Both Cory J. and Mclachlin J. wrote separate concurring opinions.", "She was the first female Lord Mayor of Belfast.", "It preserved patterns of earlier Prague-Korchak material.", "The planet includes a coordinate grid, and fictional white landmasses.", "Their fibers run in opposite directions.", "Lessig is the founder and president of Equal Citizens.", "She travelled from Kimberley and went on to Thetford.", "To solve their social problems.", "Nunn, won the governorship and the office of secretary of state.", "The car had two separate seats, imitation leather.", "He is signed as a producer to Universal Publishing.", "He ended his career as the school's all-time leading scorer.", "Solresol has been assigned the codes and in the ConLang Code Registry.", "He is also the brother of Roque Santa Cruz.", "Lartey was a founding member of the People's National Party (PNP).", "There she earned her bachelor's degree in languages and theater.", "Rosenthal, which owned the building (maintaining a common nuisance).", "The soldiers rushed to their units' armories to equip themselves.", "Rugby union in Mayotte is a minor but growing sport.", "There were riding stables and golf courses nearby.", "Germany was the three-time defending champion.", "showed the weak derandomization result that BPL is contained in SC.", "The number of the seats for each party is determined proportionally.", "Electricity is available via a pole on the property site.", "Evelyn leaves Adam standing alone in the gallery.", "Eminem (Shake That feat.", "Collim and Grace Johnston.", "Blip had been amassing exclusive comedy programming.", "Habitat: The Ecopolitical Nation.", "It offers a whole-genome sequencing service.", "Blip used to have KipKay's DIY channel on the site.", "Lathrop and Maria Margaretta ( Long) Lathrop.", "Their friends are glad to see them so agreeably situated.", "In the centre of the flag was placed a Dymaxion map of the world.", "Their first challenger was manufactured by King C.", "He was a member of the House of Este.", "I'm Through With Love (Johnny Mercer, Gus Kahn and Joseph A.", "The road has many grade separated interchanges.", "Her first name means \"victory\" in the Tigrinya language.", "Rugby was first introduced to the Réunion by the French.", "Stuart was married to actors Frank Maxwell and later, David Shaw.", "Bingo walks Algy in a baby carriage in Wimbledon Common.", "Lowe, one of the original founders of the town) Dr.", "The judge formally sentenced Randy to death the next day.", "Tourism is one of Bend's largest sectors.", "Their messages will be transmitted in Morse Code.", "Politicians from St.", "This is a large long-legged bush cricket with short wings.", "In addition to snowmen, other things can be made from snow.", "The Elabela gene (Ela) (a.k.a.", "Since then she became a regular in the telenovela circuit.", "Polk Township is one of twenty townships in Benton County, Iowa, USA.", "All twelve aboard the Falcon were killed.", "Paddock and Augustus Kountze.", "They have remained in the Championship ever since.", "John's Catholic church in Canton.", "Leah thinks that it is sign she should not marry Dan.", "The contestant was the runner-up.", "Bánffy is a Hungarian surname.", "Ro-vibrational spectroscopy concerns molecules in the gas phase.", "The first episode featured folk blues singer Josh White.", "Skrvenica is a village in the municipality of Dimitrovgrad, Serbia.", "The stack pointer is used to manage the run-time stack.", "Chainsaw Carvers from three continents compete at the event.", "Her parents were from Canada.", "\"Stateside\" (Bowie, H.", "A Yard, consisting of ten tracks.", "This cylinder is coated with a special mixture.", "Union Township is one of twenty townships in Benton County, Iowa, USA.", "\"If they want to make changes let them make changes.", "It remained unclear if the Greenup pilgrimage would continue.", "La Pine had remained an unincorporated community for many years.", "After the challenge, she was found unconscious in the bathroom.", "Calabrese grew up in Bridgewater, Massachusetts.", "Rudi buys a new apartment and proposes to Fay in it.", "Included in this trip was an extra event in the shape of a typhoon!", "UN Representative: Sylvester E.", "\"Die sorbische Keramik in Thüringen\".", "The city has two covered bridges.", "The morning after, her stern was submerged.", "Singh, a former member of the Legislative Council.", "Similar caveats apply to most architectures.", "By the end of the war, the house survived in one piece.", "Orthologs are found in the turtle, frog and fish genomes as well.", "Calapooya Creek, a tributary of the Umpqua River, flows by Oakland.", "There, she makes two flamboyant friends.", "His mother taught him to play the piano from age five.", "Oofy is in love, but nonetheless in no mood to lend money.", "De Wynter Courtney Claibourne Magillicuddy Walters).", "Planinica is a hamlet in the municipality of Dimitrovgrad, Serbia.", "He also owned an early French-built Wright-model airplane.", "A snow sculpture of a woman is called a snowwoman.", "It was created by author and statistician Dean Oliver.", "He, Cockerell and Anderson arm themselves and wait for Canfield.", "At the end of the next season, she enjoyed her first championship.", "The crater was named by the famous alpinist Mr.", "The nearest railway station five kilometres away at Schwerin.", "It was named after Heping () Street that was located within it.", "It comes down to Sullee and G-Child.", "Caravale, Mario (ed).", "A timber yard and boat building yard previously occupied the site.", "In protest, Rwanda broke diplomatic relations with France.", "Dulinicz, Frühe Slawen im Gebiet zwischen unterer Weichsel und Elbe.", "Shazli writes that the Army had three parachute brigades at the time.", "She is on her way to comfort Mrs.", "After retiring from the U.S.", "He also added color to his standard etching techniques.", "The song features uncredited vocals from British singer Vanya Taylor.", "KWRX is a simulcast of KWAX in Eugene, Oregon.", "Even country music legend Garth Brooks has shopped the Fort’s aisles.", "William Harvey, M.D., and Charles H.", "Born in Gudensberg, Hesse, he represents the CDU.", "His successor was his half-brother Ercole I d'Este.", "A day fitter was based at Hull from Bombardier at Crofton TMD.", "He then cuts ABS pipe to be used as a launcher.", "Wilson co-wrote the song with Jeff Pardo and Matthew West.", "The original music score was written by Basil Poledouris.", "Many modern ones are powered using rechargeable batteries.", "The diverted land was devoted to approved conservation practices.", "He gives her the credit but she insists on his having it.", "In every way this is a most excellent book for children.", "Borata is a town in southwestern Lesotho.", "He is a member of the Shan Nationalities League for Democracy.", "He is a member of the Committee for Food and Agriculture.", "He was defeated at the by-election by Edward Bayntun-Rolt.", "In addition, points could be won in intermediate sprints.", "And where do you fit the Samaritans in all this?", "A more general expression was given by Dunham.", "Straight razors eventually fell out of fashion.", "Lysenin's conductive properties have been studied for years.", "\"Frühslawischer Kastenbrunnen mit Holzfunden aus Eythra, Kr.", "A street in Baku was named after him.", "Oofy resents Bingo's presence and refuses to give him money.", "He wrote numerous articles for the likes of Multitudes and Regards.", "She gives Bingo ten pounds to pay a bill for a dog harness.", "This stat can't be influenced by the defense of a player's teammates.", "For instance fala means good or tasty, and lafa means bad.", "Dan is successful and Peter steps aside.", "Alaskan law limits contributions to independent political groups.", "They fought over the city of Ratzeburg.", "This species is endangered.", "Bedan, bound from Glasgow to Derry.", "Fellow Albanian Edmond Kapllani had joined the club a month earlier.", "Rosenthal and managed by Nathan Ginsburg.", "Nolan served on her dissertation committee.", "She was originally built for the South Georgia Co.", "We have achieved that.", "Chionodes irreptor is a moth in the family Gelechiidae.", "They were black in colour, the top of the grooves was polished.", "He was married twice and had six daughters.", "Scott, Tony Musante, Trish Van Devere, and Colleen Dewhurst.", "Difanis Gómez and Nicolas Humberto Gómez.", "To provide educational facilities to the untouchables.", "Dan begins drinking heavily.", "Bend is known for its annual Indian summer.", "One has been located about to the east called La Fortaleza.", "Arguments over inheritance were becoming increasingly bitter.", "His first marriage was to a woman named Sherry Buford.", "Asian Popular Culture - John A.", "Recipients of the Order of St.", "He flew aircraft in several films, including Wings and Hells Angels.", "Fleischer turned down the job, disliking the script.", "The club served the community in a variety of ways.", "Henni was in charge of St.", "They function to truncate the transcription factor Shavenbaby (Svb).", "Rome in the Middle Ages Vol.", "In October he was unplaced behind Prince Royal and Santa Claus.", "In later years, he taught voice."]''')
ROOT = Path("/kaggle/working/tg")
RAW = f"https://raw.githubusercontent.com/oaustegard/experiments/{SHA}/tesseract-gpu/"
FILES = ["tessgpu/__init__.py", "tessgpu/traineddata.py", "tessgpu/ref.py", "tessgpu/decode.py",
         "server/__init__.py", "server/torch_engine.py"]
MODELS = {"best": "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/eng.traineddata",
          "fast": "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/eng.traineddata"}
OUT = {"sha": SHA}


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                dest.write_bytes(r.read())
            return
        except Exception as e:  # noqa: BLE001
            if k == 3:
                raise
            time.sleep(2 * (k + 1))


for f in FILES:
    fetch(RAW + f, ROOT / f)
for v, u in MODELS.items():
    fetch(u, ROOT / f"models/{v}/eng.traineddata")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from server.torch_engine import TorchTess  # noqa: E402
from tessgpu import decode, ref  # noqa: E402
from tessgpu.traineddata import load  # noqa: E402

OUT["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
OUT["torch"] = torch.__version__
OUT["nvidia_smi"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()
dev = "cuda" if torch.cuda.is_available() else "cpu"

import matplotlib  # noqa: E402
FONT = str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf")
rng = np.random.default_rng(0)


def render(text, size):
    font = ImageFont.truetype(FONT, size)
    bb = font.getbbox(text)
    im = Image.new("L", (bb[2] + 16, 36), 255)
    ImageDraw.Draw(im).text((8, (36 - (bb[3] + bb[1])) // 2), text, font=font, fill=int(rng.integers(0, 60)))
    g = np.asarray(im).astype(np.int16) + rng.normal(0, 6, (36, im.width)).astype(np.int16)
    return np.clip(g, 0, 255).astype(np.uint8)


LINES = [render(SENTENCES[i % len(SENTENCES)], int(rng.integers(18, 25))) for i in range(4096)]
OUT["lines"] = len(LINES)
OUT["mean_width_px"] = float(np.mean([g.shape[1] for g in LINES]))


def sync():
    if dev == "cuda":
        torch.cuda.synchronize()


results = {}
for variant in ("fast", "best"):
    path = f"models/{variant}/eng.traineddata"
    m = load(path)
    codec = decode.Codec(m)
    r = {}
    exact = TorchTess(path, device=dev, mode="exact")
    cud = TorchTess(path, device=dev, mode="cudnn")
    # parity of exact mode on this device against the NumPy reference
    sample = LINES[:24]
    got = exact.recognize(sample)
    want = [ref.forward(m, g) for g in sample]
    r["exact_vs_ref_max_dp"] = float(max(np.abs(a - b).max() for a, b in zip(got, want)))
    r["exact_vs_ref_argmax_lines_equal"] = sum(bool((a.argmax(1) == b.argmax(1)).all()) for a, b in zip(got, want))
    r["exact_vs_ref_lines"] = len(sample)
    # cudnn vs exact text on 1024 lines
    te = exact.texts(LINES[:1024])
    tc = cud.texts(LINES[:1024])
    r["cudnn_text_equal_exact"] = sum(a == b for a, b in zip(te, tc)) / len(te)
    r["exact_text_equal_source"] = sum(decode.clean(a) == decode.clean(SENTENCES[i % len(SENTENCES)])
                                       for i, a in enumerate(te)) / len(te)
    for mode, eng in (("exact", exact), ("cudnn", cud)):
        for bs in (64, 256, 1024, 4096):
            batch = LINES[:bs]
            eng.recognize(batch[: min(bs, 64)])  # warm
            times = []
            for _ in range(3):
                sync()
                t0 = time.perf_counter()
                probs = eng.recognize(batch)
                sync()
                times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            for p in probs:
                decode.greedy(p, codec)
            dec = time.perf_counter() - t0
            s = statistics.median(times)
            r[f"{mode}_b{bs}"] = {"s": s, "lines_per_s": bs / s, "greedy_decode_s": dec}
            print(variant, mode, bs, f"{bs / s:.0f} lines/s", flush=True)
    results[variant] = r
OUT["results"] = results
Path("/kaggle/working/server_bench.json").write_text(json.dumps(OUT, indent=1))
print(json.dumps(OUT, indent=1))
