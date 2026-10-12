"""First-person fact cards for the interview writers. Each line renders one
statement in world.py and must survive into the prose with every timepiece
and reading intact. TEXTURE lists what a witness may add that carries no
time, place or movement claim."""

FACTS = {
"O": {
 "O1": "At dinner, when the temple bell struck nine, my husband made everyone show their watches. Félix's watch showed twenty minutes to nine. Aurèle scolded him for it.",
 "O2": "After dinner the doctor and I played piquet in the salon without a break. The temple struck ten while we played. We played on until he went out to fetch his cigars. I did not look at the clock when he went.",
 "O3": "As he went out he asked me to play for him. I sat down at the piano at once and played until he came back, and I stopped when he came in.",
 "O4": "A moment after he came back, the hall clock struck eleven.",
},
"F": {
 "F1": "At dinner, when the temple bell struck nine, my watch showed twenty minutes to nine. My uncle made a scene about it in front of everyone.",
 "F2": "After dinner I went up to the library and stayed there alone.",
 "F3": "At eighteen minutes past ten by my watch I heard a thump on the floor right above me, in the atelier, and then my uncle's repeater sounding: ten slow deep notes, I counted those, then some doubled notes, then quick high ones that I did not count.",
 "F4": "At a quarter past ten by my watch, someone went quickly up the stairs past the library door. At twenty-one minutes past ten by my watch, someone came quickly down. I did not see who; the door was shut.",
 "F5": "Bastien Girard came into the library at thirty-two minutes past ten by my watch, and we sat together there until the temple struck eleven.",
},
"L": {
 "L1": "At dinner, when the temple bell struck nine, the master made everyone show their watches. Mine was exact. The doctor's was two minutes ahead. Monsieur Girard would not take his out of his pocket.",
 "L2": "After dinner I worked beside the master in the atelier. The hall clock downstairs struck ten when the regulator showed twelve minutes to ten; the master said what he always says about that clock. When the temple struck ten, the regulator agreed with the bell exactly.",
 "L3": "He sent me downstairs when the regulator showed twenty past ten. The regulator was running then, and he was well. I went straight down to the kitchen.",
 "L4": "That afternoon I saw the master in the kitchen with Rosa's clock in his hands, laughing. I did not ask why.",
 "L5": "In the kitchen I looked at Rosa's clock when Madame's piano began: it showed two minutes to eleven. When the piano stopped, Rosa's clock showed eleven minutes past eleven.",
 "L6": "I stayed in the kitchen with Rosa until after the temple struck eleven.",
},
"D": {
 "D1": "At dinner, when the temple bell struck nine, Aurèle inspected our watches. Mine was two minutes ahead of the bell.",
 "D2": "After dinner Madame and I played piquet in the salon until a quarter to eleven by my watch, when I went out to the hall to fetch my cigar case from my coat. Madame began to play the piano as I went out, and she stopped when I came back.",
 "D3": "The hall clock struck eleven a moment after I sat down again.",
},
"B": {
 "B1": "At dinner I would not show the master my watch. After dinner I sat on alone in the dining room with the brandy, and heard the temple strike ten there; I did not look at my watch.",
 "B2": "I went out into the street to smoke at twenty-one minutes past ten by my watch, and I stayed under the front windows of the house.",
 "B3": "The piano started up in the salon just as I lit my cigar, at twenty-nine minutes past ten by my watch.",
 "B4": "At twenty-six minutes to eleven by my watch I heard faint quick bells from the master's window up above.",
 "B5": "I came back into the house at fourteen minutes to eleven by my watch. The hall clock had just struck eleven and its face showed two minutes past eleven. I went straight up to Félix in the library.",
},
"R": {
 "R0": "My kitchen clock is a quarter of an hour fast. I keep it so on purpose, so nothing is ever late.",
 "R1": "Fräulein Lina came down into the kitchen at thirteen minutes to eleven by my kitchen clock, and she stayed with me until after the temple struck eleven.",
 "R2": "We heard Madame's piano begin at two minutes to eleven by my kitchen clock, and stop at eleven minutes past eleven by my kitchen clock.",
 "R3": "I heard the hall clock strike eleven when my kitchen clock showed thirteen minutes past eleven.",
},
}

TEXTURE = {
"O": "Her marriage; the doctor as the family's oldest friend; Félix's debts; the quarrel at dinner between her husband and Monsieur Girard; that she played Chopin (name no piece and say nothing about how long she played); she must not give any clock time for the doctor's going out; her cold reserve about the death.",
"F": "His father Jules, who died in 1889; the silver watch was his father's and he has never let anyone regulate it, not even his uncle; his debts, some to Girard; brandy; his fear that he will be suspected because he was alone below the atelier; what the repeater meant to his uncle. He says nothing at all about the piano.",
"L": "How a minute repeater speaks: low notes for the hours, a doubled note for each quarter, high notes for each minute past the quarter (she may explain this when asked about the repeater); four years of work on it; the master's exactness; her own watch, which she regulated herself; she does not speculate.",
"D": "Thirty years as the family's friend and physician; his medical observations of the body (a blow to the back of the head, death would have been very quick); he may say the regulator 'stopped with its master', a sentimental remark that names no hour; cigars; piquet; Latin tags; that he attended Jules Monnier in 1889. He offers no hour of death of his own.",
"B": "The repeater rights; the quarrel at dinner about the price; that he is owed money by Félix; the cold; his cheap Brissago cigars; his dislike of the doctor's company; that a merchant notices time. He says nothing about the piano beyond the fact given, and nothing about what the bells were.",
"R": "Her kitchen and the champagne glasses for midnight; why she keeps the clock fast (she is sure of the quarter hour); finding the master's body when she went up to fetch him for the toast, without giving any time; her fear of the police; loyalty to Madame.",
}

ROLE = {
"O": "Odile Monnier, the widow",
"F": "Félix Monnier, the nephew",
"L": "Lina Ruedi, the régleuse",
"D": "Dr Casimir Vauthier, the family physician",
"B": "Bastien Girard, the établisseur",
"R": "Rosa Häfeli, the maid and cook",
}
