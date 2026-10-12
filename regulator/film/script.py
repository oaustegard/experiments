"""The Regulator, the film. One entry per beat: the still it plays over, the Ken Burns move,
and the lines spoken over it. A line is (speaker, text, cards); a card is (timepiece, reading)
and shows a dial with that reading while the line plays, so every clue is on screen as well as
spoken. Speakers: M Matthey (narrator), O Odile, F Félix, L Lina, D the doctor, B Bastien, R Rosa.

PART is "open", "act1", "act2", "pause", "reveal" or "coda". Everything before "reveal" is
what the viewer gets to solve from; check_fair.py proves it carries every clue.
"""

SCENES = [
 # ---------------------------------------------------------------- cold open
 dict(part="open", shot="stairs", kb="up", lines=[
  ("M", "At ten minutes to midnight, on the last night of the nineteenth century, a maid named Rosa Häfeli climbed the stairs of a watchmaker's house in La Chaux-de-Fonds with a tray of champagne glasses.", []),
  ("M", "She was going to fetch her master for the toast.", [])]),
 dict(part="open", shot="body", kb="in", lines=[
  ("M", "Aurèle Monnier lay on the floor of his atelier. Someone had struck him once, from behind, with the brass mallet from his own bench.", [])]),
 dict(part="open", shot="regulator", kb="up", lines=[
  ("M", "On the wall above him, the regulator had stopped. Its hands stood at five past ten.", [])]),
 dict(part="open", shot="repeater", kb="in", lines=[
  ("M", "And in his right hand was the watch he had spent four years making. A minute repeater, with its slide pressed home.", [])]),
 dict(part="open", kind="title", shot="exterior", kb="out", lines=[]),

 # ---------------------------------------------------------------- act one
 dict(part="act1", shot="arrival", kb="left", lines=[
  ("M", "My name is Léon Matthey. In the winter of 1899 I was an inspector of the cantonal police, and I was called to that house a little after the bells had rung in the new century.", [])]),
 dict(part="act1", shot="town", kb="right", lines=[
  ("M", "La Chaux-de-Fonds is a town built for clocks. After the great fire they laid its streets out in a grid, and every house has a top floor of long windows, so that the benches catch the daylight.", [])]),
 dict(part="act1", shot="atelier_day", kb="in", lines=[
  ("M", "Monnier worked at the top of his house, under those windows, beside his best hand, a young régleuse named Lina Ruedi.", []),
  ("M", "A minute repeater sounds the time when you press its slide. One low note for each hour. A doubled note for each quarter past the hour. And a high note for each minute past the quarter.", []),
  ("M", "Monnier pressed it as he fell. Whoever struck him was standing in that room when it sounded.", [])]),
 dict(part="act1", kind="cast", who="O", shot="cast_O", kb="in", lines=[
  ("M", "Six people were in the house that night. His wife, Odile, a pastor's daughter, who played Chopin and did not weep.", [])]),
 dict(part="act1", kind="cast", who="F", shot="cast_F", kb="in", lines=[
  ("M", "His nephew Félix, the heir, who owed money to half the town.", [])]),
 dict(part="act1", kind="cast", who="L", shot="cast_L", kb="in", lines=[
  ("M", "Lina Ruedi, who regulated watches for a living and spoke in measurements.", [])]),
 dict(part="act1", kind="cast", who="D", shot="cast_D", kb="in", lines=[
  ("M", "Doctor Casimir Vauthier, physician to the family and its friend for thirty years.", [])]),
 dict(part="act1", kind="cast", who="B", shot="cast_B", kb="in", lines=[
  ("M", "Bastien Girard, a watch merchant from Le Locle, who wanted to buy the repeater and had quarrelled with Monnier over its price.", [])]),
 dict(part="act1", kind="cast", who="R", shot="cast_R", kb="in", lines=[
  ("M", "And Rosa, the maid, who kept her kitchen clock fast so that nothing would ever be late.", [])]),
 dict(part="act1", shot="dinner", kb="right", lines=[
  ("M", "Every year at dinner, when the temple bell struck nine, Monnier made his guests show him their watches, and told each of them what was wrong with it.", []),
  ("O", "Félix's watch showed twenty minutes to nine. Aurèle scolded him for it.", [("wF", "8:40")]),
  ("L", "Mine was exact. The doctor's was two minutes ahead. Monsieur Girard would not take his out of his pocket.", [("wL", "9:00"), ("wD", "9:02")]),
  ("M", "Remember those watches. Every clock in that house kept its own time, and before morning I would need all of them.", [])]),

 # ---------------------------------------------------------------- act two: the interviews
 dict(part="act2", shot="interview", kb="in", lines=[
  ("M", "I took them one at a time, in the dining room, between one and four in the morning. I spoke to Rosa first. She had found him.", [])]),
 dict(part="act2", shot="kitchen", kb="left", lines=[
  ("R", "My kitchen clock is a quarter of an hour fast. I keep it so on purpose, so nothing is ever late.", []),
  ("R", "Fräulein Lina came down into the kitchen at thirteen minutes to eleven by my kitchen clock, and she stayed with me until after the temple struck eleven.", [("kitchen", "10:47")]),
  ("R", "We heard Madame's piano begin at two minutes to eleven by my kitchen clock, and stop at eleven minutes past eleven.", [("kitchen", "10:58"), ("kitchen", "11:11")]),
  ("R", "I heard the hall clock strike eleven when my kitchen clock showed thirteen minutes past eleven.", [("morbier", "11:00"), ("kitchen", "11:13")])]),
 dict(part="act2", shot="cards", kb="right", lines=[
  ("M", "The doctor had examined the body. He was very helpful, as doctors are.", []),
  ("D", "Madame and I played piquet in the salon until a quarter to eleven by my watch, when I went out to the hall to fetch my cigar case from my coat. Madame began to play the piano as I went out, and she stopped when I came back.", [("wD", "10:45")]),
  ("D", "The hall clock struck eleven a moment after I sat down again. Tempus edax rerum, Inspector. Time devours all things, the clocks included.", [("morbier", "11:00")])]),
 dict(part="act2", shot="piano", kb="in", lines=[
  ("M", "Madame Monnier received me as if I had come about the gas.", []),
  ("O", "The doctor and I played piquet without a break. The temple struck ten while we played. We played on until he went out to fetch his cigars. I did not look at the clock when he went.", []),
  ("O", "As he went out he asked me to play for him. I sat down at the piano at once and played until he came back, and I stopped when he came in.", []),
  ("O", "A moment after he came back, the hall clock struck eleven.", [("morbier", "11:00")])]),
 dict(part="act2", shot="atelier_night", kb="up", lines=[
  ("M", "Lina Ruedi answered every question as if it were a measurement.", []),
  ("L", "After dinner I worked beside the master in the atelier. The hall clock downstairs struck ten when the regulator showed twelve minutes to ten. When the temple struck ten, the regulator agreed with the bell exactly.", [("regulator", "9:48"), ("morbier", "10:00"), ("regulator", "10:00")]),
  ("L", "He sent me downstairs when the regulator showed twenty past ten. It was running then, and he was well. I went straight down to the kitchen.", [("regulator", "10:20")])]),
 dict(part="act2", shot="kitchen_lina", kb="left", lines=[
  ("L", "In the kitchen I looked at Rosa's clock when Madame's piano began. It showed two minutes to eleven. When the piano stopped, it showed eleven minutes past eleven.", [("kitchen", "10:58"), ("kitchen", "11:11")]),
  ("L", "I stayed in the kitchen with Rosa until after the temple struck eleven.", [])]),
 dict(part="act2", shot="afternoon", kb="in", lines=[
  ("L", "That afternoon I saw the master in the kitchen with Rosa's clock in his hands, laughing. I did not ask why.", [])]),
 dict(part="act2", shot="library", kb="up", lines=[
  ("M", "Félix Monnier had spent the evening alone in the library, directly beneath his uncle's bench. He knew how that looked.", []),
  ("F", "At a quarter past ten by my watch, someone went quickly up the stairs past the library door. I did not see who. The door was shut.", [("wF", "10:15")])]),
 dict(part="act2", shot="ceiling", kb="in", chime=(0, 0.35), lines=[
  ("F", "At eighteen minutes past ten by my watch I heard a thump on the floor right above me. Then my uncle's repeater. Ten slow deep notes, I counted those. Then some doubled notes, then quick high ones. I did not count those.", [("wF", "10:18")])]),
 dict(part="act2", shot="door", kb="in", lines=[
  ("F", "At twenty-one minutes past ten by my watch, someone came quickly down. Again I did not see who.", [("wF", "10:21")]),
  ("F", "Bastien Girard came into the library at thirty-two minutes past ten by my watch, and we sat together there until the temple struck eleven.", [("wF", "10:32")])]),
 dict(part="act2", shot="street", kb="right", lines=[
  ("M", "Bastien Girard had spent the worst of it out in the frost.", []),
  ("B", "I went out into the street to smoke at twenty-one minutes past ten by my watch, and I stayed under the front windows of the house.", [("wB", "10:21")]),
  ("B", "The piano started up in the salon just as I lit my cigar, at twenty-nine minutes past ten by my watch.", [("wB", "10:29")]),
  ("B", "At twenty-six minutes to eleven by my watch I heard faint quick bells from the master's window up above.", [("wB", "10:34")])]),
 dict(part="act2", shot="hall", kb="in", lines=[
  ("B", "I came back into the house at fourteen minutes to eleven by my watch. The hall clock had just struck eleven, and its face showed two minutes past eleven. I went straight up to Félix in the library.", [("wB", "10:46"), ("morbier", "11:02")])]),

 # ---------------------------------------------------------------- the pause
 dict(part="pause", shot="interview_empty", kb="out", lines=[
  ("M", "That is everything I was told. One of the six lied to me. The other five told me honestly what they saw and heard, though what any of them believed about a clock was another matter.", []),
  ("M", "No clock was touched that evening except by the murderer's hand. The temple bell keeps true time. And no one in that house needs more than five minutes to go from any room to any other.", []),
  ("M", "Who struck Aurèle Monnier, and at what minute? If you would like to try, stop the film here.", [])]),
 dict(part="pause", kind="board", shot="interview_empty", kb="none", hold=20, lines=[]),

 # ---------------------------------------------------------------- the reveal
 dict(part="reveal", shot="gathering", kb="in", lines=[
  ("M", "At seven in the morning I asked them all into the salon. The doctor lit a cigar.", []),
  ("M", "I began with the one clock in that house no one could have moved. The temple bell.", [])]),
 dict(part="reveal", shot="morbier", kb="up", lines=[
  ("M", "Lina told me the regulator agreed with the bell at ten. She also heard the hall clock strike ten when the regulator showed twelve minutes to ten. So the old Morbier in the hall runs twelve minutes fast, and it struck its eleven at twelve minutes to eleven, true time.", [("morbier", "11:00=10:48")])]),
 dict(part="reveal", shot="kitchen_clock", kb="in", lines=[
  ("M", "Rosa heard that same strike when her clock showed thirteen minutes past eleven. Her clock was not a quarter of an hour fast that night. It was twenty-five minutes fast.", [("kitchen", "11:13=10:48")]),
  ("M", "Monnier had moved it that afternoon, a joke for the new year. Lina saw him laughing. And by Rosa's own clock, Lina reached the kitchen at twenty-two minutes past ten, two minutes after she left her master. With a quarter of an hour, she would have taken twelve minutes to come down two flights of stairs.", [("kitchen", "10:47=10:22")])]),
 dict(part="reveal", shot="piano", kb="right", lines=[
  ("M", "So the piano began at thirty-three minutes past ten. Girard's watch agrees. He came in as the Morbier showed two minutes past eleven, which was ten minutes to eleven, and his watch said fourteen minutes to. It runs four minutes slow. The piano started at twenty-nine past by his watch. Thirty-three past, again.", [("kitchen", "10:58=10:33"), ("wB", "10:29=10:33")])]),
 dict(part="reveal", shot="ceiling", kb="in", chime=(1, 1.0), lines=[
  ("M", "Now the repeater. Félix heard it at eighteen past ten by a watch that runs twenty minutes slow. Girard heard the bells at twenty-six minutes to eleven by a watch that runs four minutes slow. Both give the same minute. Thirty-eight minutes past ten.", [("wF", "10:18=10:38"), ("wB", "10:34=10:38")]),
  ("M", "Ten low notes. Two doubled. Eight high. Monnier told us the time himself.", [("repeater", "10:38")])]),
 dict(part="reveal", shot="cards", kb="left", lines=[
  ("M", "Madame and the doctor agree on one thing. The piano began as he went out. He says he went out at a quarter to eleven by his watch, which runs two minutes fast. Forty-three minutes past ten.", [("wD", "10:45=10:43")]),
  ("M", "But the piano began at thirty-three minutes past. When the repeater sounded, at thirty-eight, the doctor's story has him at the card table. It cannot be true, and only one of them lied to me.", [])]),
 dict(part="reveal", shot="climb", kb="up", lines=[
  ("M", "He left the salon at thirty-three minutes past ten, while Madame played for him. Félix heard him on the stairs at thirty-five. He struck at thirty-eight.", [])]),
 dict(part="reveal", shot="pendulum", kb="in", lines=[
  ("M", "Then he stopped the regulator and set its hands back to five past ten, an hour when he had sat at cards in plain sight of Madame Monnier. He came down at forty-one, took his cigar case from his coat, and was back at the card table at forty-six, when the piano stopped.", [("regulator", "10:05")])]),
 dict(part="reveal", shot="gathering", kb="right", lines=[
  ("M", "If you took Rosa at her word, as I very nearly did, the piano began at forty-three past, the minute the doctor named. His story held. And the man in the street with no one to vouch for him, and a quarrel over money, was Bastien Girard.", [])]),
 dict(part="reveal", shot="accused", kb="in", lines=[
  ("M", "The doctor put out his cigar. He did not deny it.", [])]),

 # ---------------------------------------------------------------- coda
 dict(part="coda", shot="letter", kb="in", lines=[
  ("M", "He asked for an hour to write a letter, and I gave it to him.", []),
  ("D", "Aurèle had told me in the hall, before dinner, that he would tell Félix at midnight. He had found his brother's last letter among the papers. It described the doses. I had written them in my own hand, and I had signed the certificate beneath them.", []),
  ("D", "He fell, and his hand closed on the slide, and the repeater sounded. I counted it, as anyone in this town would count it. It said ten thirty-eight.", []),
  ("D", "I have no more to add.", [])]),
 dict(part="coda", shot="dawn", kb="out", lines=[
  ("M", "Monnier had meant his repeater to strike the first minute of the new century. It sounded once, at thirty-eight minutes past ten, in his own hand.", [])]),
 dict(part="coda", kind="end", shot="dawn", kb="none", hold=8, lines=[]),
]
