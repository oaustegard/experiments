"""The life-science section decision suite: question texts, options, and the two state layouts.

Questions use the Typed Decisions shapes that browser-decider's trainer reads (`noul`, `choice`
with an option -> description map, `score` with an ordered list of level descriptions), so a case
built here trains with `train_enc.py --data` unchanged. `SPIKE` groups the questions into the three
adapters the plan names.
"""

FIELDS = {
    "molecular_cell": "Molecular or cell biology, biochemistry, structural biology",
    "genetics_genomics": "Genetics, genomics, transcriptomics, bioinformatics",
    "immunology_infection": "Immunology, microbiology, virology, infectious disease",
    "neuroscience_mental": "Neuroscience, neurology, psychiatry, psychology",
    "oncology": "Cancer biology or clinical oncology",
    "cardiometabolic": "Cardiovascular, metabolic, endocrine or renal disease, nutrition",
    "clinical_other": "Other clinical medicine, surgery, dentistry, nursing, rehabilitation",
    "public_health": "Public health, epidemiology, health services, health policy, medical education",
    "ecology_plant_animal": "Ecology, evolution, plant science, agriculture, veterinary and animal science",
    "other": "Chemistry, materials, engineering, method development, or anything else",
}

SECTION_QS = {
    "role": {
        "type": "choice",
        "instructions": "Which part of a research article is this section?",
        "criteria": {
            "introduction": "Introduction or background: motivates the work and reviews what is known",
            "methods": "Methods: how the study was done",
            "results": "Results: what the study found",
            "discussion": "Discussion or conclusion: interprets the findings",
            "other": "Something else, such as a case description, a protocol, or a mixed section",
        },
    },
    "study_type": {
        "type": "choice",
        "instructions": "What kind of study does the article this section comes from report, as far as the section shows?",
        "criteria": {
            "in_vitro": "Experiments on cells, tissues, proteins or other biological material outside a living organism",
            "animal": "Experiments or observations on live non-human animals",
            "human_interventional": "People assigned to an intervention: a clinical trial or an experimental manipulation",
            "human_observational": "People observed without assignment: cohort, case-control, cross-sectional, case report, survey, qualitative study",
            "computational": "Analysis of existing datasets, modelling, simulation or software, with no new laboratory or participant data",
            "review": "A review or meta-analysis of published literature",
            "other_or_unclear": "Another kind of study (plants, field ecology, chemistry, engineering), or the section does not show which",
        },
    },
    "field": {
        "type": "choice",
        "instructions": "Which field does this section belong to?",
        "criteria": FIELDS,
    },
    "immunoassay": {
        "type": "noul",
        "instructions": "The section mentions quantifying an analyte in samples with a ligand-binding immunoassay: "
                        "ELISA, electrochemiluminescence (such as Meso Scale Discovery), a bead-based multiplex (such as "
                        "Luminex), Simoa, radioimmunoassay or a lateral-flow test. Western blots, immunohistochemistry, "
                        "immunofluorescence and flow cytometry do not count.",
    },
    "human_samples": {
        "type": "noul",
        "instructions": "The section reports on human participants or human-derived samples (patients, volunteers, human "
                        "tissue, blood, primary human cells). Established human cell lines alone do not count.",
    },
    "quantitative_result": {
        "type": "noul",
        "instructions": "The section reports at least one finding of the study itself as a number: an effect size, "
                        "percentage, count, mean, fold change or p-value. Numbers that only describe methods (doses, "
                        "volumes, planned sample sizes) and numbers quoted from other papers do not count.",
    },
    "detail": {
        "type": "score",
        "instructions": "How reproducible is what the section describes?",
        "criteria": [
            "No procedure or analysis is described: background, interpretation or narrative only",
            "Says what was done in general terms, without specifics",
            "Describes procedures or analyses with some specifics (named methods, some parameters), but a reader could not repeat them from this text alone",
            "Detailed enough to repeat: named reagents or instruments, quantities, conditions, software or statistical tests",
        ],
    },
    "evidence": {
        "type": "score",
        "instructions": "How strongly does the section support its main claim with data it presents?",
        "criteria": [
            "Makes no empirical claim (a methods description, a definition, a plan)",
            "Its claims rest on cited prior work, reasoning or qualitative description, with no data of its own",
            "Presents the study's own data for its main claim, but without statistics, controls or sample sizes",
            "Presents the study's own quantitative data with statistical support, controls or sample sizes",
        ],
    },
}

REL_Q = {
    "relevance": {
        "type": "score",
        "instructions": "How well does the section answer the search query?",
        "criteria": [
            "Unrelated to the query",
            "Same broad topic, but it does not address what the query asks",
            "Partly answers the query, or covers a closely related aspect",
            "Directly answers the query",
        ],
    },
}

SPIKE = {"classify": ["role", "study_type", "field", "immunoassay", "human_samples", "quantitative_result"],
         "rate": ["detail", "evidence"], "relevance": ["relevance"]}
ALL_QS = {**SECTION_QS, **REL_Q}


def options(q):
    """Option keys in order, as the trainer and the scorer see them."""
    if q["type"] == "noul":
        return ["true", "false"]
    if isinstance(q["criteria"], list):
        return [str(i) for i in range(len(q["criteria"]))]
    return list(q["criteria"])


def section_state(text):
    return text


def relevance_state(query, text):
    return f"query: {query}\n\nsection: {text}"


def jev_question(q):
    """Jev request form. Scores go as a choice over level keys so every level gets a probability."""
    if q["type"] == "noul":
        return {"type": "noul", "instructions": q["instructions"]}
    crit = q["criteria"]
    if isinstance(crit, list):
        crit = {str(i): d for i, d in enumerate(crit)}
    return {"type": "choice", "instructions": q["instructions"], "criteria": crit}


def jev_probs(q, ans):
    if q["type"] == "noul":
        p = float(ans["noul"])
        return {"true": p, "false": 1.0 - p}
    return {k: float(ans["probabilities"][k]) for k in options(q)}
