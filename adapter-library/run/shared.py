"""Paths, the browser-decider imports, and the one sampling rule every growth method shares."""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from _lib.paths import experiment  # noqa: E402

BD = experiment("browser-decider")
sys.path.insert(0, str(BD / "run"))
from common import WORKFLOWS, examples, load_cases, score_examples  # noqa: E402,F401

E17C = BD / "out/base/st-ettin17m-tuned-e17-cap-c-epoch1/epoch1"
E17C_URL = ("https://github.com/oaustegard/experiments/releases/download/ettin-native-20261021/"
            "st-ettin17m-tuned-e17-cap-c-epoch1.zip")
SIZES = [0, 16, 64, 300]
GPU_SEEDS = [0, 1]
KNN_SEEDS = [0, 1, 2, 3, 4]


def revealed(workflow: str, seed: int, n: int) -> list[dict]:
    """The first n of W's train cases in a fixed per-(workflow, seed) order: the same sample for knn,
    scratch, warm and full, and nested (the 16 are inside the 64)."""
    cases = load_cases(workflow, "train")
    order = sorted(cases, key=lambda c: c["id"])
    random.Random(f"{workflow}-{seed}").shuffle(order)
    return order[:n]


# --- round 4b: 12 families, 3 per workflow, split by a factor the dataset records ---
FAMILIES = [
    ("cs_billing", "customer_service", "topic",
     {"refund_status", "failed_payment", "invoice_copy", "unexpected_price", "duplicate_charge"},
     "Customer support about billing: refund status, a failed payment, an invoice copy, an unexpected price, or a duplicate charge."),
    ("cs_orders", "customer_service", "topic", {"wrong_item", "damaged_item", "late_delivery"},
     "Customer support about a physical order: a wrong item, a damaged item, or a late delivery."),
    ("cs_account", "customer_service", "topic",
     {"cannot_login", "data_export", "feature_broken", "cancel_account", "plan_downgrade", "slow_performance", "seat_management"},
     "Customer support about the account or product: login, data export, a broken feature, cancellation, a plan downgrade, slow performance, or seats."),
    ("inv_quantity", "invoice_processing", "issue", {"quantity_under", "quantity_over", "early_invoice"},
     "Invoice review where quantity or timing is off: fewer or more units billed than ordered or delivered, or billed before the goods arrived."),
    ("inv_pricing", "invoice_processing", "issue", {"price_variance", "tax_error", "currency_mismatch", "unauthorised_freight"},
     "Invoice review where the charge is off: a unit price that differs from the order, a wrong tax rate, a wrong currency, or a freight charge the order does not allow."),
    ("inv_process", "invoice_processing", "issue", {"duplicate_submission", "missing_po", "clean"},
     "Invoice review of process: a duplicate submission, a missing purchase order, or an invoice that reconciles cleanly."),
    ("sec_identity", "security_incidents", "rule",
     {"dormant_account_use", "new_country_login", "impossible_travel", "brute_force", "token_reuse"},
     "Security alert about sign-ins and credentials: dormant account use, a new-country login, impossible travel, brute force, or token reuse."),
    ("sec_privilege", "security_incidents", "rule", {"privilege_escalation", "disabled_logging", "config_drift"},
     "Security alert about privileges or configuration: privilege escalation, disabled logging, or configuration drift."),
    ("sec_data", "security_incidents", "rule", {"mass_download", "data_egress", "malware_signature", "suspicious_process"},
     "Security alert about data or hosts: a mass download, data egress, a malware signature, or a suspicious process."),
    ("agent_data", "agent_trace_observability", "task",
     {"Delete personal data for the accounts in the erasure queue.", "Backfill missing analytics events for the last seven days.",
      "Reconcile the billing ledger against the payment provider.", "Migrate the user table to the new schema without downtime."},
     "Agent run on a data task: deleting personal data, backfilling analytics events, reconciling the billing ledger, or migrating the user table."),
    ("agent_infra", "agent_trace_observability", "task",
     {"Rotate the expired TLS certificate on the staging load balancer.", "Provision a read replica for the reporting database.",
      "Update every service to the patched logging library.", "Reduce the nightly report job's runtime below fifteen minutes."},
     "Agent run on an infrastructure task: rotating a TLS certificate, provisioning a read replica, updating the logging library, or speeding up a report job."),
    ("agent_housekeeping", "agent_trace_observability", "task",
     {"Triage and close duplicate issues in the support tracker.", "Clean up stale feature branches merged more than 90 days ago."},
     "Agent run on a housekeeping task: closing duplicate issues in the support tracker or cleaning up stale branches."),
]
FAM_IDS = [f[0] for f in FAMILIES]
FAM_DESC = {f[0]: f[4] for f in FAMILIES}
FAM_WORKFLOW = {f[0]: f[1] for f in FAMILIES}
NONE_DESC = "No specialist listed here covers this kind of case."
ROUTER_Q = "choice: Which specialist should handle this case? Pick the one whose description fits the case."


def family_map() -> dict:
    """case id -> family id, for train and test cases of every workflow."""
    import json as _json
    import pandas as pd
    from common import TD
    out = {}
    for fid, w, key, vals, _ in FAMILIES:
        for split in ("train", "test"):
            df = pd.read_parquet(TD / w / f"{split}.parquet")
            for cid, fac in zip(df.id, df.factors):
                if _json.loads(fac)[key] in vals:
                    out[cid] = fid
    return out


def router_fold(i: int) -> tuple[str, str]:
    """Fold i: held-out family F and the family G whose cases train the `none` option."""
    return FAM_IDS[i], FAM_IDS[(i + 3) % len(FAM_IDS)]


def router_example(case: dict, options: list[str], label: str | None) -> dict:
    opts = [(o, FAM_DESC[o]) for o in options] + [("none", NONE_DESC)]
    names = [o for o, _ in opts]
    gold = [1.0 if o == label else 0.0 for o in names] if label is not None else [1.0 / len(names)] * len(names)
    return {"case": case["id"], "workflow": case["workflow"], "qname": "router", "type": "choice",
            "question": ROUTER_Q, "instructions": ROUTER_Q, "options": opts, "state": case["state"], "gold": gold}
