#!/usr/bin/env python3
"""Export CPU cost tables from saved counters into a new analysis directory."""
import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

from workspace import ROOT, current_batch, io_resources, now, write_json

METHODS = ("CATALOG_SHARED_GBFS", "LAZY_SHARED_GBFS", "LAZY_CERT_FIRST",
           "DECISION_SCHEDULER", "ABLATION_EAGER", "ABLATION_NO_CERT", "ABLATION_FIXED_ORDER")
SCALES = (16, 64, 256, 1024)


def validate_rows(rows, expected_layouts=None, scales=SCALES):
    expected_layouts = expected_layouts or {"calibration": 64, "test": 512}
    if not rows:
        raise ValueError("Empty CPU result table")
    fields = list(rows[0])
    required = {"split", "cluster_id", "candidate_count", "method", "audit", "decision",
                "expected_decision", "cost_proxy", "world_sha256", "obstruction", "designed_feasible",
                "candidate_materializations", "cpu_seconds_throughput", "geometry_cache_hits"}
    if required - set(fields):
        raise ValueError(f"Missing CPU columns: {sorted(required - set(fields))}")
    counters = fields[fields.index("candidate_materializations"):fields.index("cpu_seconds_throughput")]
    counters = [name for name in counters if name != "geometry_cache_hits"]
    if not counters:
        raise ValueError("No operation counters in expected column interval")
    paired = {}
    for index, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV row {index}")
        split, method = row["split"], row["method"]
        scale = int(row["candidate_count"])
        if split not in expected_layouts or method not in METHODS or scale not in scales:
            raise ValueError(f"Unexpected split, method or scale at row {index}")
        if row["audit"] != "PASS" or row["decision"] != row["expected_decision"]:
            raise ValueError(f"Failed saved decision audit at row {index}")
        counts = [int(row[name]) for name in counters]
        if any(value < 0 for value in counts) or sum(counts) != int(row["cost_proxy"]):
            raise ValueError(f"Invalid operation-count sum at row {index}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["world_sha256"]):
            raise ValueError(f"Invalid world digest at row {index}")
        if row["obstruction"] not in {"shared", "unshared"} or int(row["designed_feasible"]) not in range(3):
            raise ValueError(f"Unexpected construction stratum at row {index}")
        key = (split, scale, row["cluster_id"])
        group = paired.setdefault(key, {})
        if method in group:
            raise ValueError(f"Duplicate paired observation: {key}, {method}")
        group[method] = row
    for key, group in paired.items():
        if set(group) != set(METHODS):
            raise ValueError(f"Incomplete method pairing: {key}")
        descriptors = {(r["world_sha256"], r["obstruction"], r["designed_feasible"], r["expected_decision"])
                       for r in group.values()}
        if len(descriptors) != 1:
            raise ValueError(f"Mismatched paired world or reference: {key}")
        if int(group["LAZY_SHARED_GBFS"]["cost_proxy"]) <= 0 or int(group["DECISION_SCHEDULER"]["cost_proxy"]) <= 0:
            raise ValueError(f"Nonpositive cost denominator: {key}")
    split_ids = {}
    for split, count in expected_layouts.items():
        ids = [{cluster for s, n, cluster in paired if s == split and n == scale} for scale in scales]
        if any(len(group) != count or group != ids[0] for group in ids):
            raise ValueError(f"Incomplete or unpaired layout scales: {split}")
        split_ids[split] = ids[0]
    if len(set().union(*split_ids.values())) != sum(map(len, split_ids.values())):
        raise ValueError("Layout identifiers overlap across splits")
    return paired


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, output / "costs_and_decisions.csv")
    shutil.copy2(__file__, output / "cpu_reporting.py")
    shutil.copy2(Path(__file__).with_name("workspace.py"), output / "workspace.py")
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    validated = validate_rows(rows)
    test = [r for r in rows if r["split"] == "test"]
    methods = list(METHODS)
    names = ["Catalog shared", "Lazy shared", "Certificate first", "Decision scheduler",
             "Eager ablation", "No-certificate ablation", "Fixed-order ablation"]
    scales = list(SCALES)
    paired = {(scale, cluster): group for (split, scale, cluster), group in validated.items() if split == "test"}


    def percentile(values, p):
        values = sorted(values)
        position = (len(values) - 1) * p
        i = int(position)
        fraction = position - i
        return values[i] * (1 - fraction) + values[min(i + 1, len(values) - 1)] * fraction


    def summary(groups, method, reference):
        numerator = [int(g[method]["cost_proxy"]) for g in groups]
        denominator = [int(g[reference]["cost_proxy"]) for g in groups]
        if not denominator or any(d <= 0 for d in denominator):
            raise ValueError("A paired ratio needs positive denominators")
        ratios = [n / d for n, d in zip(numerator, denominator)]
        return dict(layouts=len(groups), paired_median=statistics.median(ratios),
                    ratio_of_means=sum(numerator) / sum(denominator),
                    paired_p90=percentile(ratios, .9),
                    more_work_fraction=sum(n > d for n, d in zip(numerator, denominator)) / len(groups))


    def write_csv(name, records):
        with (output / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


    distribution, ablations, strata = [], [], []
    for n in scales:
        groups = [g for (scale, _), g in sorted(paired.items()) if scale == n]
        for method in methods:
            distribution.append(dict(candidate_count=n, method=method, reference="LAZY_SHARED_GBFS",
                                     **summary(groups, method, "LAZY_SHARED_GBFS")))
        for method in methods[4:]:
            ablations.append(dict(candidate_count=n, method=method, reference="DECISION_SCHEDULER",
                                  **summary(groups, method, "DECISION_SCHEDULER")))
        for obstruction in ("shared", "unshared"):
            for feasible in range(3):
                subset = [g for g in groups if g[methods[0]]["obstruction"] == obstruction
                          and int(g[methods[0]]["designed_feasible"]) == feasible]
                if not subset:
                    raise ValueError("Missing construction stratum")
                for method in methods:
                    strata.append(dict(candidate_count=n, obstruction=obstruction, designed_feasible=feasible,
                                       method=method, reference="LAZY_SHARED_GBFS",
                                       **summary(subset, method, "LAZY_SHARED_GBFS")))
    write_csv("distribution_all_scales.csv", distribution)
    write_csv("ablations_vs_scheduler_all_scales.csv", ablations)
    write_csv("stratified_all_scales.csv", strata)

    table = [r"\begin{table}[tbp]", r"\centering\small",
             r"\caption{CPU operation-proxy ratios on 512 paired test layouts at 64 candidates. Panel A uses lazy shared checking as the reference; Panel B compares each ablation with the decision scheduler. Median and P90 are quantiles of per-layout ratios; mean ratio is the ratio of total work (equivalently, of means). More work is the percentage of layouts exceeding the reference. Heterogeneous operations receive unit weight; these are not wall-time ratios.}",
             r"\label{tab:cpu-ratios}", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Method & Median & Mean ratio & P90 & More work (\%) \\", r"\midrule",
             r"\multicolumn{5}{@{}l}{\textit{A. Reference: lazy shared checking}} \\"]
    for method, name in zip(methods[:4], names[:4]):
        r = next(r for r in distribution if r["candidate_count"] == 64 and r["method"] == method)
        table.append(f"{name} & {r['paired_median']:.3f} & {r['ratio_of_means']:.3f} & {r['paired_p90']:.3f} & {100*r['more_work_fraction']:.1f} " + r"\\")
    table += [r"\midrule", r"\multicolumn{5}{@{}l}{\textit{B. Reference: decision scheduler}} \\"]
    for method, name in zip(methods[4:], names[4:]):
        r = next(r for r in ablations if r["candidate_count"] == 64 and r["method"] == method)
        table.append(f"{name} & {r['paired_median']:.3f} & {r['ratio_of_means']:.3f} & {r['paired_p90']:.3f} & {100*r['more_work_fraction']:.1f} " + r"\\")
    table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output / "cpu_ratios.tex").write_text("\n".join(table) + "\n")

    table = [r"\begin{table}[tbp]", r"\centering\small",
             r"\caption{Paired median ratios to lazy shared checking, stratified by obstruction construction and the designed number of feasible goals ($f$), at 64 candidates. The 512 layouts are the same as in Table~\ref{tab:cpu-ratios}; strata are descriptive, not additional independent samples.}",
             r"\label{tab:cpu-strata}", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Obstruction & $f$ & Layouts & Scheduler & No certificate & Fixed order \\", r"\midrule"]
    for obstruction in ("shared", "unshared"):
        for feasible in range(3):
            subset = [r for r in strata if r["candidate_count"] == 64 and r["obstruction"] == obstruction
                      and r["designed_feasible"] == feasible]
            by_method = {r["method"]: r for r in subset}
            values = [by_method[m]["paired_median"] for m in (methods[3], methods[5], methods[6])]
            table.append(f"{obstruction.capitalize()} & {feasible} & {subset[0]['layouts']} & "
                         + " & ".join(f"{v:.3f}" for v in values) + r" \\")
    table += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (output / "cpu_strata.tex").write_text("\n".join(table) + "\n")

    sha256 = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = dict(at=datetime.now(timezone.utc).isoformat(), status="PASS",
                   evidence="reaggregation of saved counters; not new experiment execution",
                   source=str(source), source_sha256=sha256(source), raw_rows=len(rows),
                   test_rows=len(test), independent_test_layouts=512, scales=scales,
                   representative_table_scale=64, percentile="linear interpolation at (n-1)*p",
                   checks=["saved audit PASS", "saved decisions match expected decisions",
                           "cost proxy reconstructed from saved counters", "complete paired method/scale matrix",
                           "within-scale world hashes and strata match across methods"],
                   outputs={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "output": str(output), "test_rows": len(test),
                      "independent_test_layouts": 512, "model_inference": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Saved costs_and_decisions.csv; defaults to current imported batch")
    parser.add_argument("--output", type=Path, help="New directory under work/")
    args = parser.parse_args()
    if args.source:
        source = args.source.resolve()
    else:
        batch, _ = current_batch()
        candidates = sorted((batch / "cpu").glob("analysis_*/costs_and_decisions.csv"))
        if len(candidates) != 1:
            parser.error("Expected one saved CPU analysis; select --source explicitly")
        source = candidates[0]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.output.resolve() if args.output else ROOT / "work/cpu_reporting" / stamp
    if not out.is_relative_to(ROOT / "work") or out.exists():
        parser.error("Output must be a new directory under work/")
    resource = io_resources()
    try:
        run(source, out)
        write_json(out / "resource_plan.json", {**resource, "workers": 1,
                   "note": "A single saved CSV is reduced serially; no performance measurement."})
        write_json(ROOT / "work/latest_cpu_reporting.json", {"status": "PASS", "directory": str(out.relative_to(ROOT))})
    except BaseException as exc:
        if out.exists():
            write_json(out / "failure.json", {"status": "FAIL", "at": now(), "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
