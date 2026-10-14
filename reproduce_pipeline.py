"""Reproduce SC-Logic longitudinal metrics directly from Fowler raw data.

The pipeline deliberately ignores ``datasets/``.  It reads ``dataverse_files.zip``,
uses SH.sav to define a time-invariant, classified Senate actor universe, filters
strictly to SN/SP records, constructs monthly cumulative 2-skeleta, computes
unweighted party-partition Newman-Girvan modularity, and audits stage-indexed
Heyting negation with explicit face identifiers.

CI_A is the Republican-core contraction index.  A party-core vertex is an actor
appearing in at least three distinct qualifying bills by the current stage.  A
qualifying bill has a sponsor in that party and no classified cosponsor from the
other party.  In the subcomplex lattice, negation of the generated core support
is the induced subcomplex on vertices outside the core support.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pyreadstat


LOGGER = logging.getLogger("sc_logic_reproduction")
TARGET_CONGRESSES = (104, 107, 108)
CLASSIFIED_PARTIES = {100: "D", 200: "R"}
BILL_PATTERN = re.compile(r"^(SN|SP)_(\d{3})_(.+)\.txt$")


@dataclass(frozen=True)
class Bill:
    bill_id: str
    congress: int
    introduced: date
    sponsor: int
    participants: tuple[int, ...]
    sponsor_party: int
    qualifying_party: int | None


def month_end(year: int, month: int) -> date:
    """Return the final calendar date of a month."""
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1).fromordinal(date(year, month + 1, 1).toordinal() - 1)


def congress_months(congress: int) -> list[tuple[str, date]]:
    """Return exactly 24 monthly stages for the two-year Congress."""
    start_year = 1787 + 2 * congress
    return [
        (f"{year:04d}-{month:02d}", month_end(year, month))
        for year in (start_year, start_year + 1)
        for month in range(1, 13)
    ]


def extract_raw_archive(archive: Path, raw_dir: Path) -> None:
    """Extract the authoritative raw archive into an isolated output directory."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)
    LOGGER.info("Extracted %d raw artifacts into %s", len(list(raw_dir.iterdir())), raw_dir)


def read_lines(path: Path) -> list[str]:
    """Read a Fowler vector file, preserving one record per physical line."""
    return path.read_text(encoding="latin-1").splitlines()


def load_actor_universes(raw_dir: Path) -> dict[int, dict[int, int]]:
    """Load classified Senate actors from SH.sav with invariant Congress cohorts."""
    frame, _ = pyreadstat.read_sav(str(raw_dir / "SH.sav"))
    universes: dict[int, dict[int, int]] = {}
    for congress in TARGET_CONGRESSES:
        subset = frame[(frame["congress"] == congress) & (frame["chamber"] == "S")]
        actors: dict[int, int] = {}
        for row in subset.itertuples(index=False):
            if row.ids != row.ids or row.party != row.party:
                continue
            actor_id, party = int(row.ids), int(row.party)
            if party not in CLASSIFIED_PARTIES:
                continue
            if actor_id in actors and actors[actor_id] != party:
                raise ValueError(f"Conflicting party assignments for {congress}:{actor_id}")
            actors[actor_id] = party
        if not actors:
            raise ValueError(f"No classified Senate actors found for Congress {congress}")
        universes[congress] = actors
        LOGGER.info(
            "Congress %d actor universe: %d (%d D, %d R)",
            congress,
            len(actors),
            sum(p == 100 for p in actors.values()),
            sum(p == 200 for p in actors.values()),
        )
    return universes


def load_bills(raw_dir: Path, universes: dict[int, dict[int, int]]) -> tuple[dict[int, list[Bill]], dict]:
    """Parse aligned Fowler vectors and return valid SN/SP Senate bill events."""
    names = read_lines(raw_dir / "Bills.txt")
    dates = read_lines(raw_dir / "Dates.txt")
    sponsors = read_lines(raw_dir / "Sponsors.txt")
    cosponsors = read_lines(raw_dir / "Cosponsors.txt")
    lengths = {len(names), len(dates), len(sponsors), len(cosponsors)}
    if len(lengths) != 1:
        raise ValueError(f"Aligned raw vectors have unequal lengths: {lengths}")

    by_congress = {c: [] for c in TARGET_CONGRESSES}
    diagnostics = {c: Counter() for c in TARGET_CONGRESSES}
    for bill_name, raw_date, raw_sponsor, raw_cosponsors in zip(
        names, dates, sponsors, cosponsors, strict=True
    ):
        match = BILL_PATTERN.match(bill_name.strip())
        if not match:
            continue
        congress = int(match.group(2))
        if congress not in by_congress:
            continue
        diag = diagnostics[congress]
        diag["raw_sn_sp"] += 1
        if raw_date == "NA" or raw_sponsor == "NA":
            diag["missing_date_or_sponsor"] += 1
            continue
        try:
            introduced = datetime.strptime(raw_date, "%Y-%m-%d").date()
            sponsor = int(float(raw_sponsor))
        except ValueError:
            diag["invalid_date_or_sponsor"] += 1
            continue
        period = congress_months(congress)
        if not (date.fromisoformat(period[0][0] + "-01") <= introduced <= period[-1][1]):
            diag["outside_congress_window"] += 1
            continue
        actors = universes[congress]
        if sponsor not in actors:
            diag["sponsor_outside_classified_universe"] += 1
            continue
        raw_ids: list[int] = []
        if raw_cosponsors != "NA":
            for token in raw_cosponsors.split():
                try:
                    raw_ids.append(int(float(token)))
                except ValueError:
                    diag["invalid_cosponsor_id"] += 1
        classified_cosponsors = {actor for actor in raw_ids if actor in actors and actor != sponsor}
        diag["unclassified_cosponsor_mentions"] += sum(actor not in actors for actor in raw_ids)
        participants = tuple(sorted({sponsor, *classified_cosponsors}))
        sponsor_party = actors[sponsor]
        qualifying_party = (
            sponsor_party
            if all(actors[actor] == sponsor_party for actor in classified_cosponsors)
            else None
        )
        by_congress[congress].append(
            Bill(
                bill_id=bill_name.removesuffix(".txt"),
                congress=congress,
                introduced=introduced,
                sponsor=sponsor,
                participants=participants,
                sponsor_party=sponsor_party,
                qualifying_party=qualifying_party,
            )
        )
        diag["included_bills"] += 1
        if qualifying_party == 100:
            diag["qualifying_D_bills"] += 1
        elif qualifying_party == 200:
            diag["qualifying_R_bills"] += 1

    for congress in TARGET_CONGRESSES:
        by_congress[congress].sort(key=lambda bill: (bill.introduced, bill.bill_id))
    return by_congress, {str(c): dict(v) for c, v in diagnostics.items()}


def add_face_births(
    participants: tuple[int, ...],
    stage: int,
    face_birth: dict[tuple[int, ...], int],
    edge_weights: Counter[tuple[int, int]],
) -> None:
    """Add edges/triangles and retain the first stage of every explicit face ID."""
    for edge in itertools.combinations(participants, 2):
        edge_weights[edge] += 1
        face_birth.setdefault(edge, stage)
    for triangle in itertools.combinations(participants, 3):
        face_birth.setdefault(triangle, stage)


def modularity_unweighted(edges: Iterable[tuple[int, int]], parties: dict[int, int]) -> tuple[float, int, int]:
    """Compute party-partition Newman-Girvan modularity on the unweighted 1-skeleton."""
    edge_list = list(edges)
    m = len(edge_list)
    if m == 0:
        return 0.0, 0, 0
    degree = Counter()
    within = Counter()
    same_party = 0
    for left, right in edge_list:
        degree[left] += 1
        degree[right] += 1
        if parties[left] == parties[right]:
            within[parties[left]] += 1
            same_party += 1
    degree_sum = {
        party: sum(degree[actor] for actor, actor_party in parties.items() if actor_party == party)
        for party in CLASSIFIED_PARTIES
    }
    q_value = sum(
        within[party] / m - (degree_sum[party] / (2 * m)) ** 2
        for party in CLASSIFIED_PARTIES
    )
    return q_value, same_party, m - same_party


def negation_faces(
    face_birth: dict[tuple[int, ...], int], stage: int, core_vertices: set[int]
) -> set[tuple[int, ...]]:
    """Return explicit faces in K_stage induced by vertices outside the proposition core."""
    return {
        face
        for face, birth_stage in face_birth.items()
        if birth_stage <= stage and core_vertices.isdisjoint(face)
    }


def face_counts(faces: Iterable[tuple[int, ...]]) -> tuple[int, int, int]:
    """Count vertices, edges, and triangles in an explicit 2-skeleton face set."""
    counts = Counter(len(face) for face in faces)
    return counts[1], counts[2], counts[3]


def compute_congress(
    congress: int,
    actors: dict[int, int],
    bills: list[Bill],
) -> tuple[list[dict], list[dict], dict]:
    """Compute 24 monthly stages and a strict-pair antitonicity audit."""
    stages = congress_months(congress)
    face_birth: dict[tuple[int, ...], int] = {(actor,): 1 for actor in actors}
    edge_weights: Counter[tuple[int, int]] = Counter()
    qualifying_counts = {100: Counter(), 200: Counter()}
    neg_snapshots: dict[int, list[set[tuple[int, ...]]]] = {100: [], 200: []}
    rows: list[dict] = []
    bill_cursor = 0
    cumulative_bills = 0

    for stage, (month, cutoff) in enumerate(stages, start=1):
        bills_this_month = 0
        while bill_cursor < len(bills) and bills[bill_cursor].introduced <= cutoff:
            bill = bills[bill_cursor]
            bill_cursor += 1
            bills_this_month += 1
            cumulative_bills += 1
            add_face_births(bill.participants, stage, face_birth, edge_weights)
            if bill.qualifying_party in CLASSIFIED_PARTIES:
                qualifying_counts[bill.qualifying_party].update(set(bill.participants))

        active_faces = {face for face, birth in face_birth.items() if birth <= stage}
        f0, f1, f2 = face_counts(active_faces)
        total_faces = f0 + f1 + f2
        edges = [face for face in active_faces if len(face) == 2]
        q_value, same_party_edges, cross_party_edges = modularity_unweighted(edges, actors)
        row = {
            "congress": congress,
            "stage": stage,
            "month": month,
            "cutoff_date": cutoff.isoformat(),
            "actor_count": len(actors),
            "democratic_actors": sum(p == 100 for p in actors.values()),
            "republican_actors": sum(p == 200 for p in actors.values()),
            "bills_this_month": bills_this_month,
            "cumulative_bills": cumulative_bills,
            "f0": f0,
            "f1": f1,
            "f2": f2,
            "total_faces": total_faces,
            "same_party_edges": same_party_edges,
            "cross_party_edges": cross_party_edges,
            "Q_party_unweighted": q_value,
        }
        for party, suffix in ((200, "R"), (100, "D")):
            core = {actor for actor, count in qualifying_counts[party].items() if count >= 3}
            neg = negation_faces(face_birth, stage, core)
            neg_snapshots[party].append(neg)
            n0, n1, n2 = face_counts(neg)
            neg_total = n0 + n1 + n2
            row.update(
                {
                    f"core_{suffix}_vertices": len(core),
                    f"neg_{suffix}_f0": n0,
                    f"neg_{suffix}_f1": n1,
                    f"neg_{suffix}_f2": n2,
                    f"neg_{suffix}_total_faces": neg_total,
                    f"CI_{suffix}": neg_total / total_faces if total_faces else 0.0,
                }
            )
        row["CI_A"] = row["CI_R"]
        rows.append(row)

    audit_rows: list[dict] = []
    for party, proposition in ((200, "p_R"), (100, "p_D")):
        snapshots = neg_snapshots[party]
        for s_idx in range(24):
            neg_s = snapshots[s_idx]
            for t_idx in range(s_idx + 1, 24):
                neg_t = snapshots[t_idx]
                intersection = {face for face in neg_t if face_birth[face] <= s_idx + 1}
                violations = intersection - neg_s
                audit_rows.append(
                    {
                        "congress": congress,
                        "proposition": proposition,
                        "s_stage": s_idx + 1,
                        "t_stage": t_idx + 1,
                        "s_month": stages[s_idx][0],
                        "t_month": stages[t_idx][0],
                        "intersection_faces_checked": len(intersection),
                        "neg_s_faces": len(neg_s),
                        "neg_t_faces": len(neg_t),
                        "violation_faces": len(violations),
                    }
                )
                if violations:
                    raise AssertionError(
                        f"Antitonicity violation in Congress {congress}, {proposition}, "
                        f"stages {s_idx + 1}<{t_idx + 1}: {len(violations)} faces"
                    )

    summary = {
        "congress": congress,
        "actors": len(actors),
        "democrats": sum(p == 100 for p in actors.values()),
        "republicans": sum(p == 200 for p in actors.values()),
        "included_bills": len(bills),
        "terminal": rows[-1],
        "audit_pairs_per_proposition": 276,
        "audit_rows": len(audit_rows),
        "face_inclusions_checked": sum(r["intersection_faces_checked"] for r in audit_rows),
        "violations": sum(r["violation_faces"] for r in audit_rows),
    }
    return rows, audit_rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write dictionaries with stable field order."""
    if not rows:
        raise ValueError(f"No rows available for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("dataverse_files.zip"))
    parser.add_argument("--output", type=Path, default=Path("generated_results"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s %(message)s")

    raw_dir = args.output / "raw_extracted"
    extract_raw_archive(args.archive, raw_dir)
    universes = load_actor_universes(raw_dir)
    bills_by_congress, ingestion = load_bills(raw_dir, universes)

    all_rows: list[dict] = []
    all_audit_rows: list[dict] = []
    summaries: list[dict] = []
    for congress in TARGET_CONGRESSES:
        rows, audit_rows, summary = compute_congress(
            congress, universes[congress], bills_by_congress[congress]
        )
        all_rows.extend(rows)
        all_audit_rows.extend(audit_rows)
        summaries.append(summary)
        LOGGER.info(
            "Congress %d complete: %d bills, terminal CI_A=%.10f, Q=%.10f, violations=%d",
            congress,
            summary["included_bills"],
            summary["terminal"]["CI_A"],
            summary["terminal"]["Q_party_unweighted"],
            summary["violations"],
        )

    write_csv(args.output / "reproduced_cross_congress_ci.csv", all_rows)
    write_csv(args.output / "reproduced_antitonicity_audit.csv", all_audit_rows)
    (args.output / "ingestion_diagnostics.json").write_text(
        json.dumps(ingestion, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output / "phase1_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\nPHASE 1 EMPIRICAL SUMMARY")
    print("Congress | Actors (D/R) | Included SN/SP bills | f0/f1/f2 | Terminal CI_A | Terminal Q | Audit rows | Face inclusions | Violations")
    for summary in summaries:
        terminal = summary["terminal"]
        print(
            f"{summary['congress']:>8} | {summary['actors']} ({summary['democrats']}/{summary['republicans']}) "
            f"| {summary['included_bills']:>21} | {terminal['f0']}/{terminal['f1']}/{terminal['f2']} "
            f"| {terminal['CI_A']:.10f} | {terminal['Q_party_unweighted']:.10f} "
            f"| {summary['audit_rows']} | {summary['face_inclusions_checked']} | {summary['violations']}"
        )
    print(f"\nWrote: {args.output / 'reproduced_cross_congress_ci.csv'}")
    print(f"Wrote: {args.output / 'reproduced_antitonicity_audit.csv'}")


if __name__ == "__main__":
    main()
