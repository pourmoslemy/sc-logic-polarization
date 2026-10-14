# Beyond Dyadic Polarization: SC-Logic and Graded Negation in Legislative Coalitions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22538544.svg)](https://doi.org/10.5281/zenodo.22538544)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

This repository contains the standalone replication package and empirical pipeline for the paper:  
**"Beyond Dyadic Polarization: SC-Logic and Graded Negation in Legislative Coalitions"** (Alireza Pourmoslemi).

---

## Theoretical & Methodological Overview

Standard network-theoretic models reduce legislative co-sponsorship dynamics to dyadic graphs, conflating multi-agent consensus into pairs of independent edges and interpreting opposition through classical Boolean negation (the Law of Excluded Middle). 

**Simplicial Complex Logic (SC-Logic)** lifts dynamic legislative coalitions into time-indexed simplicial complexes ($\mathcal{K}_t$). Key structural properties include:
1. **Higher-Order Coalitions:** Co-sponsorship groups are modeled as abstract simplices rather than edge cliques, preserving irreducible polyadic consensus.
2. **Graded Heyting Negation:** In the intuitionistic lattice of subcomplexes, negation does not collapse to Boolean complements. Instead, it captures intermediate structural dissent, formalizing partial opposition and *conscientious objection*.
3. **Antitone Restriction (Proposition 1):** Proves that the graded negation operator $\neg_t$ is strictly antitone under inclusion ($p_s \subseteq p_t \implies \neg_t(p_t) \subseteq \neg_t(p_s)$) across all cumulative temporal filtration stages.

---

## Raw Data Requirements

The pipeline strictly guarantees reproducibility from raw primary sources without reliance on pre-filtered or pre-processed intermediate files.

- **Primary Archive:** James Fowler (Harvard Dataverse) — *Cosponsorship in the United States Senate*
- **Persistent Identifier (Dataverse DOI):** [10.7910/DVN/JMZS8E](https://doi.org/10.7910/DVN/JMZS8E)
- **Required File:** `dataverse_files.zip` (placed in the root directory).

The pipeline automatically inspects `SH.sav`, extracts qualifying legislative acts (`SN`/`SP` records), constructs dynamic 2-skeleta, and models stage-by-stage party dynamics across target Congresses (104, 107, 108).

---

## Installation & Environment Setup

Clone this repository and install the minimal dependencies:
```bash
git clone https://github.com/pourmoslemy/SC-Logic-Legislative-Coalitions.git
cd SC-Logic-Legislative-Coalitions
pip install -r requirements.txt
