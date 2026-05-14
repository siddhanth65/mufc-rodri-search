# MUFC Rodri Search

# Engineering Manchester United's Midfield: A Data-Driven Search for a Defensive-Midfield Anchor

A four-stage scouting pipeline that profiles every midfielder in Europe's top seven
leagues and ranks candidates to complete a midfield trio with Kobbie Mainoo and
Bruno Fernandes.

## The Question

Manchester United's midfield problem is structural, not a depth issue. The team
lacks a specific archetype, a defensive-midfield anchor who wins the ball ahead
of the back line, recycles possession safely, and builds from deep. Mainoo is a
ball-progressing No. 8; Bruno operates too high to be the defensive pivot. This
project defines that missing archetype statistically and surfaces the players in
Europe who fit it, ranked by how well they complement the two midfielders United
already have.

The output is not a recommendation engine. It is a diagnostic: a disciplined,
reproducible answer to one concrete recruitment question.

## Approach

The pipeline is a funnel. Each stage narrows the pool and adds a layer of
context, and every player keeps a score on every dimension so the ranking is
explainable rather than a black box.

**Stage 0 — Population.** All midfielders from the top seven European leagues
(Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie, Primeira
Liga) across three seasons. Counting stats are summed across seasons and divided
by total 90s; rate stats are minutes-weighted. Players split into a main track
(>= 2000 minutes) and a prospect track (600-2000 minutes) so low-sample players
are surfaced separately rather than polluting the main ranking.

**Stage 1 — Archetype filter.** Each player is described by a 20-feature
behavioural fingerprint covering passing volume and security, defensive
ball-winning, territory, and creation restraint. Features are z-scored within the
main-track population and weighted. Similarity to the archetype is measured as
the minimum weighted Euclidean distance to any of several seed players — using
minimum-to-any-seed rather than distance-to-a-mean because the defensive-midfield
archetype is genuinely multi-modal (a ball-winner and a deep distributor are both
valid). The four defensive ball-winning features use a one-sided distance: a
player is penalised for falling below seed level but not for exceeding it, since
more ball-winning is monotonically good within this role.

**Stage 2 — Complementarity ranking.** The top candidates are re-ranked by fit to
United specifically. The combined feature vector of Mainoo and Bruno defines an
"existing midfield" profile; the gap vector is where that pair sits below the
population average. Candidates are scored by how much of that gap they cover. The
final Stage 2 score blends archetype fit and complementarity, with a soft
league-strength modifier.

**Stage 3 — Practical filters.** Transfermarkt data (age, market value, contract
expiry) is layered on as soft scoring modifiers, not hard cuts. A 24-year-old on
an expiring contract scores better than an equivalent player locked up until 2031.

**Stage 4 — Profiles.** For the final shortlist, the pipeline generates a
percentile readout on the key role metrics (each player scored 0-100 against the
main-track population) and an auto-generated scouting note tying together
archetype fit, the specific gaps the player covers, and their practical
situation.

## Methodology Notes

Statistical rigour was the priority throughout. A few decisions worth calling out:

- **Pre-registered validation.** A set of known players — strong-fit, weak-fit,
  and negative controls — was written down _before_ running the model. It is the
  central guard against tuning the pipeline until it agrees with intuition.
  `NOTES.md` contains the set and the full revision log.
- **Z-scores for the model, percentiles for display.** Distance maths runs on
  z-scores because they preserve magnitude linearly; percentiles are computed
  alongside purely for legibility in the output.
- **Possession-adjusted defensive metrics.** Tackles, interceptions, blocks and
  recoveries are adjusted for opponent possession share, standard practice to
  avoid rewarding players simply for being on lower-possession teams.
- **Every methodology change is logged.** The distance metric, feature set and
  seed list each went through revisions; `NOTES.md` records why each change was
  made so the project can be reconstructed and critiqued.

## Data

All data is scraped via `ScraperFC` (Sofascore for performance data,
Transfermarkt for age, value and contract). The original plan used FBref; that
source removed its advanced stats mid-build, and the pipeline was rebuilt on
Sofascore — a resilience exercise that is itself documented in `NOTES.md`. Raw
pulls are cached as parquet so the pipeline is reproducible without re-scraping.

## Repository Structure

```
src/
  sofascore_pipeline.py       Stage 0-1: data pull, aggregation, fingerprint, archetype filter
  stage2_complementarity.py   Stage 2: complementarity ranking vs Mainoo + Bruno
  stage3_practical_filters.py Stage 3: Transfermarkt age / value / contract modifiers
  stage4_profiles.py          Stage 4: percentile readouts + scouting notes
data/
  sofascore/                  Cached raw league-season pulls
  processed/                  Pipeline outputs (parquet)
FEATURES.md                   The 20-feature spec and the rationale for each choice
NOTES.md                      Pre-registered validation set + full revision log
```

## Running It

```
python src/sofascore_pipeline.py --all     # pull + build fingerprint + Stage 1
python src/stage2_complementarity.py       # Stage 2
python src/stage3_practical_filters.py     # Stage 3 (scrapes Transfermarkt)
python src/stage4_profiles.py              # Stage 4 profiles
```

Sofascore opens a real browser window per request to clear bot protection, so the
initial pull is not instant; subsequent runs read from cache.

## Status & Limitations

The pipeline runs end to end and produces a validated shortlist. Known
limitations: the current season is partial and aggregated with the completed
seasons (impact assessed and documented in `NOTES.md`); positional data is from
the season aggregate, so players who changed roles across the window have a
blended fingerprint that may not reflect their best position; and the model
describes statistical fit, not the contextual judgement a human scout adds.
