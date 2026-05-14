# FEATURES.md — Style Fingerprint Specification

**Status:** v0.2 — reflects actual Sofascore implementation  
**Last updated:** 2026-05-14  
**Save location:** `data/FEATURES.md`

---

## What changed from v0.1

v0.1 was a pre-data-pull spec against FBref (27 features). The actual implementation uses Sofascore player-statistics, which exposes a different column set. The feature count dropped from 27 to 20. Every dropped or swapped feature is documented in the revision log below.

---

## Design principles

1. **Each feature must be role-discriminating.** Features that don't separate #6s from #8s and #10s add noise, not signal.
2. **No feature without a mathematical definition.** "Tempo control" is not a feature; "passes per touch" is.
3. **Defensive volume features get PAdj and one-sided distance.** Attacking/passing features do not.
4. **Z-score normalization within the filtered midfielder population.** Not percentile rank — z-scores preserve magnitude linearly, which Euclidean distance needs.
5. **No more features than discriminate the role.** 20 discriminating beats 40 noisy.

---

## Population

- Sofascore `selected_positions=["Midfielders"]` — Sofascore's own position classification
- Minutes ≥ 2000 (main track) across the aggregation window for ranking; 600–2000 is the prospect track
- Aggregation: **minutes-proportional across 23/24, 24/25, and 25/26** — see normalization section
- Source leagues: Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie, Primeira Liga

---

## Aggregation and normalization pipeline

Applied in order:

1. **Accumulate counts across seasons.** For `count` and `count_flip` features, sum raw totals across all season-rows for a player. For `rate` features, compute a minutes-weighted mean: `Σ(rate_s × min_s) / Σ(min_s)`.
2. **Per-90.** Count totals divided by `total_minutes / 90`. This is mathematically equivalent to minutes-proportional weighting — a season contributing 500 of 2500 total minutes contributes 20% to the per-90 rate. No separate season-level weighting is applied or needed.
3. **PAdj for defensive volume.** For tackles, interceptions, blocks, recoveries:  
   `PAdj_x = x_per_90 / OPP_POSS`  
   where `OPP_POSS = 0.48` (approximates opponent's share of the ball). Normalizes for teams that give up more or fewer defensive opportunities.
4. **Sign flip for `count_flip` features** (shots, xG, dispossessed, possession_lost). Before z-scoring, negate: `f = -raw_per_90`. A player who rarely shoots or loses the ball gets a high (good) value, so distance is correctly signed throughout.
5. **Z-score within the main-track population (≥2000 min).** Mean 0, SD 1. Prospect track is z-scored against the main-track distribution so both tracks share the same scale.

---

## One-sided distance for defensive features

For `{tackles, interceptions, recoveries, blocks}`, the distance formula clips positive differences to zero:

```
adj_diff = player_z - seed_z
if feature in ONE_SIDED_FEATURES:
    adj_diff = min(adj_diff, 0)   # only penalise shortfall, not excess
```

**Rationale:** A player who out-tackles and out-intercepts the seed archetype is not "wrong" — more defensive volume is monotonically better for the role. Only falling below the seed archetype matters. Without this, the distance metric would penalise Baleba for being too good defensively, which is incoherent.

This applies only to the four count-based defensive features, not to `tackles_won_pct` (a rate that could go either direction) or `aerials_won_pct`.

---

## Similarity metric (Stage 1)

**Weighted Euclidean on z-scored features with one-sided defensive distance.**  
`distance_to_seed = sqrt( Σ_i (w_i × adj_diff_i)^2 )`

For each player, distance is computed against every seed; the minimum distance (closest seed) is the player's archetype score.

Cosine similarity is **not used.** Cosine is direction-only — it cannot distinguish "low-volume proper 6" from "high-volume proper 6" on z-scored features, which is exactly the distinction the archetype filter must make.

---

## Feature catalog (20)

Format: `feature_name` — Sofascore API field — type — weight — normalization — rationale

### Passing volume & accuracy (5)

| Feature | Sofascore field | Type | Weight | Notes |
|---|---|---|---|---|
| `passes_total` | `totalPasses` | count | 1.0 | Volume signal — high for ball-dominant midfielders |
| `pass_cmp_pct` | `accuratePassesPercentage` | rate | 1.0 | Separates clean recyclers from spray-passers |
| `long_ball_cmp_pct` | `accurateLongBallsPercentage` | rate | 1.0 | Switch and diagonal capability |
| `passes_final_third` | `accurateFinalThirdPasses` | count | 1.0 | Progressive delivery without carrying |
| `own_half_passes` | `totalOwnHalfPasses` | count | **1.5** | Deep-lying recycler signature — discriminates true #6 from #8 |

### Defensive volume — PAdj + one-sided (4)

| Feature | Sofascore field | Type | Weight | Notes |
|---|---|---|---|---|
| `tackles` | `tackles` | count | **1.5** | PAdj + one-sided |
| `interceptions` | `interceptions` | count | **1.5** | PAdj + one-sided |
| `blocks` | `outfielderBlocks` | count | 1.0 | PAdj + one-sided |
| `recoveries` | `ballRecovery` | count | **1.5** | PAdj + one-sided |

### Defensive rate (2)

| Feature | Sofascore field | Type | Weight | Notes |
|---|---|---|---|---|
| `tackles_won_pct` | `tacklesWonPercentage` | rate | **1.5** | Efficiency signal — bidirectional, no one-sided treatment |
| `aerials_won_pct` | `aerialDuelsWonPercentage` | rate | 1.0 | Physical profile indicator |

### Creation (2, downweighted)

| Feature | Sofascore field | Type | Weight | Notes |
|---|---|---|---|---|
| `xa` | `expectedAssists` | count | **0.5** | Lower weight — most #6s have near-zero xA |
| `key_passes` | `keyPasses` | count | **0.5** | Same rationale |

### Shot output (2, sign-flipped)

| Feature | Sofascore field | Type | Weight | Notes |
|---|---|---|---|---|
| `shots` | `totalShots` | count_flip | 1.0 | High z-score = low shot output = correct #6 profile |
| `xg` | `expectedGoals` | count_flip | 1.0 | Same — high xG is a #10 flag |

### Ball security (2, sign-flipped)

| Feature | Sofascore field | Type | Weight | Notes |
|---|---|---|---|---|
| `dispossessed` | `dispossessed` | count_flip | 1.0 | Press resistance |
| `possession_lost` | `possessionLost` | count_flip | 1.0 | Broader security signal including dispossessions and poor touches |

### Physical / presence (3)

| Feature | Sofascore field | Type | Weight | Notes |
|---|---|---|---|---|
| `touches` | `touches` | count | 1.0 | Ball involvement — surrogate for positional centrality |
| `aerial_duels_won` | `aerialDuelsWon` | count | 1.0 | Aerial volume, complements aerial_won_pct |
| `clearances` | `clearances` | count | 1.0 | Last-line defensive presence |

**Total: 20 features.**

---

## Features dropped from v0.1 FBref spec and why

The switch from FBref to Sofascore eliminated some planned features and exposed others not in the original spec:

| v0.1 Feature | Status | Reason |
|---|---|---|
| `short_pass_completion_pct` | **Dropped** | Not available as a distinct field in Sofascore |
| `medium_pass_completion_pct` | **Dropped** | Same |
| `prog_passes_p90` | **Dropped** | Sofascore exposes `accurateFinalThirdPasses` instead of a progressive-pass count |
| `prog_carries_p90` | **Dropped** | No equivalent carry-progression field in Sofascore |
| `passes_into_pen_area_p90` | **Dropped** | Not separately exposed |
| `switches_p90` | **Dropped** | Not exposed |
| `sca_p90` | **Dropped** | Not available |
| `pct_touches_def_third`, `mid_third`, `att_third` | **Dropped** | Not available |
| `miscontrols_p90` | **Dropped** | Not available; `possession_lost` covers this broadly |
| `ball_recoveries_p90_padj` | **Renamed** | Sofascore field is `ballRecovery` → `recoveries` |
| `touches`, `aerial_duels_won`, `clearances`, `own_half_passes` | **Added** | Not in v0.1 spec; available in Sofascore and discriminating |
| `possession_lost` | **Added** | Broader than v0.1's `dispossessed_p90` alone |

---

## Weights

| Group | Features | Weight | Rationale |
|---|---|---|---|
| Defensive volume + own-half passes | tackles, interceptions, recoveries, own_half_passes, tackles_won_pct | **1.5** | Role is defined by defensive output and deep positioning |
| Blocks | blocks | 1.0 | Included at baseline — less discriminating than tackles/interceptions |
| Creation | xa, key_passes | **0.5** | Most #6s have near-zero; downweighted to avoid penalising good defenders who don't create |
| Everything else | — | 1.0 | Baseline |

---

## Open questions resolved post-pull

1. **`passes_per_touch` audit** — *Dropped*: FBref → Sofascore switch made this unavailable before the audit ran.
2. **Aerial duels threshold** — Not enforced. `aerial_duels_won` and `aerials_won_pct` are included unconditionally; low-aerial players get near-zero/mean values.
3. **Team strength as feature?** — Not included. PAdj on defensive features partially accounts for defensive opportunity. Deferred to Phase 2.
4. **Seed-set variance check** — Completed. Switched from "distance to mean" to "min distance to any seed" after Lobotka was removed; the remaining 3-seed cluster (Rodri / Rice / Tchouaméni) is tight relative to the population median pairwise distance.

---

## What this spec does NOT include

- **Pressure / pressing intensity** — StatsBomb event data only; goes in Stage 2
- **Off-ball positioning / heatmap zones** — requires tracking data
- **xT (expected threat)** — possible Phase 2

---

## Revision log

- **v0.1 (2026-05-12)** — initial FBref spec, 27 features, weighted Euclidean, PAdj on defensive volume, z-score normalization
- **v0.2 (2026-05-14)** — rewritten to Sofascore 20-feature implementation. FBref dropped (ScraperFC issues). 7 features removed (unavailable in Sofascore: short/medium pass completion, progressive passes/carries, switches, SCA, territory % touches, miscontrols). 4 features added (touches, aerial_duels_won, clearances, own_half_passes — available in Sofascore and discriminating). Added one-sided distance for defensive features. Added possession_lost as sign-flipped feature.
