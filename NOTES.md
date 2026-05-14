# Build Notes

## Validation set (pre-registered 2026-05-12, BEFORE any Stage 1 run)

This is the validation set for the archetype filter. Written before pulling full data and before running similarity scoring. Any post-hoc change to this list must be justified with a one-line reason in the revision log below.

### Must rank top 10 (seed-bug check)
- Rodri (Man City)
- Martin Zubimendi (Real Sociedad)
- Aurelien Tchouameni (Real Madrid)

If any seed falls out of top 10, archetype filter is broken — not a finding.

### Must rank top 50 (archetype validity)
- Declan Rice (Arsenal)
- Moises Caicedo (Chelsea)
- Stanislav Lobotka (Napoli)
- Angelo Stiller (Stuttgart)
- Edson Alvarez (West Ham)
- Khephren Thuram (Juventus)
- Joshua Kimmich (Bayern)
- Joao Palhinha (Bayern / Fulham window)
- Granit Xhaka (Leverkusen)

### Must rank below top 100 (negative controls)
- Bruno Fernandes (Man Utd) — wrong archetype, attacking 10
- Jude Bellingham (Real Madrid) — B2B/10, not a 6
- Florian Wirtz (Leverkusen) — 10
- Thomas Muller (Bayern) — 10, attacking
- Bukayo Saka (Arsenal) — wide forward
- Kevin De Bruyne — 8/10

### Boundary cases (interesting either way, no pass/fail)
- Ruben Neves
- Joey Veerman
- Frenkie de Jong
- Pedri
- Sandro Tonali

### Leave-one-out check
For each seed, drop it from the archetype mean, recompute, verify the dropped seed still appears in top 10. If not, archetype is overfit to that seed.

---

## Methodology decisions

### One-sided distance for defensive features (resolved 2026-05-14)

Defensive features {tackles, interceptions, recoveries, blocks} use one-sided distance: if a player **exceeds** the seed archetype in defensive volume, that shortfall contributes **zero** to their distance score. Only falling *below* the seed is penalized.

**Rationale:** Exceeding the seed on defensive volume is monotonically good for the role. Without this, Baleba or Kamara would be penalized for being more defensively active than Rodri, which is incoherent. The one-sided treatment applies only to the four count-based defensive features, not to rate features (tackles_won_pct, aerials_won_pct) which can go either direction.

**Implementation:** `adj_diff[:, :, one_sided] = np.minimum(adj_diff[:, :, one_sided], 0.0)` in `run_stage1()`.

### 25/26 partial-season inclusion (resolved 2026-05-14)

**Question:** Does including the incomplete 25/26 season distort rankings via unequal season weighting?

**Finding — aggregation is already minutes-proportional:** The pipeline sums count-feature totals across all season rows and divides by `total_minutes / 90`. A season contributing 500 of 2500 total minutes contributes exactly 20% to the per-90 rate. Rate features are explicitly minutes-weighted: `Σ(rate_s × min_s) / Σ(min_s)`. There is no equal-season-weight problem in the aggregation mechanics.

**Finding — ranks do shift materially:** Comparing top-30 archetype ranks with 3 seasons (25/26 + 24/25 + 23/24) vs 2 seasons (24/25 + 23/24):

| Rank movement | Players |
|---|---|
| ≥ 30 positions | Tyler Adams (+43), Thomas Partey (+37), Youri Regeer (+39), Amadou Haidara (+30) |
| 10–29 positions | Lucien Agoumé (−20), Melle Meulensteen (−14), Christian Nørgaard (+15), Pervis Estupiñán (+15) |
| 5–9 positions | Marten de Roon (+14), Ellyes Skhiri (−11), Edson Álvarez (−8), Alexis Mac Allister (−8) |
| < 5 positions | 11 of 30 players |

25/26 contributes ~30% of minutes for most top-30 players (mean 29.7%, median 31.6%).

**Interpretation of rank shifts:** The movements are driven by three distinct mechanisms, not by methodological distortion: (1) **form changes** — Tyler Adams (+43) is genuinely better in 25/26 than in his 24/25 form, which is real signal; (2) **population composition** — players like Ethan Ampadu exist only in 25/26 (100% of their minutes), entering and exiting the ranked pool entirely; (3) **reference pool shifts** — z-score means/stdevs change as the qualifying population changes.

**Decision: include 25/26 as designed.** Excluding the most recent season to avoid partial-season noise would be arbitrary censorship of real signal. Minutes-proportional weighting naturally limits the influence of players with very few 25/26 minutes. The rank movements reflect real form differences.

**Caveat to carry forward:** Ethan Ampadu (2943 min, 100% from 25/26) is effectively a single-partial-season profile. His rank in the main track should be noted as lower-confidence than players with multi-season data.

---

## Revision log

- **2026-05-12:** initial validation set, pre-registered
- **2026-05-14:** removed Stanislav Lobotka from SEED_NAMES — distributor specialist in a controlled possession system, outlier among seeds that pulled archetype away from athletic/defensive profiles (Baleba, Kamara, Lamine Camara); retained Rodri + Rice + Tchouaméni as tighter "complete 6" cluster matching United's actual need
- **2026-05-14:** switched from FBref to Sofascore data source — ScraperFC FBref scraper failed for grouped stat types; Sofascore returns flat player-statistics tables without that issue; feature count dropped from 27 to 20 (see data/FEATURES.md v0.2 for full drop list)
- **2026-05-14:** added one-sided distance for defensive features {tackles, interceptions, recoveries, blocks} — player exceeding seed on defensive volume contributes zero distance; only shortfall penalized; rationale: more defensive volume is monotonically good for the role
- **2026-05-14:** resolved 25/26 partial-season question — aggregation confirmed minutes-proportional (no distortion); rank shifts are real signal; decision to include 25/26 stands; Ethan Ampadu (100% from 25/26) flagged as single-season profile
