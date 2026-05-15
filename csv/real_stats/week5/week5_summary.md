# Week5 Ablations — Summary
- Significance threshold: |Z| ≥ 1.96


## half_life
- Best: **recap_k4**  (t½ ≈ 87.65)
- Top-5 by t½:
- recap_k4: t½≈87.65
- recap_k12: t½≈23.31
- recap_k6: t½≈21.49
- recap_k16: t½≈19.51
- recap_k8: t½≈11.32

## mixed_effects
- Best: **recap_k16**  (Z̄ = -7.64)
- Top-5 by Z̄:
- recap_k16: Z̄=-7.64 — significant
- recap_k12: Z̄=-4.23 — significant
- recap_k3: Z̄=-2.73 — significant
- recap_k4: Z̄=-1.05
- recap_k8: Z̄=-0.65

## Overall pick
- **recap_k4** (rank_mean = 2.50)


## Rank table (lower is better)
variant	rank_mean	rank_std	rank_min	rank_max	n_metrics
recap_k4	2.50	2.12	1.0	4.0	2
recap_k6	2.50	0.71	2.0	3.0	2
recap_k12	4.00	2.83	2.0	6.0	2
recap_k2	4.00	4.24	1.0	7.0	2
recap_k8	4.00	1.41	3.0	5.0	2
recap_k16	5.50	2.12	4.0	7.0	2
recap_k3	5.50	0.71	5.0	6.0	2