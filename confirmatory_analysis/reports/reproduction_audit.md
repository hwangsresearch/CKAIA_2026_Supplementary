# Reproduction audit

- Donors total: 111 (expected 111): OK
- Donor Pathology: {'earlyAD': 33, 'nonAD': 57, 'lateAD': 21} (expected nonAD 57 / earlyAD 33 / lateAD 21)
- ATAC nuclei: 1217165 (expected 1,217,165): OK
- Cell classes analysed: ['Exc', 'Inh', 'Ast', 'Oli', 'OPC', 'Mic_Immune'] (Vasc excluded)

## Labeled donors (lateAD/nonAD) with >=200 nuclei, per class
- Exc: n=71 (lateAD 16, nonAD 55)
- Inh: n=69 (lateAD 14, nonAD 55)
- Ast: n=64 (lateAD 16, nonAD 48)
- Oli: n=74 (lateAD 18, nonAD 56)
- OPC: n=57 (lateAD 14, nonAD 43)
- Mic_Immune: n=59 (lateAD 15, nonAD 44)

## Preprocessing and representation-scope audit
- Exploratory persistence-image range/bandwidth was fitted globally. The confirmatory downstream implementation fits those parameters on outer-training diagrams.
- The prior outside-CV post-hoc residualized value is not retained. The binary target is unchanged and covariates enter the fold-specific downstream model.
- Original single-split permutation test used one fixed split -> high-variance; superseded by repeated 20x5 CV + donor bootstrap.
- The released LSI–Harmony and saved COAST representations were computed once on the full cohort without outcome labels; they are not fold-wise retrained. The target (lateAD vs nonAD) is never residualized.