# EXP08 Mechanism Report

## Scope
Mechanistic ablation of IQP losses: parity_mse vs prob_mse/xent under matched protocol.

## Main Outcome
Parity-moment training is compared against standard losses via holdout mass flow and Q80.

## Dataset Summary
- atlas_rows: 24
- atlas_pairs: 16

## Pairwise Effects (Median + 95% bootstrap CI)
- paper_even__prob_mse (n=4, finite Q80 ratio n=4):
  delta_qH: median=0.0477421, CI=[0.0408342, 0.0528097]
  delta_qH_ratio: median=9.77759, CI=[8.36284, 10.8154]
  Q80_ratio: median=0.447576, CI=[0.240558, 1.03196]
  eta_H_plus: median=0.124467, CI=[0.107466, 0.142413]
  Wilcoxon(delta_qH=0): p=0.125
- paper_even__xent (n=4, finite Q80 ratio n=4):
  delta_qH: median=0.00310959, CI=[-0.0202592, 0.0295637]
  delta_qH_ratio: median=0.636845, CI=[-4.14909, 6.05464]
  Q80_ratio: median=1.4214, CI=[0.331345, 2.73117]
  eta_H_plus: median=0.0467474, CI=[0.0272481, 0.107208]
  Wilcoxon(delta_qH=0): p=0.875
- paper_nonparity__prob_mse (n=4, finite Q80 ratio n=0):
  delta_qH: median=0.043571, CI=[0.0375022, 0.048639]
  delta_qH_ratio: median=8.92334, CI=[7.68046, 9.96127]
  Q80_ratio: median=nan, CI=[nan, nan]
  eta_H_plus: median=0.0915743, CI=[0.0821117, 0.108306]
  Wilcoxon(delta_qH=0): p=0.125
- paper_nonparity__xent (n=4, finite Q80 ratio n=0):
  delta_qH: median=0.0213295, CI=[-0.00813513, 0.0269952]
  delta_qH_ratio: median=4.36828, CI=[-1.66607, 5.52862]
  Q80_ratio: median=nan, CI=[nan, nan]
  eta_H_plus: median=0.0780476, CI=[0.0208765, 0.0837174]
  Wilcoxon(delta_qH=0): p=0.25

## Mechanistic Correlations
- cos_align vs qH (all losses): rho=0.6547826086956522, p=0.000516719100681465, n=24
- cos_align vs logQ80 (all losses): rho=-0.5174825174825175, p=0.08486877113393489, n=12
- parity_mse: rho(cos,qH)=0.11904761904761905, rho(cos,logQ80)=1.0
- prob_mse: rho(cos,qH)=0.5952380952380953, rho(cos,logQ80)=-0.39999999999999997
- xent: rho(cos,qH)=0.261904761904762, rho(cos,logQ80)=-0.39999999999999997

## Mass-flow Interpretation
Positive delta_qH and eta_H_plus indicate directed mass transfer toward unseen holdout states.

## Limitations
- Controlled synthetic benchmark; no broad OOD/real-world generalization claim.
- Q80=inf cases are reported explicitly and affect finite-ratio analyses.

## Artifacts
- Output directory: `/Users/superposition/Coding/iqp_generative/outputs/exp08_mechanism_atlas`
- Required files: figA..figJ, atlas_rows.csv, atlas_pairs.csv, atlas_summary.json
