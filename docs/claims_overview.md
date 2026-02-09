# Claims Overview

This document records the current claim set in professional wording.

## Claims

| Claim | Formulation |
|---|---|
| **1** | **Fit =/= Discovery:** Strong fit to training or target distributions is neither necessary nor sufficient for efficient discovery of unseen states; discovery must be evaluated as an independent axis via metrics such as \(q(H)\), \(R(Q)\), and \(Q_{80}\). |
| **2** | **Budget Law:** The sample budget required to recover holdout states is primarily governed by the model-assigned holdout mass \(q(H)\), yielding a quantitative law for discovery cost. |
| **3** | **Visibility / Invisibility:** Discovery performance is mechanistically controlled by the spectral visibility of the holdout under the chosen feature family; spectrally visible holdouts are recovered substantially faster than spectrally invisible ones. |
| **4** | **Spectral Reconstruction:** Spectral reconstruction provides a predictive and mechanistically interpretable approximation of recovery kinetics, explaining the observed recovery dynamics as a function of sample budget \(Q\). |
| **5** | **IQP > strong classical baselines on high-score holdout:** Under fair, matched training conditions, IQP-QCBM consistently outperforms strong classical baselines on discovery-oriented metrics for high-score/high-value holdouts. |
| **6** | **Fair baseline protocol:** Under matched supervision, data, and optimization budgets, IQP and classical controls can be compared pairwise without confounding from protocol mismatch. |
| **7** | **IQP > strong classical baselines on global holdout:** The IQP discovery advantage remains when the holdout is defined globally over the support rather than restricted to high-score states. |
| **8** | **IQP > strong classical baselines across beta sweep:** The IQP advantage in discovery metrics remains stable across a broad beta range, demonstrating robustness beyond a single target-distribution regime. |
