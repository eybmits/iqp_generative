Cover Letter (Journal/Workshop) - Focus: "second axis: discovery"

Dear Editor [Name],

We submit our manuscript "Sampling the Unseen: Holdout Recovery under Band-Limited Parity Constraints" for consideration in [Journal/Track].

A common motivation for parity-feature MMD / parity-moment losses in (IQP-)QCBMs is hybrid trainability: the objective reduces to estimating parity observables classically while keeping sampling quantum ("train classically, deploy quantum"). While this motivation is well established, it does not address an operational question that dominates many scientific discovery settings in exponentially large discrete spaces: how many samples are required before rare, high-value configurations appear at all.

Our manuscript introduces a second, independent motivation axis for parity-moment training—discovery-oriented sampling—and makes it explicit, measurable, and mechanistically attributable. Concretely, we argue that band-limited Walsh/parity constraints do not merely make training practical; they can systematically transfer probability mass into unseen/holdout regions, improving time-to-discovery relative to uniform sampling and relative to likelihood-style training that tends to concentrate mass on observed modes.

We operationalize this viewpoint through holdout recovery, where a high-value set H is strictly removed from training data and must be rediscovered purely by sampling. This yields a model-agnostic, black-box metric—recovery curves and a discovery budget Q80—that evaluates generalization in the currency of samples, rather than global distribution distances. We further provide a simple budget law linking discovery cost to the learned holdout mass, and a mechanism explaining why parity-moment constraints can induce nontrivial holdout mass even when H is never seen in training: global Walsh constraints act as linear "balance laws" that couple seen and unseen regions, and completion under the probability simplex forces a "fill-in" of probability mass. Finally, we identify conditions and failure modes that determine when discovery succeeds or collapses (e.g., feature budget K, training sample size m, underdeterminacy, noise, and spectral visibility of the holdout set).

To isolate the effect of the model class under identical supervision, we compare IQP-QCBMs to matched classical controls trained under the same parity-moment supervision and optimization budget, showing that the discovery-relevant mass transfer and budgets depend on inductive bias under fixed information constraints.

We believe this framing—trainability (hybrid estimation) vs. discovery (sampling budget) as two separable design axes for parity-moment objectives—adds a useful conceptual tool for the community: it clarifies why objectives that are easy to train can still fail at discovery, and it provides concrete knobs and diagnostics for discovery-driven generative modeling.

To our knowledge, the literature has not separated parity-moment training into two independent axes—hybrid trainability and discovery efficiency—nor provided a mechanistic account of how band-limited Walsh constraints can transfer probability mass into unseen regions and how this breaks under finite budgets/noise.

Thank you for considering our submission. We are happy to provide additional experiments or clarifications if needed.

Sincerely,
Markus Baumann (corresponding author)
[Affiliation, Address]
[Email]
