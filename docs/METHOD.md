# Method

The baseline pipeline loads an image, obtains low-threshold person proposals from a fixed detector, builds a continuous importance map, assigns tile qualities, encodes/decodes the TAC representation and evaluates the reconstruction with the same detector.

Risk-Aware importance combines confidence uncertainty, small-object risk, spatial/context support and a texture contribution. The historical headline uses mean tile pooling, 256-pixel tiles, background Q8 and foreground Q82. A uniform JPEG is selected separately for each adaptive result by closest file size over a dense quality ladder. This is approximate size matching, not guaranteed byte equality.

The TAC representation stores tile metadata and independently coded JPEG payloads. It is a research container rather than a standardized learned codec. Header overhead is included in byte accounting. Candidate detection adds sender-side cost.

Learned Allocation V2 uses a local residual MLP over tile/image/proposal/budget features. The detector and codec remain fixed; native detector loss and output constraints train quality allocation. Actual JPEG payload costs and container overhead determine the discrete inference budget. Ground-truth labels supervise training and evaluation, not inference-time allocation. Public training/evaluation source is retained; checkpoints are withheld.

Human ROI, Random ROI and optional Oracle controls remain available through the baseline workflow. Oracle uses GT to construct importance and is not deployable. Their different direct file sizes must not be treated as strict matched-rate comparisons.

See ../CLAIMS.md for exact claim boundaries and ../results/ for frozen settings. Prototype preprocessors, historical tuning sweeps and third-party preprocessing adapters are outside v0.1.0.

## Frozen importance semantics

The v0.1 Risk-Aware headline uses historical_full_gaussian: Gaussian fields cover the full image. Learned V2 uses truncated_6sigma: fields stop at six standard deviations. Tiny tails becoming zero alter the existing positive-support texture gate, so the modes are scientifically distinct. Explicit configs/API choices preserve both. The sanitized allocator identifies its V2 method version and retains the truncated behavior; exporting state does not change the model or codec.
