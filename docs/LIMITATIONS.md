# Limitations and deployment status

- Validation images were reused during research and method exploration. This is not an untouched final test set. Filename-prefix allocator splits do not prove complete scene independence or detector-selection independence.
- The custom task merges VisDrone categories 1 and 2 into one human/person class. It differs from the official multi-class and ignored-region evaluation protocol.
- Mean per-image Recall, pooled Recall, per-image AP and dataset AP are distinct. Use CLAIMS.md rather than combining their numbers or confidence intervals.
- Size-matched JPEG is approximate. V2's actual-byte constraint is a different learned-versus-baseline comparison. No universal rate/performance advantage is established.
- Historical RGB/BGR interface handling is preserved. Correcting it requires a separate comparable re-evaluation, not reinterpretation of saved results.
- Learned V2 is a separate experimental result and does not universally replace Risk-Aware. Its tuned Recall interval includes zero.
- CPU latency is not evidence of real-time UAV deployment. No airborne compute, radio, power, reliability, operator workflow or field safety evaluation is established here.
- **This system has not been deployed in real search-and-rescue operations.** These are offline research results, not validated operational rescue performance.
- Models and raw evidence are withheld. Public aggregates alone cannot reproduce full AP/bootstrap or inference. The sanitized allocator candidate is external to Git; distribution clearance is separate. No detector download link is provided.
- AGPL preparation does not establish ownership of every contribution or rights over external models/data. Final rights/ownership review remains pending; local clean-environment and release checks have passed.
