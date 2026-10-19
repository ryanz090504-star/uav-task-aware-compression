# Risk-Aware reproduction discrepancy investigation

> Current follow-up (2026-10-18): the remaining single-image mismatch is explained by Ultralytics version drift at an IoU boundary. The full clean-environment replay remains 547/548 identical in detection outcomes, with 548/548 encoded byte-count matches; it is not an exact all-image detection replay. The recovered historical stack reproduces the affected image's frozen metrics. Earlier investigation and stop decisions below are retained as dated background; see the final single-image follow-up.

Date: 2026-10-17. Scope: Risk-Aware only. No Learned V2 execution, public scientific edits, archive edits, CLAIMS.md edits or frozen-result edits occurred. Investigation scripts and raw/per-image artifacts remain external.

## Finding

**The primary cause is a historical/current implementation difference in importance-map support, amplified by the texture gate. It is not established as a dependency-version regression.** The public code matches the archive's current code, but the archived handoff contains an earlier implementation consistent with the frozen final records.

The historical implementation evaluates each Gaussian on the full image. Current importance.py limits each Gaussian/hard-region computation to a rectangle extending six standard deviations from its center. Although the removed Gaussian values are tiny, both versions add texture only where importance > 0. Turning tiny positive tails into exact zero disables up to 0.12 of texture contribution over large image regions. That changes tile means, integer JPEG qualities, adaptive sizes, matched JPEG qualities and detections. The optimization therefore does not preserve this method's behavior.

The earlier verification's source comparison was against the archive's current source, which already contains the same truncation. It did not establish identity with the earlier implementation preserved alongside the frozen records.

Three controlled images, including the two largest negative contributors, reproduce both historical and current saved byte/quality/Recall/count fields exactly under the **same current clean environment**, same checkpoint and same newly computed proposals. This establishes a direct implementation effect on those cases. A new full 548-image historical inference run was not performed: investigation stopped once this cause was demonstrated, as requested. Attribution for every remaining changed image has not been independently replayed.

## Recovered environment

Historical evidence: requirements-cpu-lock.txt, matching installed distribution metadata read without executing the old environment, and pyvenv.cfg. Python version from the archived environment is 3.12.13; the clean environment uses 3.12.14. These are recoverable archive environment snapshots, not an immutable complete runtime inventory captured at the exact final inference launch.

| Dependency | Archive snapshot | Validated clean environment |
|---|---|---|
| Pillow | 12.3.0 | 13.0.0 |
| numpy | 2.5.2 | 2.5.4 |
| torch | 2.13.0 | 2.14.1 |
| torchvision | 0.28.0 | 0.29.1 |
| ultralytics | 8.4.129 | 8.4.181 |
| opencv-python | 5.0.0.93 | 5.0.0.93 |

The historical lock SHA256 is 2719a1f696420eb659b96eaa820c247f435c6aa325f366bd195a41155d8e0ed0. Both historical and current OpenCV versions are 5.0.0.93. Pillow's current JPEG codec reports 8.0 and libjpeg-turbo 3.2.0. The historical native JPEG library version was not recorded in the available run records; it cannot be inferred from the Pillow package version alone. OpenCV is used in Ultralytics preprocessing; the project's JPEG encoding/decoding uses Pillow, not OpenCV or a standalone JPEG package.

A historical dependency environment was **not created**: the conditional reason for one was not met. Source isolation reproduces the old and new saved outcomes under fixed current dependencies, while fixed-quality JPEG sizes and Recall/counts agree over all images. Downgrading packages would not remove the discovered source difference. This does not prove that all library drift is irrelevant to every detection coordinate.

## JPEG and input behavior

- Encoder helper is AST-identical in historical and current source: Pillow Image.fromarray(..., mode="RGB"), format JPEG, integer quality, optimize=True, subsampling=0 (4:4:4), no progressive flag and no preserved EXIF/ICC metadata.
- Quality is converted with int; paired matching searches all integer Q=1..100, minimizing absolute byte difference. Ties follow the same ascending insertion order. No JPEG quality policy change was found.
- Input decoding is identical: Image.open(...).convert("RGB") into uint8 NumPy arrays. All 548 input images are RGB and none has an EXIF orientation tag. Neither dataset loader applies EXIF transpose.
- Decoder helpers are identical: Pillow JPEG decoding followed by RGB conversion. The tile dimension fallback is the same bilinear resize; tile layout/dimensions match.
- Historical/current original source sizes match for all 548 images. All **2,192 fixed-quality JPEG sizes** match (Q15/Q35/Q60/Q90), with identical TP/FP/FN, Recall and Precision. Historical encoded payload hashes were not retained, so equal lengths alone do not establish full bitstream identity.
- The codec uses RGB arrays consistently. The detector wrapper passes those arrays unchanged as Ultralytics NumPy sources; both installed upstream versions treat NumPy inputs as BGR and reverse channels. This is a shared historical/current channel-convention concern, not the discovered difference. No channel fix was attempted; changing it would alter the scientific pipeline and require separate review.
- Controlled archived/current adaptive decoding changes pixels because selected JPEG qualities differ. On the first controlled frame, six tiles change quality, 971,571 decoded channel values differ, mean absolute channel error is 1.414080 on the 0..255 scale, and 499,511 pixels change the positive-support mask. Maximum importance change is about 0.120000012, not merely a 1e-8 tail error.
- On the two largest negative-contribution frames, respectively four and two tile qualities change; decoded channel MAE is 2.728152 and 0.522020. The current encoder reproduces the new saved sizes; the archived encoder reproduces the historical sizes, with current dependencies throughout.

## Configuration, ordering and evaluation checks

| Component | Evidence / conclusion |
|---|---|
| Nominal Risk-Aware parameters | Same bg8/fg82, tile256, tile_workers8, context18, small-object reference0.018, texture weight0.12, risk weights0.58/0.42, smoothstep, mean tile pooling, max aggregation, zero offsets |
| Importance implementation | **Different**: full-frame fields versus six-sigma cropped support; nominal config equality is insufficient |
| Proposal/receiver confidence | Same wrapper default and recorded final invocation: 0.12 |
| Model and data | Exact hashes/counts match the prior verification; see REAL_DATA_RELEASE_VERIFICATION.md |
| Inference | Same project predict/predict_batch method ASTs, imgsz1280, CPU, batch8; upstream default NMS IoU0.7, max_det300, augment=False and agnostic_nms=False agree |
| Tile geometry/container | Same row-major tile coordinates, edge handling, TAC1/version2 header schema and payload layout; default quality mapping agrees, but its importance input changes |
| Matching | benchmark.py is byte-identical to the archived handoff, including nearest-byte JPEG selection and pending batch order |
| Metrics | _match_detections, average_precision and evaluate_detections are AST-identical; IoU threshold0.5; mean paired per-image Recall and pooled target Recall remain different aggregations |
| Dataset ordering | dataset.py is byte-identical; all 9,316 record keys and their ordering agree exactly |
| Complete detector equivalence | **Not claimed**: original TP/FP/FN match on all548 and uniform TP/FP/FN on all2192, but IoU/coordinate-related numerical differences exist |

Original mean-matched-IoU values differ on 479/548 rows (maximum absolute change 0.000004706). Uniform mean-matched-IoU differs on 1,913/2,192 rows (maximum 0.005368267); one uniform row has an AP difference of 0.000109778 despite unchanged Recall/counts. These are residual inference/order differences, not evidence of full bitwise detector equivalence. They do not explain the replicated importance/byte/Recall changes in the controlled cases.

Historical source provenance comes from the archived September handoff; its frozen final records.csv hash equals the root archive's final records hash: 8c424482c06baabc71f28e18c4bd35f1a687192dbe308d09271c4f457fa9d9e6. The separately verified handoff has the same historical importance.py hash. Its REPRODUCE.md records confidence0.12, CPU, all548, tile256, Q1..100 matching, detection batch8 and tile workers8. These associations support the historical implementation identification; no original Git commit or complete executable snapshot was recovered.

| Source | Historical handoff SHA256 | Current public SHA256 |
|---|---|---|
| compressor.py | c821c12e4958f8bba81a68e771ed317aa2777aabe8d9e7d64c225625a294b708 | c87f8600e084fc0e0e0980835a4d5a1c5428ee499f28398863ee1762361b36a2 |
| importance.py | 348ec7d66fcb2b9aa5c6a50b3bebfd295ab5877381d2e987ed35d0b93c5b41e0 | 56685ef0e1593c62a9280a170b6ada43c9f0ab070f3d122cc496cd8c4c1f842d |
| types.py | 7b9d1a43527fde678c29b092a4c1f6dee1ba6ea40e4977fd6b8539326cd70662 | 1a81753594440a331d4ac54f84718342d1861a431e0f87693d8cac34ef5933f7 |
| detector.py | 32cfa881b07c72a53994d06c78dc5e223ee06a327d295594ef9ed65b661f2a44 | aee7c26398597784ea4d427f0f4da45714739217acb0047d04fb7da8477375f5 |
| benchmark.py | c27fb0d88f52ee7eb7e1a1867283386c94e327a23a750ff96364bb77670fb4cb | c27fb0d88f52ee7eb7e1a1867283386c94e327a23a750ff96364bb77670fb4cb |
| dataset.py | d0d059b22d909de08e82cc746ae5633694f31c05d868e65d3524a0858788bb01 | d0d059b22d909de08e82cc746ae5633694f31c05d868e65d3524a0858788bb01 |
| metrics.py | adaa3aa7eee80e6cf9c4a732c5a0ed471bf15aa18e3a5b3d422018619b95b4cc | 22d38c7284683fa13fbdbcfd3a1bac7699cf1c73c14ed62824ca31e006408a74 |

## Full saved-record comparison

Comparisons directly join historical final records.csv against the prior verification's regenerated records.csv, without changing either.

- Mean paired per-image Recall gain: historical **+2.617401515 pp**, current **+2.312588910 pp**, change **-0.304812605 pp**.
- Historical adaptive/JPEG mean Recall: 62.229248508% / 59.611846993%. Current: 61.672268196% / 59.359679286%.
- Separate pooled target-level delta: historical **+3.149831770 pp**, current **+3.250053690 pp**. This is not the public headline.
- Claim adaptive Recall changes on **104/548** images; paired JPEG Recall on **68/548**; matched JPEG quality on **115/548**.
- Adaptive encoded size changes on **474/548** images; **248/548** change by more than 1%, none by more than 5%. Median signed size change is -0.883113%; range -4.110231% to +0.007353%.
- Nonzero paired-delta contributions occur on **141/548** images: 73 positive and 68 negative. They sum exactly to -0.304812605 pp. Absolute contributions sum to 2.071474578 pp, demonstrating substantial cancellation.
- The five largest absolute contributors net -0.154240528 pp; ten net -0.199860966 pp; twenty net -0.380495417 pp, with other images offsetting part of that loss. The discrepancy is sensitive to a small set of low-count frames, but changes are not confined to a handful of images.

The external per-image-discrepancy.private.csv contains every image and all requested historical/current adaptive/JPEG Recalls, byte sizes, matched qualities, counts and its contribution. That file is withheld from the public candidate. The table below shows the 20 largest absolute contributions; Recall values are fractions, and bytes refer to adaptive streams. Rows not replayed are labeled as inference rather than individual causal proof.

| Image | Historical adaptive Recall | Current adaptive Recall | Historical JPEG Recall | Current JPEG Recall | Historical adaptive bytes | Current adaptive bytes | Likely cause |
|---|---:|---:|---:|---:|---:|---:|---|
| 0000242_04600_d_0000014 | 0.750000 | 0.500000 | 0.250000 | 0.500000 | 31565 | 30682 | Importance support/texture gate; controlled replay confirmed |
| 0000359_01961_d_0000707 | 0.500000 | 0.000000 | 0.500000 | 0.500000 | 25834 | 25680 | Importance support/texture gate; controlled replay confirmed |
| 0000316_01001_d_0000524 | 0.428571 | 0.428571 | 0.142857 | 0.571429 | 32325 | 32236 | Consistent with importance support change; not individually replayed |
| 0000360_07057_d_0000749 | 0.000000 | 0.000000 | 0.333333 | 0.000000 | 26867 | 26152 | Consistent with importance support change; not individually replayed |
| 0000021_00800_d_0000003 | 0.500000 | 0.500000 | 0.500000 | 0.250000 | 20088 | 19493 | Consistent with importance support change; not individually replayed |
| 0000276_02201_d_0000518 | 0.500000 | 0.250000 | 0.250000 | 0.250000 | 52807 | 51614 | Consistent with importance support change; not individually replayed |
| 0000276_02401_d_0000519 | 0.250000 | 0.000000 | 0.000000 | 0.000000 | 53767 | 52105 | Consistent with importance support change; not individually replayed |
| 0000287_03001_d_0000774 | 0.250000 | 0.500000 | 0.000000 | 0.000000 | 37974 | 37852 | Consistent with importance support change; not individually replayed |
| 0000333_02157_d_0000012 | 0.600000 | 0.800000 | 0.600000 | 0.600000 | 44892 | 43911 | Consistent with importance support change; not individually replayed |
| 0000276_01401_d_0000514 | 0.200000 | 0.000000 | 0.000000 | 0.000000 | 44503 | 44026 | Consistent with importance support change; not individually replayed |
| 0000220_01242_d_0000004 | 1.000000 | 1.000000 | 0.800000 | 1.000000 | 87232 | 86011 | Consistent with importance support change; not individually replayed |
| 0000287_02601_d_0000772 | 0.454545 | 0.272727 | 0.363636 | 0.363636 | 61465 | 59005 | Consistent with importance support change; not individually replayed |
| 0000359_03333_d_0000711 | 0.833333 | 0.666667 | 0.666667 | 0.666667 | 31852 | 31600 | Consistent with importance support change; not individually replayed |
| 0000162_00801_d_0000001 | 0.333333 | 0.166667 | 0.333333 | 0.333333 | 20412 | 20150 | Consistent with importance support change; not individually replayed |
| 0000333_02745_d_0000015 | 0.666667 | 0.500000 | 0.388889 | 0.388889 | 42684 | 42135 | Consistent with importance support change; not individually replayed |
| 0000359_03529_d_0000712 | 0.500000 | 0.666667 | 0.333333 | 0.333333 | 27340 | 27272 | Consistent with importance support change; not individually replayed |
| 0000249_01514_d_0000005 | 0.076923 | 0.153846 | 0.153846 | 0.076923 | 37051 | 36703 | Consistent with importance support change; not individually replayed |
| 0000359_02353_d_0000708 | 0.428571 | 0.285714 | 0.285714 | 0.285714 | 40160 | 39542 | Consistent with importance support change; not individually replayed |
| 0000242_06010_d_0000017 | 0.857143 | 0.714286 | 0.571429 | 0.571429 | 26849 | 26213 | Consistent with importance support change; not individually replayed |
| 0000244_00001_d_0000001 | 0.857143 | 0.714286 | 0.714286 | 0.714286 | 29981 | 29021 | Consistent with importance support change; not individually replayed |

## Controlled replay

All dependencies and source files were held fixed while two **unmodified** saved implementations were loaded in isolated module namespaces. The historical implementation was imported directly read-only with bytecode writing disabled. No monkey patch of the public package occurred. The same public detector proposals were supplied to both. Public helpers constructed the same risk/human/random settings, JPEG search and 12 pending adaptive/paired images in the original order, with detection batch8.

| Image | Archived/current adaptive bytes | Archived/current JPEG quality | Archived/current adaptive Recall | Archived/current JPEG Recall | Saved byte/quality/Recall/count fields reproduced? |
|---|---|---|---|---|---|
| 0000001_02999_d_0000005 | 126390 / 124415 | 12 / 12 | 0.641025641 / 0.641025641 | 0.564102564 / 0.564102564 | Yes, both versions |
| 0000242_04600_d_0000014 | 31565 / 30682 | 12 / 11 | 0.750000000 / 0.500000000 | 0.250000000 / 0.500000000 | Yes, both versions |
| 0000359_01961_d_0000707 | 25834 / 25680 | 16 / 16 | 0.500000000 / 0.000000000 | 0.500000000 / 0.500000000 | Yes, both versions |

For 0000242_04600_d_0000014, the old paired gain is 0.75-0.25=0.50, and the current gain is 0.50-0.50=0. The contribution to the aggregate discrepancy is -0.50/548*100=-0.091240876 pp. For 0000359_01961_d_0000707 it is also -0.091240876 pp. Together these two directly demonstrated cases account for -0.182481752 pp of the net -0.304812605 pp change; other images include compensating improvements.

No thresholds were tuned. No historical dependency downgrade was attempted. No full historical 548-image inference replay was run after the cause was established.

## Commands and retained evidence

All harnesses run with the validated clean interpreter from the workspace parent; VERIFICATION_ARTIFACTS denotes the external verification directory. Full expanded private command paths are retained externally. Public source imports resolve to the installed editable public package.

~~~powershell
python "$VERIFICATION_ARTIFACTS/investigation/compare_records.py" "$VERIFICATION_ARTIFACTS"
python "$VERIFICATION_ARTIFACTS/investigation/compare_sources.py" "$VERIFICATION_ARTIFACTS"
python "$VERIFICATION_ARTIFACTS/investigation/isolate_implementation.py" "$VERIFICATION_ARTIFACTS"
python "$VERIFICATION_ARTIFACTS/investigation/replay_recall.py" "$VERIFICATION_ARTIFACTS"
python "$VERIFICATION_ARTIFACTS/investigation/additional_evidence.py" "$VERIFICATION_ARTIFACTS"
python "$VERIFICATION_ARTIFACTS/investigation/metric_fields.py" "$VERIFICATION_ARTIFACTS"
python "$VERIFICATION_ARTIFACTS/investigation/small_differences.py" "$VERIFICATION_ARTIFACTS"
~~~

External evidence includes environment-source-comparison.private.json, record-comparison.private.json, per-image-discrepancy.private.csv, implementation-isolation.private.json, controlled-recall-replay.private.json, metric-field-comparison.private.json, small-inference-differences.private.json, additional-evidence.private.json, logs and private controlled-isolation arrays. The latter contain actual external image/importance arrays and must not be redistributed.

## Decisions and next action

1. **Most likely cause:** implementation drift in Gaussian support combined with the importance > 0 texture gate. The causal behavior is directly demonstrated on three images under current dependencies, including both largest negative contributors.
2. **Historical +2.6174 pp:** exactly recovers from all548 frozen records, and selected old inference outcomes reproduce. Full548 fresh historical inference is **not yet independently certified**. Do not describe this investigation as a complete historical rerun.
3. **Current +2.3126 pp:** is the measured result for the current truncated implementation and its changed byte operating point. It should **not automatically replace** the historical claim: the two results describe different executable methods. CLAIMS.md remains unchanged.
4. **Code change justified?** A scientific versioning/restoration decision is justified by the discovered mismatch; a dependency-only portability fix is not supported. Restoring historical behavior would be a scientific change requiring explicit separate authorization and validation. No such change was made.
5. **Recommended next action:** explicitly select the release's baseline implementation. For the frozen +2.6174 pp claim, preserve/version the historical full-frame semantics and perform a full548 verification with that identified source before asserting reproduction. If retaining truncation, treat it as a separate variant, validate its claims and budget, and review the shared detector channel convention separately. Only after that decision and baseline verification should Learned V2 run; its frozen design depends on its own baseline semantics.

The archive, public scientific code, CLAIMS.md and frozen results remain untouched. Model/dataset distribution restrictions remain in force.


## Full historical replay follow-up

Status: **FAIL / STOP**. This follow-up supersedes the original investigation's pending-full-replay statements.

The existing semantics are explicit: historical_full_gaussian is the v0.1 Risk-Aware configuration; truncated_6sigma is a distinct later variant retained for the learned allocator experiments. The earlier +2.312588910 pp run measured that later variant, not the historical executable definition.

Four disjoint contiguous ranges of 137 image IDs cover all 548 validation images. Nineteen completed checkpointed evaluations were recovered; only the remaining 529 were evaluated on resume. The aborted eight-image single-process attempt was excluded. No scientific code/config/threshold changes were made during this replay. Final shard-level aggregation was disabled. Records were merged in original image order, checked for duplicates and omissions, and then aggregated once. Normal-approximation paired-mean uncertainty was checked on the merged data; no bootstrap was rerun.

- Image count: 548; target count: 13969.
- Mean per-image Recall gain: +2.622203666 pp (frozen +2.617401515 pp).
- Separate pooled target-level gain: +3.156990479 pp (frozen +3.149831770 pp).
- The existing 31-test suite remains passed; it was not weakened or rerun for orchestration.
- Full comparison tables, input/model hashes, runtime and per-image exact-match counts are in REAL_DATA_RELEASE_VERIFICATION.md. Adaptive byte counts match on 548/548; adaptive Recall/count outcomes match on 547/548, with one additional TP in the replay. All matched-JPEG bytes/qualities/Recall/counts match on 548/548. This is not accepted as arithmetic roundoff; the cause is unconfirmed and further experiments have stopped.
- Frozen per-tile quality lists/TAC payloads were not retained with the CSV. Direct exact tile-list/bitstream equivalence is not verifiable. This limitation is distinct from file-size and Recall/count comparisons.

No provenance investigation was repeated. The already established training/evaluation semantics for V1/V2 remain truncated_6sigma; the exact historical truncation edit date remains unconfirmed. Learned V2 was not run.

STOP: remaining differences prevent full historical certification. No further algorithm adjustment, threshold tuning, frozen-result edit or Learned V2 execution was attempted. CLAIMS.md remains unchanged.


## Single-image boundary follow-up (2026-10-18)

Scope: only image 0000193_01497_d_0000111. No 548-image rerun, Learned V2 evaluation, scientific code edit, threshold adjustment, claim change or frozen-record replacement occurred. All TAC streams, images, model heads, predictions, GT assignments, package copies and diagnostic scripts are external and excluded from the public candidate. The internal archive was read-only throughout.

### Finding and confidence

**Primary classification B: dependency/version inference drift, specifically Ultralytics 8.4.129 versus 8.4.181.** The mechanism is an evaluation-IoU boundary crossing (category C), with no matching tie, assignment-order change or NMS survivor change. Confidence is high: changing only the Ultralytics package while retaining the current PyTorch/torchvision/NumPy/Pillow stack restores the complete frozen detection metrics for this image.

The same preprocessed tensor and checkpoint produce slightly different pre-NMS model-head values between Ultralytics versions. The affected candidate remains in both output lists at confidence rank 37 (zero-based rank 36), with the same raw anchor index 6289. NMS keeps the same 71 anchor IDs in exactly the same order. Its confidence is far above 0.12 in both runs. One existing FP becomes a TP, removing one FN; no extra detection is added.

| Quantity | Ultralytics 8.4.129 | Ultralytics 8.4.181 |
|---|---:|---:|
| TP / FP / FN | 30 / 41 / 8 | 31 / 40 / 7 |
| Affected confidence | 0.37319067120552063 | 0.37319108843803406 |
| Affected IoU with the same GT | 0.49999989713380127 | 0.5000036525006567 |
| Evaluation IoU threshold | 0.5 | 0.5 |
| Prediction count | 71 | 71 |
| AP@0.5 for this image | 0.6791276054856219 | 0.70094904773051525 |

The affected left-edge coordinate changes by -0.00006103515625 pixels; its other three coordinates are identical. The largest coordinate difference across all 71 corresponding output boxes is 9.1552734375e-05 pixels; maximum confidence difference is 1.9967555999755859e-06. The matched GT is entry 13 in the existing label order (index 12). Exact prediction coordinates and GT coordinates/assignment traces remain in root-cause.private.json and the external per-run prediction JSONs, rather than distributing annotations here.

Only rank 36 changes TP/FP assignment. No other retained prediction has IoU above 0.2 with the affected GT, so there is no competing assignment for it. Historical/public matching and BoundingBox.iou functions have identical ASTs. Matching uses stable descending confidence order, strict > when choosing the best available GT, and >= 0.5 for acceptance. The traces agree with the unmodified public evaluator. Detector coordinates originate as float32; the public box/IoU evaluator uses Python floats after conversion, with its original formulas and comparisons. Reordering the same predictions would not explain this isolated GT's crossing. NMS order/ties are excluded by identical retained anchor IDs/order; the coordinate difference is already present before NMS.

Upstream source inspection also finds a changed Conv/BatchNorm fusion implementation: the older package computes fused parameters explicitly, while the newer package calls PyTorch fuse_conv_bn_weights. This is a plausible source of the tiny deterministic numerical changes. The controlled experiment isolates the package version; it does not establish this individual upstream function as the sole causal internal edit. No upstream function was patched or substituted to force agreement.

### Codec and available artifact verification

Reconstruction used the unmodified public codec, historical_full_gaussian, original proposals, tile size 256, eight tile workers, BG8/FG82 and the frozen risk-map settings. Adaptive bytes: **49,246**. Q1-Q100 nearest-size JPEG pairing selects Q26, **49,673** bytes (427 bytes larger), matching the frozen row and the full replay. The 12 tile qualities match the full replay's retained per-image metadata exactly. TAC header version 2, dimensions 960 x 540, tile geometry/lengths and worker metadata were saved externally.

Reconstructed TAC SHA256: 89037e524fd33306e6fc48cb186408deb8d2ab34a44b99804a5c50ee88fd3331.

The earlier read-only archive search found no reliable frozen historical TAC stream/header/list for this image; the frozen CSV has no such fields. Direct byte-for-byte identity against an original frozen stream is therefore **not verifiable**. The current reconstruction and the reconstruction with recovered historical Torch/Ultralytics/NumPy/Pillow packages are, however, byte-for-byte identical. Decoding the same TAC with Pillow 12.3.0 and 13.0.0 produces identical pixels; both reconstructed decoded images also equal the current decoded input exactly. The two native JPEG libraries report libjpeg 8.0, with libjpeg-turbo 3.1.4.1 and 3.2.0 respectively. This excludes a decoded-image difference in these controlled comparisons, without inventing a missing original artifact.

The encoder uses Pillow RGB, optimize=True, subsampling=0 and unchanged integer JPEG qualities. The source image is RGB with no EXIF orientation value; conversion uses RGB without EXIF transposition. Codec/container and decoder behavior were not changed.

### Inference settings and input identity

| Setting | Recovered historical / current controlled comparison |
|---|---|
| Ultralytics | 8.4.129 / 8.4.181 |
| PyTorch | 2.13.0+cpu in recovered stack; 2.14.1 in current and package-isolation runs |
| torchvision | 0.28.0+cpu recovered; 0.29.1 current |
| NumPy / Pillow | 2.5.2 / 12.3.0 recovered; 2.5.4 / 13.0.0 current |
| OpenCV distribution | 5.0.0.93 in both; module reports 5.0.0 |
| Execution | CPU float32, 8 Torch threads, no CUDA, no FP16 |
| Checkpoint | Same withheld detector, SHA256 263cfd00bbd54283ad6982d79b04e7f9306d3d63c82a3264f500fb81be25426a |
| Inference image size | 1280; actual letterboxed tensor 3 x 736 x 1280 |
| Letterbox | rect=True, stride 32, isotropic resize 960 x 540 to 1280 x 720, centered 8-pixel top/bottom padding, value 114, linear interpolation; no scale-fill |
| Input channels and normalization | Both wrappers supply RGB arrays unchanged; both upstream NumPy paths interpret them as BGR and reverse channels, then float32 / 255. This shared convention is retained; no channel correction was made. |
| Confidence | 0.12 |
| NMS IoU / max detections | 0.7 / 300 |
| Class filtering | Upstream classes=None, agnostic_nms=False; public conversion retains class 0 |
| Augmentation | False |
| Deterministic setting | Upstream args.deterministic=True; actual torch.are_deterministic_algorithms_enabled()=False in both. No determinism flags were changed. |
| Batch handling | Public batch argument 8; supplied list of 12 same-image adaptive/paired variants is processed as an observed batch of 12; isolated inference batch is 1. |

The affected image's actual first-convolution input tensor SHA256 is identical in both package versions and both batch modes: fb5c7a146f55b6c070957964a2f19624e383eb4aab6536576b1bf05a33995055. Thus preprocessing/channel handling/letterboxing differences do not cause this mismatch. Float32 raw heads differ slightly, before NMS and output scaling. Diagnostic hooks only recorded values and did not replace model outputs.

### Repetition and historical-environment recovery

- Current validated environment: five complete 12-variant inference repetitions and five isolated decoded-image repetitions all return exactly the same predictions and 31/40/7. Three further instrumented repetitions of each mode confirm the same result.
- Separate recovered stack: three batch and three isolated repetitions all return exactly the same predictions and **30/41/8**. Recall, Precision, AP@0.5, small AP/Recall, miss rate and mean matched IoU all equal the frozen row exactly.
- Package isolation: current Torch 2.14.1 / torchvision 0.29.1 / NumPy 2.5.4 / Pillow 13.0.0 with only Ultralytics 8.4.129 substituted gives the exact same predictions as the recovered stack, for three batch and three isolated repetitions. The current Ultralytics result is therefore not caused by the Torch version alone.

Historical packages were copied read-only from the archived installed distributions into a new external environment. The original archive interpreter was never executed, no archive package path was placed on sys.path, and bytecode writes were disabled. The new environment uses Python 3.12.14 rather than the archived snapshot's 3.12.13; remaining dependencies fall back to the validated external clean environment. This is the closest recovered component stack, not a claim to have restored the original OS/process state. No package download, GitHub connection or public dependency-file edit occurred. An initial old-package import attempted an unavailable fallback settings directory; precreating its external YOLO_CONFIG_DIR resolved that infrastructure error before inference.

The reconstructed historical stream exactly equals the current stream despite independently recomputed proposals and older Pillow. Frozen per-detection coordinates were not retained, so recovered historical inference is supporting evidence, not a recovered original detection log. The frozen row remains anchored by records.csv SHA256 8c424482c06baabc71f28e18c4bd35f1a687192dbe308d09271c4f457fa9d9e6.

### Classification and release interpretation

| Category | Conclusion |
|---|---|
| A. Runtime nondeterminism | Not observed: repeated outputs are identical within each tested stack and batch mode. |
| B. Dependency/version drift | Confirmed at Ultralytics package level by a single-package controlled comparison. |
| C. Matching/tie boundary | IoU threshold crossing confirmed; evaluator/tie/order change excluded. |
| D. Decoded-image difference | Excluded for current versus recovered-stack reconstruction/decoding. Original frozen stream is unavailable. |
| E. Historical inconsistency | No supporting evidence: all frozen detection metrics recover under the older package. |
| F. Unresolved | Exact original process/stream are unavailable; the specific upstream internal arithmetic edit is not uniquely isolated. The observed one-image mismatch is explained. |

The frozen **+2.617401515 pp** remains defensible as the historical recorded result. The current full replay measures **+2.622203666 pp**; both round to **+2.62 pp**. Preserve these as environment-specific measurements, not interchangeable exact values. CLAIMS.md and all frozen rows remain unchanged; no regenerated row was replaced by historical output.

Describe the full current-environment verification as **547/548 exact adaptive detection-outcome matches and 548/548 exact adaptive encoded byte-count matches**, plus 548/548 matched-JPEG bytes/qualities/outcomes. Do not shorten byte-count agreement to 548/548 bit-for-bit identity: original full-dataset TAC streams were not retained. The one image has an explained deterministic package-sensitive IoU boundary; it now matches the frozen row under recovered historical components. This follow-up is not a full 548-image rerun under that stack and does not retroactively mark the current-environment exact replay as passed.

The discrepancy investigation is complete. Learned V2 verification can proceed as a separate step with its established truncated_6sigma semantics, exact frozen checkpoint/design/thresholds and explicit dependency inventory. This does not certify any V2 metric or authorize silently accepting its mismatches. Learned V2 has not been run here. No scientific code change is justified to force the historical result; any future dependency pin is a separate release packaging decision.

External evidence: codec.private.json; current-inference.private.json; historical-inference.private.json; current-torch-old-ultralytics-inference.private.json; current-controlled-inference.private.json; initial-boundary-analysis.private.json; root-cause.private.json; upstream-source-diff.private.txt; historical-env-build.private.json; per-run prediction/assignment JSONs; model-head arrays; reconstructed TAC streams. Diagnostic script sources and their exact commands remain alongside that external directory. Public documentation contains only the compact findings; GT coordinates and raw predictions remain external.
