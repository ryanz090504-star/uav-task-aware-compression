> Historical audit snapshot. Current v0.1.0 scope supersedes distribution recommendations below: first-party code is prepared for AGPL-3.0 subject to final review; both checkpoints and all GT-containing assets remain unreleased. The defective report, pilot plotter and detector/pilot trainers have been removed. Current entry points are in docs/REPRODUCE.md. No new rights clearance is implied.

# Licensing and provenance audit — historical snapshot

Audit date: 2026-10-17. This is an evidence inventory and release recommendation, not a determination of legal rights. At that audit stage no project license had been selected; the current candidate includes standard AGPL-3.0 text subject to final ownership/integration review. Claims, numerical results and scientific code are unchanged.

## Scope and evidence limits

Reviewed the public Python source, tests, scripts, dependency declarations, result/config manifests and model provenance. Read the internal archive's training scripts, local third-party licenses, installed package METADATA/license files and two checkpoints using ZIP inspection plus pickletools opcode parsing. No checkpoint was deserialized or executed. No authenticated GitHub connection, remote repository operation or experiment occurred. Upstream licensing/dataset pages were consulted through public web search.

No contributor history, employment/IP assignments, institutional authorization, signed contributor declarations or commercial model license was supplied. A lack of copied-code markers is not proof of copyright ownership. Exact-file comparison against the archived third_party Python tree found no matches; it does not rule out adapted fragments or sources outside that tree.

## Apparent first-party material

All 16 Python files under src/task_aware_compression/ appear to be this project's implementation:

- Core/custom infrastructure: __init__.py, types.py, compressor.py, importance.py, dataset.py, metrics.py, synthetic.py, benchmark.py, cli.py, visualization.py and workflow.py.
- External interfaces: detector.py wraps Ultralytics; hevc.py invokes an external ffmpeg executable. Wrapping an API is distinct from vendoring its implementation; integration licensing still needs review.
- Learned/prototype components: learned_allocation.py, jpeg_proxy.py and tile_preprocessor.py use PyTorch operators to implement the local allocator, transform proxy and residual preprocessor. No imported PreprocessingICM/MMDetection implementation was found. jpeg_proxy.py contains conventional JPEG quantization tables and transform/color-conversion constants; their source/attribution should be confirmed by the author, rather than inferred from the absence of a header.

The three tests (test_core.py, test_jpeg_proxy.py, test_learned_allocation.py), conversion/evaluation scripts and public configuration appear project-authored. The report script has a documented English presentation edit. Dataset converters implement conventional VisDrone-to-YOLO conversion; that resemblance alone neither proves copying nor proves independent authorship. Obtain an author confirmation identifying any borrowed snippets, generated code, external contributions or employer/university rights before selecting a license.

No vendored package, third_party directory, external license header identifying copied implementation, or external implementation file was found in the public copy. No definitive copied/derived code finding can be made from this archive-only inspection. Scientific influence and use of a third-party training loss must be acknowledged separately from ownership of local source.

## Important dependencies

Versions below are recorded local archive metadata, not a newly resolved or validated environment. They are not guarantees about future releases. Package binaries and their license texts are not bundled in the public candidate.

| Package/component | Recorded version | Role | Available license evidence / release relevance |
|---|---|---|---|
| NumPy | 2.5.2 | Arrays, metrics, tile features, statistics | METADATA expression: BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0; bundled-library notices matter if binaries are later shipped. |
| Pillow | 12.3.0 | JPEG encoding/decoding, image I/O and drawing | MIT-CMU in local METADATA; codec/native-library notices need review for a bundled runtime. |
| PyTorch (torch) | 2.13.0 | Allocator parameters/autograd, JPEG proxy, checkpoint I/O | Local expression includes Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause, BSL-1.0 and MIT. Do not label the entire wheel with a single simplified license. |
| Ultralytics | 8.4.129 | YOLO inference/training and native detection loss | Local full license is GNU AGPL v3; METADATA says AGPL-3.0 and a v3-or-later classifier. Integration and model terms are material unresolved issues. |
| ultralytics-thop | 2.1.6 | Ultralytics transitive profiling dependency | Local metadata says AGPL-3.0; not directly imported by project code. |
| ultralytics-platform | 0.1.16 | Ultralytics transitive platform dependency on Python >=3.11 | Local expression AGPL-3.0-only; not directly imported by project code. Included in the environment lock. |
| opencv-python | 5.0.0.93 | Exact resize/letterbox for learned-v2 training | Local metadata says Apache 2.0 and lists LICENSE.txt / LICENSE-3RD-PARTY.txt. Review wheel/native third-party terms before binary redistribution. |
| Matplotlib | 3.11.1 | Workflow figures and report/plot scripts | Local metadata contains the Matplotlib license agreement and PSF classifier; retain actual notices if redistributing it. |
| torchvision | 0.28.0 | Ultralytics runtime dependency, listed CPU environment | Local metadata says BSD; native/image-codec components may carry additional notices. |
| PyYAML | 6.0.3 | Dataset/config support through Ultralytics; CPU requirement | MIT in local metadata; no direct public-code import. |
| pandas | 3.0.5 | Recorded CPU environment | BSD 3-Clause in metadata; no direct public-code import found. Candidate dependency simplification, not changed here. |
| tqdm | 4.70.0 | Training progress via dependencies; CPU requirement | Local license says MPL-2.0 AND MIT; file-specific terms require review if bundled or modified. No direct import found. |
| pytest | >=7.4 declared | Optional dev extra | Retained tests use standard-library unittest. Exact installed pytest license/version was not verified from the archive; no pytest code is shipped. |
| TensorBoard | >=2.14 declared | Optional monitoring extra | No public monitoring script/direct import retained; exact distribution license/version not verified here. Consider dropping the unused extra in a separate dependency review. |
| Python/stdlib | >=3.10 declared | Runtime, CLI, CSV/JSON and unit tests | No interpreter is bundled. Review the selected interpreter distribution if later packaging a runtime. |
| ffmpeg + libx265 | External, unpinned | Optional single-frame HEVC comparison | Executed through subprocess; no binaries included. FFmpeg's license depends on build options; libx265 is among GPL components. This is material if distributing binaries; codec/patent questions are not resolved by this audit. |

Other lockfile entries (HTTP libraries, plotting dependencies, system monitoring and native runtime packages) are transitive environment records, not a complete binary-distribution SBOM. Audit the actually resolved wheel set before shipping a bundled environment. Installing dependencies and redistributing their binaries are different actions; neither optional extras nor missing vendored source alone resolves AGPL scope. [FFmpeg licensing](https://ffmpeg.org/doxygen/trunk/md_LICENSE.html)

## Ultralytics: three distinct uses

1. **Importing the package.** detector.py constructs YOLO(model), calls predict, selects class 0, and converts boxes/confidences into project types. train_visdrone_gpu.py calls YOLO.train. The allocator training scripts import DEFAULT_CFG_DICT and v8DetectionLoss, freeze YOLO parameters, differentiate through its native loss, and use detector outputs for distillation/background constraints. Evaluation uses predictor settings and raw-head hooks. This is substantive runtime integration, not a name-only reference.
2. **Using upstream weights.** detector.py defaults to yolov8n.pt. Loading that name can cause the dependency to obtain upstream weights. No upstream weights are included. The archive's early training script starts from yolov8n.pt, followed by fine-tuning/continuation. Acquisition, exact upstream hash/version and applicable model terms still need documentation.
3. **Distributing the project detector.** The named VisDrone checkpoint is a separately redistributed fine-tuned artifact, not merely an import. It contains serialized Ultralytics model classes, training arguments and private paths. Its exact initialization chain, upstream rights, author authorization, dataset-related restrictions and metadata-cleaning procedure remain unresolved.

The vendor describes AGPL and commercial licensing routes and applies its guidance to custom-trained models. This is the vendor's position, not an independent conclusion that every project file is automatically subject to one license. No commercial grant or final integration-license decision was available. Do not claim that an optional import, exporting weights or changing file format removes applicable obligations. [Ultralytics licensing](https://www.ultralytics.com/license)

## Dataset and third-party research material

No raw VisDrone image/archive or per-image label directory is present. The repository does contain derived measurements, image identifiers, sequence/split lists, GT counts and training history. Its supplementary-asset manifest identifies prediction files containing ground-truth boxes. Those assets require their own rights decision; moving them outside Git does not clear redistribution. See DATASET.md.

PreprocessingICM/MMDetection source, import paths and reproduction adapters are absent. The archive's PreprocessingICM and two bundled MMDetection copies have Apache-2.0 license texts. Current retention of historical names does not justify copying the third-party tree or imposing unrelated notices on original code. If code is later included, inspect the exact files and notices and comply with applicable license/attribution/change-notice requirements. [Apache-2.0, section 4](https://www.apache.org/licenses/LICENSE-2.0)

## Script simplification recommendation — no deletions made

| Script | Recommendation for minimal v0.1.0 | Reason |
|---|---|---|
| convert_visdrone_train.py / convert_visdrone_val.py | Keep | Local preparation from officially acquired data. |
| strict_evaluation.py | Keep | Historical saved-record analysis and helpers imported elsewhere. |
| train_rate_allocator_v2.py / evaluate_rate_allocator_v2.py | Keep | V2 training/evaluation path; checkpoint and data acquisition remain prerequisites. |
| bootstrap_saved_detections.py | Keep | Recomputes uncertainty from approved/restored saved detections. |
| train_rate_allocator.py | Keep for now as an imported dependency | V2 imports shared path/sampling/serialization/report/tensor helpers. Removing it would break V2. A later authorized refactor can isolate helpers and remove the pilot entry point. |
| train_visdrone_gpu.py | Keep as an incomplete training recipe | Useful provenance; does not alone reproduce the entire initialization chain. |
| generate_experiment_report.py | Recommend exclude from the minimal public script set | Not needed for frozen numerical evidence; render_table references undefined paired_best. Its historical text/heuristics are not the claims authority. No fix recommended in this pass. |
| plot_rate_allocation.py | Recommend exclude | Targets the earlier pilot schema and pilot conclusion; not a V2 plotter. Repair is unnecessary for the frozen claims. |

If the report script is excluded, retain its historical paired-mean/normal-interval definition in CLAIMS.md and document an offline saved-record calculation; avoid implying that report generation is required to verify the headline. No source/test/script/evidence file was removed by this audit.

## Publication gate

Obtain author/IP confirmation, settle the integration/project license, approve dataset-derived asset rights, document the detector chain, and approve a metadata-clean model format before publishing corresponding components. Numerical summaries are the least encumbered candidate evidence, but this audit does not grant rights over them. The table in MODEL_PROVENANCE.md records item-by-item recommendations.
