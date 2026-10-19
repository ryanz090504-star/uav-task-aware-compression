# VisDrone dataset preparation

This repository does not distribute the VisDrone dataset. Users must obtain VisDrone2019-DET training and validation data from the official project under the terms applicable to that exact download. No data download or conversion was performed during this audit.

## What is currently included?

| Material | Public repository status |
|---|---|
| Original images, dataset ZIPs, per-image annotation/YOLO label directories, crops and previews | Not included. |
| Dataset configuration and conversion scripts | Included as project preparation code. |
| Aggregate numerical results, aggregate person counts and frozen image/sequence split lists | Included as compact evidence/provenance. Per-image result CSVs are excluded. |
| Validation and tune prediction JSONL | Not distributed in Git or v0.1.0 assets; they contain GT boxes and are excluded by release policy. |
| Example images | examples/ is empty. Use synthetic/generated examples for now; keep VisDrone images, label examples, overlays and crops excluded unless rights are confirmed. |

Releasing fewer images or moving annotations into a supplementary asset does not by itself establish permission. Aggregate summaries, image identifiers, copied annotations and model weights are distinct materials whose rights must not be assumed identical.

## Official acquisition and terms

Start with the [official download page](https://aiskyeye.com/download/). It identifies its object-detection data as the same as VisDrone2019. Obtain the DET train and val archives through the official route; do not substitute mirrors or another task/year silently. Preserve the download date, archive hashes and supplied terms privately or in a rights-cleared acquisition record.

The [official privacy/copyright page](https://aiskyeye.com/data-protection/) describes academic use and CC BY-NC-SA 3.0 for the dataset named there, VisDrone2021. This is evidence of restrictions, not a verified license grant for every 2019-DET artifact. Confirm exact 2019 terms, annotation redistribution, examples and model-related conditions before publication. Do not label this project or all outputs CC-licensed solely from that page. Citation alone is insufficient permission.

## Local layout and preparation

After separately acquiring and extracting the official archives, place them under the repository's ignored local data directory:

```text
data/VisDrone/
  VisDrone2019-DET-train/
    images/
    annotations/
  VisDrone2019-DET-val/
    images/
    annotations/
```

From the repository root, the existing preparation scripts are:

```text
python scripts/convert_visdrone_train.py
python scripts/convert_visdrone_val.py
```

These preparation commands were checked for help and clear missing-data errors; no VisDrone data was used or converted. With lawfully acquired local data, the train converter writes labels alongside train images and JSON metadata under metadata/train. The val converter copies validation images into images/val and writes labels/val plus metadata/val. Allow local disk space for that additional copy. The recorded validation experiment uses 548 images; verify the expected filenames and complete label coverage before any authorized reproduction.

Both scripts retain categories 1 (pedestrian) and 2 (people), merge them into YOLO class 0 (person), discard annotation rows whose score field is 0, and skip other categories. Coordinates are normalized by image dimensions. Metadata sidecars retain category, truncation and occlusion. This custom task is not the official multi-class/ignored-region VisDrone evaluation protocol. See CLAIMS.md; this document does not revise results or evaluation semantics.

configs/visdrone_person.yaml is the dataset config. Pass its path explicitly when a training script defaults to the original data/ location. Ultralytics dataset-root resolution can depend on its settings and working directory; verify the resolved local path rather than embedding a contributor's machine path in public metadata. Learned scripts also accept explicit train/eval image and label arguments. Do not publish locally generated data, resolved absolute paths or environment captures.

## Splits and reproduction evidence

Retained training/split manifests and frozen_design.json identify the allocator's train/tune/validation samples and the detector hash. Their separation rules and validation reuse are documented in CLAIMS.md. Do not replace these with a fresh random split while calling the result the same experiment.

GT-containing supplementary predictions are not part of v0.1.0. Historical source hashes are provenance only in RELEASE_MANIFEST.md; there is no asset restoration or hosting plan for this release. No dataset images are needed to inspect saved summaries; original model inference requires locally acquired images and matching detector/allocator artifacts.

## Citation draft

Credit the AISKYEYE/VisDrone dataset project and the task-specific paper: Dawei Du et al., *VisDrone-DET2019: The Vision Meets Drone Object Detection in Image Challenge Results*, ICCV Workshops, 2019. Use the [publisher's citation record](https://openaccess.thecvf.com/content_ICCVW_2019/html/VISDrone/Du_VisDrone-DET2019_The_Vision_Meets_Drone_Object_Detection_in_Image_Challenge_ICCVW_2019_paper.html) to finalize complete BibTeX; this audit's direct page fetch was unavailable, so no full author list or DOI is invented. The original dataset overview is [Vision Meets Drones: A Challenge](https://arxiv.org/abs/1804.07437), Pengfei Zhu et al., 2018. Follow any additional citation instructions accompanying the exact official download.

This draft supplies acquisition/preparation guidance, not authority to redistribute the dataset, example images, annotations or derived checkpoints.
