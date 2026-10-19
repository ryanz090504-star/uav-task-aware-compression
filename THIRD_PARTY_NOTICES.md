# Third-party notices

This v0.1.0 candidate is prepared for an AGPL-3.0 release of first-party code, subject to final ownership and licensing review. LICENSE contains the standard unmodified GNU AGPL version 3 text. No dataset, model checkpoint, installed runtime or external source tree is included in Git. A sanitized first-party allocator candidate is prepared separately and remains subject to model/dataset distribution review. This document does not grant rights over third-party material.

## Ultralytics / YOLO

The optional YOLO interface uses Ultralytics for inference. Learned V2 training also uses its native detection loss and frozen model outputs. Recorded Ultralytics 8.4.129 metadata identifies AGPL-3.0; the recorded ultralytics-thop and ultralytics-platform dependencies also identify AGPL terms. Review the resolved distributions and applicable model/software terms. Neither upstream YOLO weights nor this project's fine-tuned detector is bundled. [Ultralytics licensing](https://www.ultralytics.com/license).

## Material runtime dependencies

- NumPy: arrays/statistics; recorded metadata includes BSD-3-Clause and bundled permissive components.
- Pillow: image I/O and JPEG coding; recorded license MIT-CMU, with native codec notices where applicable.
- PyTorch: allocator/autograd; the recorded distribution has a mixed permissive license expression, detailed in LICENSE_AUDIT.md.
- OpenCV: native-resolution resize/letterbox; recorded opencv-python metadata identifies Apache 2.0 and a third-party notice file.
- Matplotlib: optional plots; preserve the selected distribution's license agreement if bundled.
- torchvision and PyYAML support the YOLO dependency environment. pandas/tqdm and other CPU lock entries are environment records, not redistributed source. Review exact installed wheel notices before any binary bundling.
- Optional HEVC comparison invokes an externally installed ffmpeg with libx265. No executable is supplied; license obligations depend on its build. [FFmpeg licensing](https://ffmpeg.org/doxygen/trunk/md_LICENSE.html).

Listing or installing a dependency is distinct from redistributing it. This source-only inventory is not a complete bundled-runtime SBOM and does not replace applicable license texts.

## VisDrone acknowledgement

Historical experiments use VisDrone2019-DET. Obtain data from the [official project](https://aiskyeye.com/download/) and review its [terms](https://aiskyeye.com/data-protection/). Cite Dawei Du et al., *VisDrone-DET2019: The Vision Meets Drone Object Detection in Image Challenge Results*, ICCV Workshops 2019, and the dataset overview [Vision Meets Drones: A Challenge](https://arxiv.org/abs/1804.07437). See DATASET.md for the custom merged-person conversion. Images, labels, annotations, crops and GT-containing detections are excluded. Citation is not redistribution permission.

## Historical research reference

PreprocessingICM: Guo Lu et al., *Preprocessing Enhanced Image Compression for Machine Vision*, IEEE TCSVT 2024, DOI 10.1109/TCSVT.2024.3441049. [Upstream reference](https://github.com/XingtongGe/PreprocessingICM). PreprocessingICM and its MMDetection copies were explored historically but are not vendored, imported or required by v0.1.0. No unrelated Apache notices are asserted for original code merely because these works are cited; revisit file-specific notices if implementation is later redistributed.
