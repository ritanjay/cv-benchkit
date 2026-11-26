# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2026-11-26

### Added
- **COCO Dataset Filter CLI (`cv-benchkit filter`)**: Added `filter` subcommand supporting category name/ID filtering (`--categories`), bounding-box area bounds (`--min-area`, `--max-area`), and `--drop-empty-images` while preserving COCO metadata and annotation IDs ([#91](https://github.com/udayraj1238/cv-benchkit/pull/91) by [@BhushanSonar10](https://github.com/BhushanSonar10)).
- **Greedy Non-Maximum Suppression (`compute_nms`)**: Implemented `compute_nms(boxes, scores, iou_threshold=0.5, class_ids=None)` with `O(N)` per-step `compute_iou` evaluation, stable score ordering, and optional class-aware mode ([#90](https://github.com/udayraj1238/cv-benchkit/pull/90) by [@ritanjay](https://github.com/ritanjay)).

### Fixed
- **Convert CLI Input Path Validation**: Validated input path types (`file` vs `directory`) across `coco`/`yolo`/`voc` conversion sources with explicit error messages ([#89](https://github.com/udayraj1238/cv-benchkit/pull/89) by [@pratuyshswain](https://github.com/pratuyshswain)).
- **Confusion Matrix Input Validation**: Reordered `num_classes <= 0` check before label-range validation and rejected multidimensional label arrays (`ndim > 1`) in `compute_confusion_matrix()` ([#92](https://github.com/udayraj1238/cv-benchkit/pull/92) by [@DevByPurva](https://github.com/DevByPurva)).

## [0.2.2] - 2026-11-24

### Added
- **Average Precision Calculation**: Implemented `compute_average_precision(precision, recall, method="all_points")` supporting both COCO / PASCAL VOC 2010+ all-point interpolation and PASCAL VOC 2007 11-point interpolation with monotonic precision envelopes ([#88](https://github.com/udayraj1238/cv-benchkit/pull/88) by [@Malaika-Abid](https://github.com/Malaika-Abid)).

### Fixed
- **COCO `iscrowd` Binary Validation**: Enforced strict `0` or `1` binary integer constraint on `iscrowd` field in `COCOAnnotation` using Pydantic validation ([#87](https://github.com/udayraj1238/cv-benchkit/pull/87) by [@haseemajaz](https://github.com/haseemajaz)).
- **COCO Polygon Segmentation Validation**: Added validation and normalization for COCO polygon segmentations, rejecting non-list or odd-length coordinate sequences ([#86](https://github.com/udayraj1238/cv-benchkit/pull/86) by [@LuciferAksh](https://github.com/LuciferAksh)).
- **Metrics Box Array Dimension Rejection**: Reject bounding box arrays with more than two dimensions (e.g. 3D tensors) in IoU computation functions with clear `ValueError` ([#85](https://github.com/udayraj1238/cv-benchkit/pull/85) by [@sharvmahajan](https://github.com/sharvmahajan)).

## [0.2.1] - 2026-11-21

### Added
- Precision-recall curve computation (`compute_precision_recall_curve`) ([#79](https://github.com/udayraj1238/cv-benchkit/pull/79) by [@Malaika-Abid](https://github.com/Malaika-Abid)).
- Comprehensive coordinate format documentation in README ([#82](https://github.com/udayraj1238/cv-benchkit/pull/82) by [@HiZhenyu](https://github.com/HiZhenyu)).

### Fixed
- Reject empty category names in YOLO and VOC converters ([#50](https://github.com/udayraj1238/cv-benchkit/pull/50) by [@sharvmahajan](https://github.com/sharvmahajan)).
- Validate non-negative bounding box dimensions in `BBox` ([#51](https://github.com/udayraj1238/cv-benchkit/pull/51) by [@Rainyyyyyyy111](https://github.com/Rainyyyyyyy111)).

## [0.2.0] - 2026-11-19

### Added
- Complete IoU metrics suite (DIoU, CIoU, confusion matrix).
- Dataset translators between COCO, YOLO, and VOC formats.
- CLI subcommands for dataset conversion and validation.
