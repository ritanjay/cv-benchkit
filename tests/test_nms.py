"""Unit tests for compute_nms."""

from __future__ import annotations

import numpy as np
import pytest

from cv_benchkit import compute_iou, compute_nms


def _reference_nms(
    boxes: np.ndarray, scores: np.ndarray, threshold: float, class_ids: np.ndarray | None = None
) -> list[int]:
    """Straightforward O(N^2) greedy NMS used to cross-check the implementation."""
    iou = compute_iou(boxes, boxes)
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    suppressed: set[int] = set()
    keep: list[int] = []
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in order:
            same_class = class_ids is None or class_ids[i] == class_ids[j]
            if j != i and same_class and iou[i, j] > threshold:
                suppressed.add(j)
    return keep


class TestBasic:
    def test_overlapping_boxes_are_suppressed(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=float)
        scores = np.array([0.9, 0.8, 0.7])
        result = compute_nms(boxes, scores, iou_threshold=0.5)
        np.testing.assert_array_equal(result, [0, 2])
        assert result.dtype == np.int64

    def test_lower_scored_box_is_the_one_suppressed(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
        np.testing.assert_array_equal(compute_nms(boxes, np.array([0.3, 0.9])), [1])

    def test_non_overlapping_boxes_are_all_kept_in_score_order(self) -> None:
        boxes = np.array([[0, 0, 1, 1], [5, 5, 6, 6], [10, 10, 11, 11]], dtype=float)
        scores = np.array([0.2, 0.9, 0.5])
        np.testing.assert_array_equal(compute_nms(boxes, scores), [1, 2, 0])

    def test_identical_boxes_keep_only_the_highest_score(self) -> None:
        boxes = np.tile([[2.0, 2.0, 8.0, 8.0]], (4, 1))
        scores = np.array([0.1, 0.7, 0.4, 0.6])
        np.testing.assert_array_equal(compute_nms(boxes, scores), [1])

    def test_equal_scores_keep_input_order(self) -> None:
        boxes = np.tile([[0.0, 0.0, 4.0, 4.0]], (3, 1))
        np.testing.assert_array_equal(compute_nms(boxes, np.array([0.5, 0.5, 0.5])), [0])

    def test_single_box(self) -> None:
        result = compute_nms(np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([0.3]))
        np.testing.assert_array_equal(result, [0])
        assert result.dtype == np.int64

    def test_suppression_is_strictly_greater_than_threshold(self) -> None:
        # IoU of these two boxes is exactly 0.5.
        boxes = np.array([[0, 0, 2, 1], [0, 0, 1, 1]], dtype=float)
        assert compute_iou(boxes[:1], boxes[1:])[0, 0] == 0.5
        scores = np.array([0.9, 0.8])
        np.testing.assert_array_equal(compute_nms(boxes, scores, iou_threshold=0.5), [0, 1])
        np.testing.assert_array_equal(compute_nms(boxes, scores, iou_threshold=0.49), [0])

    def test_threshold_bounds(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=float)
        scores = np.array([0.9, 0.8, 0.7])
        # 1.0 keeps everything; 0.0 suppresses any overlap but keeps disjoint boxes.
        np.testing.assert_array_equal(compute_nms(boxes, scores, iou_threshold=1.0), [0, 1, 2])
        np.testing.assert_array_equal(compute_nms(boxes, scores, iou_threshold=0.0), [0, 2])

    def test_suppressed_box_does_not_suppress_others(self) -> None:
        # B overlaps both A and C, A and C do not overlap. B is suppressed by A,
        # so C survives even though it overlaps B.
        boxes = np.array([[0, 0, 10, 10], [4, 0, 14, 10], [8, 0, 18, 10]], dtype=float)
        scores = np.array([0.9, 0.8, 0.7])
        np.testing.assert_array_equal(compute_nms(boxes, scores, iou_threshold=0.3), [0, 2])

    def test_inputs_are_not_modified(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
        scores = np.array([0.2, 0.9])
        boxes_copy, scores_copy = boxes.copy(), scores.copy()
        compute_nms(boxes, scores)
        np.testing.assert_array_equal(boxes, boxes_copy)
        np.testing.assert_array_equal(scores, scores_copy)


class TestEmpty:
    @pytest.mark.parametrize("boxes", [np.empty((0, 4)), np.array([])])
    def test_empty_inputs_return_empty_int64(self, boxes: np.ndarray) -> None:
        result = compute_nms(boxes, np.array([]))
        assert result.shape == (0,)
        assert result.dtype == np.int64

    def test_empty_with_class_ids(self) -> None:
        result = compute_nms(np.empty((0, 4)), np.array([]), class_ids=np.array([], dtype=np.int64))
        assert result.shape == (0,)


class TestClassAware:
    def test_overlapping_boxes_of_different_classes_are_both_kept(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
        scores = np.array([0.9, 0.8])
        np.testing.assert_array_equal(compute_nms(boxes, scores), [0])
        np.testing.assert_array_equal(
            compute_nms(boxes, scores, class_ids=np.array([0, 1])), [0, 1]
        )

    def test_same_class_still_suppressed(self) -> None:
        boxes = np.array(
            [[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10], [1, 1, 11, 11]], dtype=float
        )
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        result = compute_nms(boxes, scores, class_ids=np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(result, [0, 2])


class TestAgainstReference:
    @pytest.mark.parametrize("threshold", [0.0, 0.3, 0.5, 0.7, 1.0])
    @pytest.mark.parametrize("class_aware", [False, True])
    def test_random_boxes_match_reference(self, threshold: float, class_aware: bool) -> None:
        rng = np.random.default_rng(7)
        xy = rng.uniform(0, 50, size=(60, 2))
        boxes = np.hstack([xy, xy + rng.uniform(1, 30, size=(60, 2))])
        scores = rng.uniform(0, 1, size=60)
        class_ids = rng.integers(0, 3, size=60) if class_aware else None
        result = compute_nms(boxes, scores, iou_threshold=threshold, class_ids=class_ids)
        assert result.tolist() == _reference_nms(boxes, scores, threshold, class_ids)
        assert np.all(np.diff(scores[result]) <= 0)


class TestValidation:
    @pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
    def test_invalid_threshold(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="iou_threshold"):
            compute_nms(np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([0.5]), iou_threshold=threshold)

    @pytest.mark.parametrize(
        "boxes",
        [
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.zeros((2, 3)),
            np.zeros((1, 2, 4)),
        ],
    )
    def test_boxes_must_be_n_by_4(self, boxes: np.ndarray) -> None:
        with pytest.raises(ValueError, match="boxes"):
            compute_nms(boxes, np.array([0.5]))

    @pytest.mark.parametrize(
        "scores", [np.array([0.5]), np.array([[0.5, 0.4]]), np.array([0.1, 0.2, 0.3])]
    )
    def test_scores_must_match_boxes(self, scores: np.ndarray) -> None:
        boxes = np.array([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]])
        with pytest.raises(ValueError, match="scores"):
            compute_nms(boxes, scores)

    def test_non_finite_scores(self) -> None:
        boxes = np.array([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]])
        with pytest.raises(ValueError, match="finite"):
            compute_nms(boxes, np.array([0.5, np.nan]))

    @pytest.mark.parametrize("class_ids", [np.array([0]), np.array([[0, 1]]), np.array([0, 1, 2])])
    def test_class_ids_must_match_boxes(self, class_ids: np.ndarray) -> None:
        boxes = np.array([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]])
        with pytest.raises(ValueError, match="class_ids"):
            compute_nms(boxes, np.array([0.5, 0.4]), class_ids=class_ids)

    @pytest.mark.parametrize(
        "bad", [[5.0, 0.0, 1.0, 1.0], [0.0, 0.0, np.inf, 1.0], [0.0, np.nan, 1.0, 1.0]]
    )
    def test_invalid_box_coordinates(self, bad: list[float]) -> None:
        boxes = np.array([[0.0, 0.0, 1.0, 1.0], bad])
        with pytest.raises(ValueError):
            compute_nms(boxes, np.array([0.5, 0.4]))
