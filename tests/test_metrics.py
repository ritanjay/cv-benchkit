"""Unit tests for cv_benchkit.metrics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from cv_benchkit.metrics import (
    compute_confusion_matrix,
    compute_giou,
    compute_iou,
)


class TestMetrics:
    """Test suite for IoU, GIoU, and confusion matrix."""

    def test_iou_identical_boxes(self) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 10.0], [10.0, 10.0, 20.0, 20.0]])
        iou = compute_iou(boxes, boxes)
        assert iou.shape == (2, 2)
        np.testing.assert_allclose(np.diag(iou), [1.0, 1.0])
        assert iou[0, 1] == 0.0
        assert iou[1, 0] == 0.0

    def test_iou_1d_input(self) -> None:
        b1 = np.array([0.0, 0.0, 10.0, 10.0])
        b2 = np.array([0.0, 0.0, 10.0, 10.0])
        iou = compute_iou(b1, b2)
        assert iou.shape == (1, 1)
        np.testing.assert_allclose(iou[0, 0], 1.0)

    def test_iou_partial_overlap(self) -> None:
        b1 = np.array([[0.0, 0.0, 10.0, 10.0]])
        b2 = np.array([[5.0, 0.0, 15.0, 10.0]])
        iou = compute_iou(b1, b2)
        np.testing.assert_allclose(iou[0, 0], 1.0 / 3.0)

    def test_iou_disjoint_boxes(self) -> None:
        b1 = np.array([[0.0, 0.0, 5.0, 5.0]])
        b2 = np.array([[10.0, 10.0, 20.0, 20.0]])
        iou = compute_iou(b1, b2)
        assert iou[0, 0] == 0.0

    def test_empty_inputs_iou(self) -> None:
        empty = np.empty((0, 4))
        b = np.array([[0.0, 0.0, 10.0, 10.0]])
        assert compute_iou(empty, b).shape == (0, 1)
        assert compute_iou(b, empty).shape == (1, 0)
        assert compute_iou(empty, empty).shape == (0, 0)

    def test_iou_empty_1d_array_as_first_argument(self) -> None:
        boxes_b = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
        result = compute_iou(np.empty(0), boxes_b)
        assert result.shape == (0, len(boxes_b))

    def test_iou_empty_1d_array_as_second_argument(self) -> None:
        boxes_a = np.array([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
        result = compute_iou(boxes_a, np.empty(0))
        assert result.shape == (len(boxes_a), 0)

    def test_iou_rejects_boxes_without_four_coordinates(self) -> None:
        boxes_b = np.array([[0.0, 0.0, 10.0, 10.0]])
        with np.testing.assert_raises_regex(
            ValueError, r"Boxes must have 4 coordinates \[x_min, y_min, x_max, y_max\]"
        ):
            compute_iou(np.array([[0, 0, 10]]), boxes_b)

    @pytest.mark.parametrize("metric", [compute_iou, compute_giou], ids=["iou", "giou"])
    @pytest.mark.parametrize("shape", [(2, 3, 4), (2, 2, 3, 4), (0, 3, 4)])
    @pytest.mark.parametrize("input_index", [0, 1], ids=["boxes-a", "boxes-b"])
    def test_metrics_reject_box_arrays_with_more_than_two_dimensions(
        self,
        metric: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
        shape: tuple[int, ...],
        input_index: int,
    ) -> None:
        valid_boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        inputs = [valid_boxes, valid_boxes]
        inputs[input_index] = np.zeros(shape)

        with pytest.raises(
            ValueError, match=rf"Boxes array must be 1D or 2D, got {len(shape)}D array"
        ):
            metric(*inputs)

    @pytest.mark.parametrize("start_axis", [0, 1], ids=["x-min", "y-min"])
    @pytest.mark.parametrize("input_index", [0, 1], ids=["boxes-a", "boxes-b"])
    def test_iou_rejects_inverted_coordinates(self, start_axis: int, input_index: int) -> None:
        valid_boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        inverted_batch = np.repeat(valid_boxes, 2, axis=0)
        inverted_batch[1, start_axis] = 11.0
        inputs = [valid_boxes, valid_boxes]
        inputs[input_index] = inverted_batch

        with pytest.raises(
            ValueError, match=r"Box coordinates must satisfy x_min <= x_max and y_min <= y_max"
        ):
            compute_iou(*inputs)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
    @pytest.mark.parametrize("coord_index", [0, 1, 2, 3])
    @pytest.mark.parametrize("input_index", [0, 1], ids=["boxes-a", "boxes-b"])
    def test_iou_rejects_non_finite_coordinates(
        self, bad_value: float, coord_index: int, input_index: int
    ) -> None:
        valid_boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        bad_boxes = np.repeat(valid_boxes, 2, axis=0)
        bad_boxes[1, coord_index] = bad_value
        inputs = [valid_boxes, valid_boxes]
        inputs[input_index] = bad_boxes

        with pytest.raises(ValueError, match=r"Bounding box coordinates must be finite"):
            compute_iou(*inputs)

    def test_giou_rejects_non_finite_coordinates(self) -> None:
        valid_boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        bad_boxes = np.array([[0.0, np.nan, 10.0, 10.0]])

        with pytest.raises(ValueError, match=r"Bounding box coordinates must be finite"):
            compute_giou(valid_boxes, bad_boxes)

    def test_iou_accepts_equal_minimum_and_maximum_coordinates(self) -> None:
        degenerate_boxes = np.array([[0.0, 0.0, 0.0, 10.0], [0.0, 0.0, 10.0, 0.0]])
        valid_box = np.array([0.0, 0.0, 10.0, 10.0])

        np.testing.assert_array_equal(compute_iou(degenerate_boxes, valid_box), [[0.0], [0.0]])

    def test_giou_empty_1d_array(self) -> None:
        boxes_b = np.array([[0.0, 0.0, 10.0, 10.0]])
        assert compute_giou(np.empty(0), boxes_b).shape == (0, 1)
        assert compute_giou(boxes_b, np.empty(0)).shape == (1, 0)
        with np.testing.assert_raises_regex(
            ValueError, r"Boxes must have 4 coordinates \[x_min, y_min, x_max, y_max\]"
        ):
            compute_giou(np.array([[0, 0, 10]]), boxes_b)

    def test_giou_identical_boxes(self) -> None:
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        giou = compute_giou(boxes, boxes)
        np.testing.assert_allclose(giou[0, 0], 1.0)

    def test_giou_1d_and_empty(self) -> None:
        b1 = np.array([0.0, 0.0, 10.0, 10.0])
        b2 = np.array([0.0, 0.0, 10.0, 10.0])
        assert compute_giou(b1, b2).shape == (1, 1)

        empty = np.empty((0, 4))
        assert compute_giou(empty, b1).shape == (0, 1)
        assert compute_giou(b1, empty).shape == (1, 0)

    def test_giou_disjoint_boxes(self) -> None:
        b1 = np.array([[0.0, 0.0, 10.0, 10.0]])
        b2 = np.array([[20.0, 0.0, 30.0, 10.0]])
        giou = compute_giou(b1, b2)
        np.testing.assert_allclose(giou[0, 0], -1.0 / 3.0, atol=1e-5)

    def test_confusion_matrix_normal(self) -> None:
        gt_boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [50.0, 50.0, 60.0, 60.0],
            ]
        )
        gt_labels = np.array([0, 1, 0])

        pred_boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [21.0, 21.0, 31.0, 31.0],
                [80.0, 80.0, 90.0, 90.0],
            ]
        )
        pred_labels = np.array([0, 0, 1])

        cm = compute_confusion_matrix(
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            num_classes=2,
            iou_threshold=0.5,
        )
        assert cm.shape == (3, 3)
        assert cm.sum() > 0

    def test_confusion_matrix_edge_cases(self) -> None:
        # Both empty
        cm_empty = compute_confusion_matrix(
            gt_labels=np.array([]),
            pred_labels=np.array([]),
            gt_boxes=np.empty((0, 4)),
            pred_boxes=np.empty((0, 4)),
            num_classes=2,
        )
        assert cm_empty.shape == (3, 3)
        assert cm_empty.sum() == 0

        # Empty GT (all predictions are false positives)
        cm_no_gt = compute_confusion_matrix(
            gt_labels=np.array([]),
            pred_labels=np.array([0, 1]),
            gt_boxes=np.empty((0, 4)),
            pred_boxes=np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            num_classes=2,
        )
        assert cm_no_gt[2, 0] == 1  # background row, class 0 col
        assert cm_no_gt[2, 1] == 1  # background row, class 1 col

        # Empty Pred (all GTs are false negatives)
        cm_no_pred = compute_confusion_matrix(
            gt_labels=np.array([0, 1]),
            pred_labels=np.array([]),
            gt_boxes=np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            pred_boxes=np.empty((0, 4)),
            num_classes=2,
        )
        assert cm_no_pred[0, 2] == 1  # class 0 row, background col
        assert cm_no_pred[1, 2] == 1  # class 1 row, background col

    @pytest.mark.parametrize("iou_threshold", [-0.1, 1.5, -1.0, 2.0])
    def test_confusion_matrix_rejects_invalid_iou_threshold(self, iou_threshold: float) -> None:
        with pytest.raises(ValueError, match=r"iou_threshold must be between 0\.0 and 1\.0"):
            compute_confusion_matrix(
                gt_labels=np.array([0]),
                pred_labels=np.array([0]),
                gt_boxes=np.array([[0.0, 0.0, 10.0, 10.0]]),
                pred_boxes=np.array([[0.0, 0.0, 10.0, 10.0]]),
                num_classes=2,
                iou_threshold=iou_threshold,
            )

    @pytest.mark.parametrize("iou_threshold", [0.0, 1.0])
    def test_confusion_matrix_accepts_boundary_iou_threshold(self, iou_threshold: float) -> None:
        cm = compute_confusion_matrix(
            gt_labels=np.array([0]),
            pred_labels=np.array([0]),
            gt_boxes=np.array([[0.0, 0.0, 10.0, 10.0]]),
            pred_boxes=np.array([[0.0, 0.0, 10.0, 10.0]]),
            num_classes=2,
            iou_threshold=iou_threshold,
        )
        assert cm.shape == (3, 3)


@pytest.mark.parametrize("num_classes", [0, -1])
def test_confusion_matrix_rejects_nonpositive_classes(num_classes: int) -> None:
    with pytest.raises(ValueError, match="num_classes must be a positive integer"):
        compute_confusion_matrix(
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.empty((0, 4)),
            np.empty((0, 4)),
            num_classes,
        )


@pytest.mark.parametrize("num_labels,num_boxes", [(2, 1), (1, 2), (0, 1), (1, 0)])
@pytest.mark.parametrize("side", ["ground-truth", "predicted"])
def test_confusion_matrix_rejects_mismatched_lengths(
    num_labels: int,
    num_boxes: int,
    side: str,
) -> None:
    labels = np.zeros(num_labels, dtype=int)
    boxes = np.zeros((num_boxes, 4))
    empty_labels = np.array([], dtype=int)
    empty_boxes = np.empty((0, 4))
    message = f"Number of {side} labels must match number of {side} boxes"
    with pytest.raises(ValueError, match=message):
        if side == "ground-truth":
            compute_confusion_matrix(labels, empty_labels, boxes, empty_boxes, 2)
        else:
            compute_confusion_matrix(empty_labels, labels, empty_boxes, boxes, 2)


@pytest.mark.parametrize(
    "gt_labels,pred_labels",
    [
        (np.array([-1]), np.array([0])),
        (np.array([0]), np.array([-1])),
        (np.array([2]), np.array([0])),  # num_classes=2 → valid [0,1]
        (np.array([0]), np.array([2])),
        (np.array([-1, 0]), np.array([0])),
        (np.array([0]), np.array([0, 5])),
    ],
)
def test_confusion_matrix_rejects_out_of_range_labels(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> None:
    """Out-of-range class labels must raise ValueError."""
    num_classes = 2
    gt_boxes = np.zeros((len(gt_labels), 4))
    pred_boxes = np.zeros((len(pred_labels), 4))
    with pytest.raises(ValueError, match=r"Class labels must be in range \[0, 1\]"):
        compute_confusion_matrix(
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            num_classes=num_classes,
        )


def test_confusion_matrix_accepts_boundary_labels() -> None:
    """Labels at 0 and num_classes-1 are valid."""
    cm = compute_confusion_matrix(
        gt_labels=np.array([0, 1]),
        pred_labels=np.array([1, 0]),
        gt_boxes=np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        pred_boxes=np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        num_classes=2,
    )
    assert cm.shape == (3, 3)


class TestConfusionMatrixValidation:
    """Regression tests for confusion matrix input validation (issue #64)."""

    @pytest.mark.parametrize("num_classes", [0, -1])
    @pytest.mark.parametrize("nonempty", [True, False], ids=["nonempty", "empty"])
    def test_nonpositive_num_classes_raises_expected_error(
        self, num_classes: int, nonempty: bool
    ) -> None:
        if nonempty:
            labels = np.array([0])
            boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        else:
            labels = np.array([], dtype=int)
            boxes = np.empty((0, 4))

        with pytest.raises(
            ValueError, match=rf"num_classes must be a positive integer, got {num_classes}"
        ):
            compute_confusion_matrix(labels, labels, boxes, boxes, num_classes)

    @pytest.mark.parametrize("shape", [(3, 1), (3, 2)], ids=["2d-col", "2d-multi"])
    @pytest.mark.parametrize("side", ["gt", "pred"])
    def test_multidimensional_labels_rejected(self, shape: tuple[int, int], side: str) -> None:
        valid_labels = np.array([0, 1, 0])
        invalid_labels = np.zeros(shape, dtype=int)
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [40.0, 40.0, 50.0, 50.0],
            ]
        )
        gt_labels = invalid_labels if side == "gt" else valid_labels
        pred_labels = invalid_labels if side == "pred" else valid_labels

        with pytest.raises(ValueError, match=r"gt_labels and pred_labels must be 1D arrays"):
            compute_confusion_matrix(gt_labels, pred_labels, boxes, boxes, num_classes=2)

    def test_valid_1d_labels_exact_confusion_matrix(self) -> None:
        gt_boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [50.0, 50.0, 60.0, 60.0],
            ]
        )
        gt_labels = np.array([0, 1, 0])
        pred_boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [80.0, 80.0, 90.0, 90.0],
            ]
        )
        pred_labels = np.array([0, 0, 1])

        cm = compute_confusion_matrix(
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            num_classes=2,
            iou_threshold=0.5,
        )
        expected = np.array(
            [
                [1, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(cm, expected)

    def test_empty_and_out_of_range_behavior_unchanged(self) -> None:
        cm_empty = compute_confusion_matrix(
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.empty((0, 4)),
            np.empty((0, 4)),
            num_classes=2,
        )
        np.testing.assert_array_equal(cm_empty, np.zeros((3, 3), dtype=np.int64))

        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        with pytest.raises(ValueError, match=r"Class labels must be in range \[0, 1\]"):
            compute_confusion_matrix(np.array([2]), np.array([0]), boxes, boxes, num_classes=2)
