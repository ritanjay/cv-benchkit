import json

import pytest

from cv_benchkit.cli import main


def test_filter_single_category_name(tmp_path):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"

    dataset = {
        "images": [{"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 20, 30, 40],
                "area": 1200,
                "iscrowd": 0,
            },
            {
                "id": 20,
                "image_id": 1,
                "category_id": 2,
                "bbox": [50, 60, 70, 80],
                "area": 5600,
                "iscrowd": 0,
            },
        ],
    }

    input_file.write_text(json.dumps(dataset), encoding="utf-8")

    result = main(
        [
            "filter",
            str(input_file),
            "-o",
            str(output_file),
            "--categories",
            "cat",
        ]
    )

    assert result == 0

    filtered = json.loads(output_file.read_text(encoding="utf-8"))

    assert len(filtered["annotations"]) == 1
    assert filtered["annotations"][0]["category_id"] == 1
    assert filtered["annotations"][0]["id"] == 10


def test_filter_single_category_id(tmp_path):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"

    dataset = {
        "images": [{"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 20,
                "image_id": 1,
                "category_id": 2,
                "bbox": [0, 0, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
        ],
    }

    input_file.write_text(json.dumps(dataset), encoding="utf-8")

    result = main(["filter", str(input_file), "-o", str(output_file), "--categories", "2"])

    assert result == 0
    filtered = json.loads(output_file.read_text(encoding="utf-8"))
    assert [ann["id"] for ann in filtered["annotations"]] == [20]


def test_filter_multiple_category_names(tmp_path):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"

    dataset = {
        "images": [{"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
            {"id": 3, "name": "bird"},
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 20,
                "image_id": 1,
                "category_id": 2,
                "bbox": [0, 0, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 30,
                "image_id": 1,
                "category_id": 3,
                "bbox": [0, 0, 30, 30],
                "area": 900,
                "iscrowd": 0,
            },
        ],
    }

    input_file.write_text(json.dumps(dataset), encoding="utf-8")

    result = main(
        [
            "filter",
            str(input_file),
            "-o",
            str(output_file),
            "--categories",
            "cat, bird",
        ]
    )

    assert result == 0
    filtered = json.loads(output_file.read_text(encoding="utf-8"))
    assert [ann["id"] for ann in filtered["annotations"]] == [10, 30]


def test_filter_min_and_max_area(tmp_path):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"

    dataset = {
        "images": [{"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 5, 5],
                "area": 25,
                "iscrowd": 0,
            },
            {
                "id": 20,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
            {
                "id": 30,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
        ],
    }

    input_file.write_text(json.dumps(dataset), encoding="utf-8")

    result = main(
        [
            "filter",
            str(input_file),
            "-o",
            str(output_file),
            "--min-area",
            "50",
            "--max-area",
            "200",
        ]
    )

    assert result == 0
    filtered = json.loads(output_file.read_text(encoding="utf-8"))
    assert [ann["id"] for ann in filtered["annotations"]] == [20]


def test_filter_drop_empty_images(tmp_path):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"

    dataset = {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "image2.jpg", "width": 640, "height": 480},
        ],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
    }

    input_file.write_text(json.dumps(dataset), encoding="utf-8")

    result = main(
        [
            "filter",
            str(input_file),
            "-o",
            str(output_file),
            "--drop-empty-images",
        ]
    )

    assert result == 0
    filtered = json.loads(output_file.read_text(encoding="utf-8"))

    assert [image["id"] for image in filtered["images"]] == [1]
    assert [ann["id"] for ann in filtered["annotations"]] == [10]


def test_filter_nonexistent_input_file(tmp_path, capsys):
    input_file = tmp_path / "missing.json"
    output_file = tmp_path / "output.json"

    result = main(
        [
            "filter",
            str(input_file),
            "-o",
            str(output_file),
        ]
    )

    captured = capsys.readouterr()

    assert result == 1
    assert "not found" in captured.err
    assert not output_file.exists()


def test_filter_invalid_json(tmp_path, capsys):
    input_file = tmp_path / "invalid.json"
    output_file = tmp_path / "output.json"

    input_file.write_text("{ invalid json }", encoding="utf-8")

    result = main(
        [
            "filter",
            str(input_file),
            "-o",
            str(output_file),
        ]
    )

    captured = capsys.readouterr()

    assert result == 1
    assert "invalid COCO input" in captured.err
    assert not output_file.exists()


def test_filter_nonexistent_category(tmp_path, capsys):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"

    dataset = {
        "images": [{"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
    }

    input_file.write_text(json.dumps(dataset), encoding="utf-8")

    result = main(
        [
            "filter",
            str(input_file),
            "-o",
            str(output_file),
            "--categories",
            "dog",
        ]
    )

    captured = capsys.readouterr()

    assert result == 1
    assert "category 'dog' not found" in captured.err
    assert not output_file.exists()


def test_filter_requires_output_argument(tmp_path):
    input_file = tmp_path / "input.json"
    input_file.write_text(
        json.dumps(
            {
                "images": [],
                "annotations": [],
                "categories": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["filter", str(input_file)])

    assert exc_info.value.code == 2


def test_filter_keeps_empty_images_by_default(tmp_path):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"

    dataset = {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "image2.jpg", "width": 640, "height": 480},
        ],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
    }

    input_file.write_text(json.dumps(dataset), encoding="utf-8")

    result = main(["filter", str(input_file), "-o", str(output_file)])

    assert result == 0
    filtered = json.loads(output_file.read_text(encoding="utf-8"))

    assert len(filtered["images"]) == 2
    assert len(filtered["annotations"]) == 1


def test_filter_min_area_only(tmp_path):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"

    dataset = {
        "images": [{"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 5, 5],
                "area": 25,
                "iscrowd": 0,
            },
            {
                "id": 20,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
        ],
    }

    input_file.write_text(json.dumps(dataset), encoding="utf-8")

    result = main(["filter", str(input_file), "-o", str(output_file), "--min-area", "50"])

    assert result == 0
    filtered = json.loads(output_file.read_text(encoding="utf-8"))
    assert [ann["id"] for ann in filtered["annotations"]] == [20]


def test_filter_max_area_only(tmp_path):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"

    dataset = {
        "images": [{"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 5, 5],
                "area": 25,
                "iscrowd": 0,
            },
            {
                "id": 20,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            },
        ],
    }

    input_file.write_text(json.dumps(dataset), encoding="utf-8")

    result = main(["filter", str(input_file), "-o", str(output_file), "--max-area", "50"])

    assert result == 0
    filtered = json.loads(output_file.read_text(encoding="utf-8"))
    assert [ann["id"] for ann in filtered["annotations"]] == [10]


def test_filter_invalid_output_directory(tmp_path, capsys):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "missing_dir" / "output.json"

    dataset = {
        "images": [{"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {
                "id": 10,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
    }

    input_file.write_text(json.dumps(dataset), encoding="utf-8")

    result = main(["filter", str(input_file), "-o", str(output_file)])

    captured = capsys.readouterr()

    assert result == 1
    assert "could not write output file" in captured.err
    assert not output_file.exists()
