
import cv2
import numpy as np
from fastapi.testclient import TestClient

import app as app_module

client = TestClient(app_module.app)


def make_test_image():
    """Create a small valid JPEG image in memory."""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", frame)

    assert success
    return encoded.tobytes()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ppe_detection_with_valid_image(monkeypatch):
    expected_result = {
        "total_persons": 0,
        "persons": {},
    }

    def fake_run_pipeline(frame):
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (64, 64, 3)
        return expected_result

    monkeypatch.setattr(
        app_module, "run_pipeline", fake_run_pipeline
    )

    response = client.post(
        "/ppe-detection",
        files={
            "image": (
                "test.jpg",
                make_test_image(),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200
    assert response.json() == expected_result


def test_ppe_detection_rejects_unsupported_type():
    response = client.post(
        "/ppe-detection",
        files={
            "image": (
                "test.txt",
                b"not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_ppe_detection_rejects_invalid_image():
    response = client.post(
        "/ppe-detection",
        files={
            "image": (
                "broken.jpg",
                b"not a valid JPEG",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400
