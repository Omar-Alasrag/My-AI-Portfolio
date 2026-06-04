from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_welcome():
    res = client.get("/")
    assert res.status_code == 200


def test_predict():
    res = client.get("/predict")
    assert res.status_code == 200
    assert res.text.lower().find("prediction") != -1
