import pytest
from fastapi.testclient import TestClient


class FakeQA:
    def answer_question(self, question, context, max_answer_length=100):
        return {
            "answer": "100 million",
            "confidence": 0.9,
            "start_position": 12,
            "end_position": 23,
        }

    def extract_financial_metrics(self, context):
        return {"revenue": "100 million"}

    def batch_answer(self, questions, context):
        return [{"question": q, "answer": "100 million", "confidence": 0.9, "start_position": 0, "end_position": 12} for q in questions]


def test_openapi_and_health(monkeypatch):
    import api.main as main
    from api.main import app

    monkeypatch.setattr(main, "qa_model", FakeQA())
    monkeypatch.setattr(main, "literacy_model", main.FinancialLiteracyModel())

    with TestClient(app) as client:
        openapi = client.get("/openapi.json")
        assert openapi.status_code == 200
        schema = openapi.json()["paths"]["/api/v1/financial-qa"]["post"]
        assert schema["requestBody"]["content"]["application/json"]
        assert client.get("/health").status_code == 200


def test_qa_body(monkeypatch):
    import api.main as main
    from api.main import app

    monkeypatch.setattr(main, "qa_model", FakeQA())
    monkeypatch.setattr(main, "literacy_model", main.FinancialLiteracyModel())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/financial-qa",
            json={"question": "What was revenue?", "context": "Revenue was 100 million."},
        )
        assert response.status_code == 200
        assert response.json()["answer"] == "100 million"


def test_literacy_body(monkeypatch):
    import api.main as main
    from api.main import app

    monkeypatch.setattr(main, "qa_model", FakeQA())
    monkeypatch.setattr(main, "literacy_model", main.FinancialLiteracyModel())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/financial-literacy",
            json={
                "user_profile": {
                    "age": 28, "income": 60000, "savings": 10000,
                    "expenses": 45000, "debt": 5000,
                    "has_emergency_fund": False,
                    "has_budget": True,
                    "has_insurance": True,
                },
                "question": "Как начать копить?",
            },
        )
        assert response.status_code == 200
        assert "literacy_score" in response.json()


def test_metrics_context_is_json_body(monkeypatch):
    import api.main as main
    from api.main import app

    monkeypatch.setattr(main, "qa_model", FakeQA())
    monkeypatch.setattr(main, "literacy_model", main.FinancialLiteracyModel())

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/financial-qa/extract-metrics",
            json={"context": "Revenue was 100 million."},
        )
        assert response.status_code == 200
        assert "metrics" in response.json()
