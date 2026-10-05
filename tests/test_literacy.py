from models.financial_literacy import FinancialLiteracyModel


def test_literacy_is_deterministic():
    profile = {
        "age": 30,
        "income": 60000,
        "savings": 10000,
        "expenses": 45000,
        "debt": 5000,
        "has_emergency_fund": False,
        "has_budget": True,
        "has_insurance": True,
    }
    model = FinancialLiteracyModel()
    assert model.analyze_profile(profile) == model.analyze_profile(profile)
