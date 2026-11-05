import unittest
from datetime import datetime, timezone, timedelta

from app.services.cloudmorph_final_recommendation import build_final_recommendation


class FinalRecommendationTests(unittest.TestCase):
    def setUp(self):
        now = datetime.now(timezone.utc)
        self.rec_time = now.isoformat()
        self.scenario_time = (now + timedelta(seconds=1)).isoformat()

    def scenario(self, action, safety_status, high_pressure=False, memory_risk=False):
        return {
            "result": {
                "safety_assessment": {
                    "resource_optimizer_action": action,
                    "status": safety_status,
                    "high_resource_pressure": high_pressure,
                    "high_memory_risk": memory_risk,
                },
                "cost_comparison": {
                    "comparison_status": "SAVINGS",
                    "estimated_monthly_savings": 8.10,
                    "estimated_savings_percent": 20.0,
                },
                "pricing_metadata": {"provider_pricing_verified": False},
            }
        }

    def run_case(self, action, scenario):
        return build_final_recommendation(
            {"action": action, "priority": "MEDIUM"},
            scenario,
            {"workload_source": "Alibaba Cluster Trace 2017"},
            self.rec_time,
            self.scenario_time,
        )["final_recommendation"]["action"]

    def test_keep_is_never_overridden_by_cheaper_scenario(self):
        result = self.scenario("KEEP", "NOT_APPROVED_BY_WORKLOAD_OPTIMIZER")
        self.assertEqual(self.run_case("KEEP", result), "KEEP")

    def test_resize_is_review_only(self):
        result = self.scenario("RESIZE", "RESIZE_SUPPORTED_REQUIRES_MANUAL_VALIDATION")
        self.assertEqual(self.run_case("RESIZE", result), "RESIZE")

    def test_switch_requires_migration_review(self):
        result = self.scenario(
            "SWITCH",
            "SWITCH_REQUIRES_COMPATIBILITY_AND_MIGRATION_VALIDATION",
        )
        self.assertEqual(self.run_case("SWITCH", result), "SWITCH")

    def test_high_pressure_blocks_resize(self):
        result = self.scenario("RESIZE", "BLOCKED_HIGH_RESOURCE_PRESSURE", high_pressure=True)
        self.assertEqual(self.run_case("RESIZE", result), "HOLD")

    def test_missing_scenario_holds_resize(self):
        self.assertEqual(self.run_case("RESIZE", None), "HOLD")


if __name__ == "__main__":
    unittest.main()
