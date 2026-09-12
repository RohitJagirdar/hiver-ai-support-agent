"""
Smoke Test Suite for Hiver AI Support Agent.
Verifies that all components (Schemas, Retriever, Guardrails, Agent, Baselines) initialize
and run synchronously without errors.
"""

import unittest
from src.config import IntentCategory, ActionDecision, AgentResolutionOutput, UrgencyLevel
from src.retriever import load_retriever_from_disk
from src.agent import HiverSupportAgent
from src.baselines import TrivialBaseline, SimpleZeroShotBaseline


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = HiverSupportAgent(mock_mode=True)
        cls.trivial_baseline = TrivialBaseline()
        cls.zero_shot_baseline = SimpleZeroShotBaseline(cls.agent)

    def test_pydantic_schema_validation(self):
        """Ensures AgentResolutionOutput enforces field types and validation."""
        valid_output = AgentResolutionOutput(
            predicted_intent=IntentCategory.HARDWARE_ISSUES,
            confidence_score=0.95,
            action=ActionDecision.AUTO_HANDLE,
            escalation_reason="Verified hardware resolution.",
            draft_reply="Try a force restart.",
            urgency_level=UrgencyLevel.MEDIUM,
            retrieved_reference_ids=[1, 2]
        )
        self.assertEqual(valid_output.predicted_intent, IntentCategory.HARDWARE_ISSUES)
        self.assertEqual(valid_output.confidence_score, 0.95)

    def test_agent_auto_handle_workflow(self):
        """Tests standard troubleshooting query execution."""
        query = "My iPhone screen went black and won't turn on"
        output = self.agent.process_query(query)
        self.assertIsInstance(output, AgentResolutionOutput)
        self.assertEqual(output.predicted_intent, IntentCategory.HARDWARE_ISSUES)
        self.assertEqual(output.action, ActionDecision.AUTO_HANDLE)
        self.assertIn("force restart", output.draft_reply.lower())

    def test_agent_deterministic_escalation_guardrail(self):
        """Tests immediate short-circuit escalation for legal / lawsuit threats."""
        query = "I am suing Apple for gross negligence and contacting my lawyer"
        output = self.agent.process_query(query)
        self.assertIsInstance(output, AgentResolutionOutput)
        self.assertEqual(output.action, ActionDecision.ESCALATE_TO_HUMAN)
        self.assertIn("guardrail", output.escalation_reason.lower())

    def test_baselines_execution(self):
        """Ensures both baselines return valid AgentResolutionOutput."""
        query = "My battery drains in 2 hours on my iPhone"
        out_b1 = self.trivial_baseline.process_query(query)
        self.assertIsInstance(out_b1, AgentResolutionOutput)
        self.assertEqual(out_b1.predicted_intent, IntentCategory.HARDWARE_ISSUES)

        out_b2 = self.zero_shot_baseline.process_query(query)
        self.assertIsInstance(out_b2, AgentResolutionOutput)


if __name__ == "__main__":
    unittest.main()
