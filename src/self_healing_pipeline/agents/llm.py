"""LLM integration with replay fixture support for deterministic testing."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM client with replay fixture support.

    In replay mode (CI): reads cached responses from fixtures.
    In record mode (manual test): saves responses to fixtures.
    In live mode: calls actual API (when key available).
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-6",
        fixtures_dir: Path | None = None,
        replay_mode: bool = True,
    ) -> None:
        """Init LLM client.

        Args:
            model_name: Claude model to use
            fixtures_dir: where to store/load fixture responses
            replay_mode: if True, use fixtures; if False, record new responses
        """
        self.model_name = model_name
        self.fixtures_dir = fixtures_dir or Path("tests/fixtures/claude_responses")
        self.replay_mode = replay_mode
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)

    def call(
        self,
        prompt: str,
        agent_type: str,
        incident_type: str,
        fixture_key: str | None = None,
    ) -> str:
        """Call LLM or fetch from fixture.

        Args:
            prompt: the prompt to send
            agent_type: which agent is calling (e.g., "threshold")
            incident_type: type of incident (e.g., "DRIFT")
            fixture_key: cache key for fixture lookup

        Returns:
            JSON response string
        """
        fixture_key = fixture_key or f"{agent_type}_{incident_type}"
        fixture_path = self.fixtures_dir / f"{fixture_key}.json"

        # Replay mode: load from fixture
        if self.replay_mode:
            if fixture_path.exists():
                with open(fixture_path) as f:
                    data = json.load(f)
                logger.debug(f"Loaded fixture: {fixture_key}")
                return data.get("response", "{}")
            else:
                logger.warning(f"Fixture not found: {fixture_key}. Using fallback.")
                return self._fallback_response(agent_type, incident_type)

        # Record mode: call API and save
        response = self._call_api(prompt)
        fixture_data = {
            "model": self.model_name,
            "agent_type": agent_type,
            "incident_type": incident_type,
            "prompt_hash": hash(prompt),
            "response": response,
        }
        with open(fixture_path, "w") as f:
            json.dump(fixture_data, f, indent=2)
        logger.debug(f"Saved fixture: {fixture_key}")
        return response

    def _call_api(self, prompt: str) -> str:
        """Call actual Claude API.

        Placeholder: will implement when API key available.
        """
        logger.warning("LLM API call not implemented (no API key). Using fallback.")
        return self._fallback_response("agent", "incident")

    def _fallback_response(self, agent_type: str, incident_type: str) -> str:
        """Return heuristic fallback response."""
        # Default confidence/savings/risk for any agent
        return json.dumps(
            {
                "confidence": 0.7,
                "estimated_business_savings": 2000.0,
                "estimated_risk": 0.15,
                "estimated_compute_cost": 10.0,
                "estimated_time": 30.0,
                "rationale": f"Heuristic proposal from {agent_type} for {incident_type}",
            }
        )
