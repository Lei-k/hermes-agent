from pathlib import Path
import re
import unittest


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "install-e2e.yml"


class InstallE2EForkGuardTests(unittest.TestCase):
    def test_release_picker_only_runs_in_canonical_release_repository(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        pick_job = re.search(
            r"(?ms)^  pick-releases:\n(?P<body>.*?)(?=^  update:\n)",
            workflow,
        )

        self.assertIsNotNone(pick_job, "pick-releases job must remain present")
        assert pick_job is not None
        self.assertRegex(
            pick_job.group("body"),
            r"(?m)^    if: github\.repository == 'NousResearch/hermes-agent'$",
        )


if __name__ == "__main__":
    unittest.main()
