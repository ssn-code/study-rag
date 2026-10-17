import os
import subprocess
import sys
import unittest


class ConfigTests(unittest.TestCase):
    def test_invalid_llm_settings_report_actionable_errors(self) -> None:
        cases = (
            ("LLM_TEMPERATURE", "oops", "LLM_TEMPERATURE must be a number"),
            ("LLM_TEMPERATURE", "3", "LLM_TEMPERATURE must be between 0 and 2"),
            ("LLM_TOP_P", "nan", "LLM_TOP_P must be a finite number"),
            ("LLM_MAX_TOKENS", "0", "LLM_MAX_TOKENS must be at least 1"),
            (
                "LLM_TIMEOUT_SECONDS",
                "-1",
                "LLM_TIMEOUT_SECONDS must be greater than 0",
            ),
        )
        for name, value, message in cases:
            with self.subTest(setting=name, value=value):
                environment = os.environ.copy()
                environment[name] = value
                result = subprocess.run(
                    [sys.executable, "-c", "import src.config"],
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()