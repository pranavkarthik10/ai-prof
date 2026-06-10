import unittest

from ai_prof.agent import _extract_json, _validate_actions


class AgentValidationTests(unittest.TestCase):
    def test_extract_json_accepts_fenced_output(self):
        self.assertEqual(
            _extract_json('```json\n{"narration": "Hello"}\n```'),
            {"narration": "Hello"},
        )

    def test_validate_actions_bounds_navigation_and_whiteboard(self):
        actions = _validate_actions(
            [
                {"tool": "goto_slide", "args": {"index": 3}},
                {"tool": "next_slide", "args": {}},
                {"tool": "goto_slide", "args": {"index": 99}},
                {
                    "tool": "write_note",
                    "args": {"title": "Kernel", "body": "Weights"},
                },
                {"tool": "write_latex", "args": {"expression": "a+b"}},
                {"tool": "clear_whiteboard", "args": {}},
                {"tool": "unknown", "args": {}},
            ],
            total_slides=8,
        )

        self.assertEqual(
            [action.tool for action in actions],
            ["goto_slide", "write_note", "write_latex"],
        )
        self.assertEqual(actions[0].args, {"index": 3})

    def test_validate_actions_rejects_empty_whiteboard_content(self):
        actions = _validate_actions(
            [
                {"tool": "write_note", "args": {"title": "", "body": ""}},
                {"tool": "write_latex", "args": {"expression": ""}},
            ],
            total_slides=2,
        )
        self.assertEqual(actions, ())


if __name__ == "__main__":
    unittest.main()
