import unittest
from unittest.mock import patch

from ai_prof.agent import AgentAction, TeachingBeat
from ai_prof.pdf_utils import Deck, Slide
import app


def _state() -> dict:
    state = app._new_state()
    state["deck"] = Deck(
        slides=[
            Slide(index=0, image_path="/tmp/slide-1.png", text="One"),
            Slide(index=1, image_path="/tmp/slide-2.png", text="Two"),
            Slide(index=2, image_path="/tmp/slide-3.png", text="Three"),
        ]
    )
    state["readings"] = {
        0: "TITLE: One\nCONCEPTS: first",
        1: "TITLE: Two\nCONCEPTS: second",
        2: "TITLE: Three\nCONCEPTS: third",
    }
    state["deck_index"] = "1. One\n2. Two\n3. Three"
    return state


class OrchestratorTests(unittest.TestCase):
    def test_execute_actions_navigates_and_updates_whiteboard(self):
        state = _state()
        navigated = app._execute_actions(
            state,
            (
                AgentAction("goto_slide", {"index": 3}),
                AgentAction(
                    "write_note",
                    {"title": "Remember", "body": "Use the third slide."},
                ),
            ),
        )

        self.assertTrue(navigated)
        self.assertEqual(state["index"], 2)
        self.assertEqual(state["whiteboard"][0]["title"], "Remember")

    @patch.object(app, "tts_speak_full", return_value=None)
    @patch.object(app, "plan_teaching_beat")
    def test_question_can_move_to_supporting_slide(self, plan, _tts):
        plan.return_value = TeachingBeat(
            narration="Let’s revisit the definition.",
            actions=(AgentAction("goto_slide", {"index": 2}),),
            continue_lecture=False,
        )
        outputs = list(app.on_ask("What was the definition?", _state(), []))
        final = outputs[-1]

        self.assertEqual(final[0]["index"], 1)
        self.assertEqual(final[2], "Slide 2 / 3")
        self.assertEqual(final[3][-1]["content"], "Let’s revisit the definition.")

    @patch.object(app, "tts_speak_full", return_value=None)
    @patch.object(app, "plan_teaching_beat")
    def test_lecture_advances_when_agent_does_not_navigate(self, plan, _tts):
        plan.side_effect = [
            TeachingBeat("First beat.", continue_lecture=True),
            TeachingBeat("Second beat.", continue_lecture=False),
        ]
        outputs = list(app.on_teach_deck(_state(), []))
        final = outputs[-1]

        self.assertEqual(final[0]["index"], 1)
        self.assertEqual(
            [message["content"] for message in final[3]],
            ["First beat.", "Second beat."],
        )


if __name__ == "__main__":
    unittest.main()
