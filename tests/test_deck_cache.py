import tempfile
import unittest
from pathlib import Path

from ai_prof.deck_cache import DeckCache
from ai_prof.pdf_utils import Deck, Slide


class DeckCacheTests(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.png"
            source.write_bytes(b"png")
            cache = DeckCache(root=str(Path(tmp) / "cache"))
            deck = Deck(
                slides=[
                    Slide(index=0, image_path=str(source), text="First slide"),
                ]
            )

            cache.save(
                "deck-key",
                deck=deck,
                readings={0: "TITLE: First\nCONCEPTS: caching"},
                deck_index="1. First - caching",
                metadata={"title": "Caching 101", "dpi": 150},
            )
            loaded = cache.load("deck-key")

            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.deck.slides[0].text, "First slide")
            self.assertEqual(
                loaded.readings[0],
                "TITLE: First\nCONCEPTS: caching",
            )
            self.assertEqual(loaded.deck_index, "1. First - caching")
            self.assertTrue(Path(loaded.deck.slides[0].image_path).is_file())
            self.assertEqual(
                cache.list_decks()[0].title,
                "Caching 101",
            )

    def test_key_changes_with_processing_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "lecture.pdf"
            pdf.write_bytes(b"same lecture")
            cache = DeckCache(root=str(Path(tmp) / "cache"))

            first = cache.key(str(pdf), dpi=150, vision_model="minicpm-v")
            second = cache.key(str(pdf), dpi=200, vision_model="minicpm-v")
            third = cache.key(str(pdf), dpi=150, vision_model="other-model")

            self.assertNotEqual(first, second)
            self.assertNotEqual(first, third)


if __name__ == "__main__":
    unittest.main()
