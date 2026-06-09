"""Turn an uploaded lecture PDF into per-slide images + text-layer ground truth.

The rendered image serves double duty: it's what the user sees AND what the
vision model reads. The extracted text layer is exact (vision misreads small
text), so it's kept alongside as a complement to the vision reading.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class Slide:
    index: int  # 0-based
    image_path: str  # rendered PNG on disk
    text: str  # PDF text-layer extraction (may be empty for image-only decks)


@dataclass
class Deck:
    slides: list[Slide] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.slides)

    def outline(self) -> str:
        """A compact slide -> first-line index, so the brain can plan/navigate."""
        lines = []
        for s in self.slides:
            head = next((ln.strip() for ln in s.text.splitlines() if ln.strip()), "")
            head = head[:80] or "(no text layer)"
            lines.append(f"  {s.index + 1}. {head}")
        return "\n".join(lines)


def render_pdf(pdf_path: str, dpi: int = 150) -> Deck:
    """Render every page to a PNG and pull its text layer."""
    out_dir = Path(tempfile.mkdtemp(prefix="ai_prof_slides_"))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    deck = Deck()
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix)
            img_path = out_dir / f"slide_{i:03d}.png"
            pix.save(img_path)
            deck.slides.append(
                Slide(index=i, image_path=str(img_path), text=page.get_text().strip())
            )
    return deck
