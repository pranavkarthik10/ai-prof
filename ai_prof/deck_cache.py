"""Content-addressed cache for rendered and vision-indexed lecture decks."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .pdf_utils import Deck, Slide

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CachedDeck:
    deck: Deck
    readings: dict[int, str]
    deck_index: str


@dataclass(frozen=True)
class DeckSummary:
    key: str
    title: str
    slide_count: int


def pdf_sha256(pdf_path: str) -> str:
    digest = hashlib.sha256()
    with open(pdf_path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DeckCache:
    def __init__(
        self,
        *,
        root: str,
        repo_id: str | None = None,
        token: str | None = None,
        write_remote: bool = False,
    ) -> None:
        self.root = Path(root).expanduser()
        self.repo_id = repo_id
        self.token = token
        self.write_remote = write_remote

    def key(self, pdf_path: str, *, dpi: int, vision_model: str) -> str:
        identity = f"{pdf_sha256(pdf_path)}:{dpi}:{vision_model}:v{SCHEMA_VERSION}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def load(self, key: str) -> CachedDeck | None:
        cached = self._load_local(key)
        if cached is not None or not self.repo_id:
            return cached
        if self._download_remote(key):
            return self._load_local(key)
        return None

    def list_decks(self) -> list[DeckSummary]:
        manifests: dict[str, Path] = {}
        if self.root.is_dir():
            for path in self.root.glob("*/manifest.json"):
                manifests[path.parent.name] = path

        if self.repo_id:
            try:
                from huggingface_hub import HfApi, hf_hub_download

                for filename in HfApi(token=self.token).list_repo_files(
                    self.repo_id,
                    repo_type="dataset",
                ):
                    parts = Path(filename).parts
                    if (
                        len(parts) == 3
                        and parts[0] == "decks"
                        and parts[2] == "manifest.json"
                    ):
                        key = parts[1]
                        if key not in manifests:
                            manifests[key] = Path(
                                hf_hub_download(
                                    repo_id=self.repo_id,
                                    repo_type="dataset",
                                    filename=filename,
                                    token=self.token,
                                )
                            )
            except Exception as exc:
                print(f"[deck-cache] Hub catalog skipped: {exc}")

        summaries = []
        for key, path in manifests.items():
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                metadata = manifest.get("metadata") or {}
                slides = manifest.get("slides") or []
                title = str(metadata.get("title") or f"Prepared lecture {key[:8]}")
                summaries.append(
                    DeckSummary(key=key, title=title, slide_count=len(slides))
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return sorted(summaries, key=lambda item: item.title.lower())

    def save(
        self,
        key: str,
        *,
        deck: Deck,
        readings: dict[int, str],
        deck_index: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        target = self.root / key
        slides_dir = target / "slides"
        if target.exists():
            shutil.rmtree(target)
        slides_dir.mkdir(parents=True, exist_ok=True)

        slides = []
        for slide in deck.slides:
            filename = f"{slide.index:03d}.png"
            shutil.copy2(slide.image_path, slides_dir / filename)
            slides.append(
                {
                    "index": slide.index,
                    "image": f"slides/{filename}",
                    "text": slide.text,
                    "reading": readings.get(slide.index, ""),
                }
            )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "deck_index": deck_index,
            "metadata": metadata or {},
            "slides": slides,
        }
        (target / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        if self.repo_id and self.write_remote:
            self._upload_remote(key, target)

    def _load_local(self, key: str) -> CachedDeck | None:
        target = self.root / key
        manifest_path = target / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != SCHEMA_VERSION:
                return None
            slides = []
            readings = {}
            for raw in manifest["slides"]:
                image_path = target / raw["image"]
                if not image_path.is_file():
                    return None
                index = int(raw["index"])
                slides.append(
                    Slide(
                        index=index,
                        image_path=str(image_path),
                        text=str(raw.get("text", "")),
                    )
                )
                readings[index] = str(raw.get("reading", ""))
            return CachedDeck(
                deck=Deck(slides=slides),
                readings=readings,
                deck_index=str(manifest.get("deck_index", "")),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _download_remote(self, key: str) -> bool:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token,
                allow_patterns=[f"decks/{key}/**"],
                local_dir=self.root,
            )
            remote_dir = self.root / "decks" / key
            local_dir = self.root / key
            if remote_dir.is_dir():
                if local_dir.exists():
                    shutil.rmtree(local_dir)
                shutil.move(str(remote_dir), str(local_dir))
            return (local_dir / "manifest.json").is_file()
        except Exception as exc:
            print(f"[deck-cache] Hub download skipped: {exc}")
            return False

    def _upload_remote(self, key: str, source: Path) -> None:
        try:
            from huggingface_hub import HfApi

            HfApi(token=self.token).upload_folder(
                repo_id=self.repo_id,
                repo_type="dataset",
                folder_path=str(source),
                path_in_repo=f"decks/{key}",
                commit_message=f"Cache processed AI Prof deck {key[:12]}",
            )
        except Exception as exc:
            print(f"[deck-cache] Hub upload skipped: {exc}")
