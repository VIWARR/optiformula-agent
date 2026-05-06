from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from llama_cloud import AsyncLlamaCloud, omit
from llama_cloud.resources.parsing import ParsingGetResponse

from src.core.schema import ExpandType, PathLike, Tier
from src.utils.logger import setup_logger

logger = setup_logger("parser")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class PDFParser:
    """
    Async PDF parser backed by LlamaCloud.

    Responsibilities:
    - Upload a PDF file to LlamaCloud.
    - Trigger parsing with per-tier options (OCR, table extraction, prompts).
    - Persist the resulting Markdown or plain text to disk.

    Out of scope: chunking, embedding, indexing, and RAG pipeline logic.

    Args:
        api_key: LlamaCloud API key.  Must be non-empty.

    Example::

        parser = PDFParser(api_key="llx-...")
        response = await parser.parse_range("guide.pdf", page_ranges="1-10")
        await parser.save_markdown_full(response, "output/guide.md")
    """

    _AGENTIC_TIERS = frozenset({"agentic", "agentic_plus"})

    # Injected into agentic-tier jobs to improve table fidelity.
    _CUSTOM_PROMPT = """
    Parse this document following these rules:
    1. For merged cells (rowspan/colspan), repeat the cell value in each row — do not leave empty cells.
    2. Remove completely empty rows from tables.
    3. Preserve strikethrough formatting as ~~text~~.
    4. Preserve heading hierarchy (H1, H2, H3).
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("LlamaCloud API key is required")
        self.client = AsyncLlamaCloud(api_key=api_key)
        logger.info("Parser initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def parse_range(
            self, 
            file_path: PathLike, 
            page_ranges: str, 
            expand_type: ExpandType = "markdown",
            tier: Tier = "agentic",
            languages: Optional[list[str]] = None,
    ) -> ParsingGetResponse:
        """
        Upload and parse a PDF, returning the raw LlamaCloud response.

        Args:
            file_path:    Path to the local PDF file.
            page_ranges:  LlamaCloud range string, e.g. ``"1-5,8"``.
            expand_type:  Response field to materialise: ``"text"``,
                          ``"markdown"``, or ``"items"``.
            tier:         Parsing tier.  Use ``"agentic"`` for complex tables.
            languages:    OCR language hints.  Defaults to ``["ru", "en"]``.

        Raises:
            FileNotFoundError: When *file_path* does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        languages = languages or ["ru", "en"]
        file_obj = await self._upload_file(path)
        logger.info("Uploaded '%s' → file_id=%s", path.name, file_obj.id)

        processing_opts = self._build_processing_options(languages)
        output_opts = self._build_output_options()
        logger.info(
            "Parsing pages='%s' tier='%s' processing=%s",
            page_ranges,
            tier,
            processing_opts,
        )

        response = await self.client.parsing.parse(
            file_id=file_obj.id,
            tier=tier,
            version="latest",
            expand=[expand_type],
            page_ranges={"target_pages": page_ranges},
            processing_options=processing_opts,
            output_options=output_opts,
            agentic_options=(
                {"custom_prompt": self._CUSTOM_PROMPT}
                if tier in self._AGENTIC_TIERS
                else omit
            ),
        )
        logger.info("Parsing complete for '%s'", path.name)
        return response   

    async def save_markdown_full(self, response: ParsingGetResponse, output_path: PathLike) -> None:
        """Write ``response.markdown_full`` to *output_path*."""
        content = response.markdown_full
        if not content:
            logger.warning("Response contains no markdown_full content - skipping save")
            return
        await self._write_file(response.markdown_full, output_path)

    async def save_text(self, text: str, output_path: PathLike) -> None:
        """Write arbitrary *text* to *output_path*."""
        await self._write_file(text, output_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _upload_file(self, file_path: Path):
        """Upload *file_path* to LlamaCloud and return the file object."""
        with file_path.open("rb") as fh:
            return await self.client.files.create(file=fh, purpose="parse")

    @staticmethod
    def _build_processing_options(languages: list[str]) -> dict:
        return {
            "aggressive_table_extraction": True,
            "cost_optimizer": {"enable": True},
            "ignore": {
                "ignore_diagonal_text": True,
                "ignore_hidden_text": True,
            },
            "ocr_parameters": {
                "languages": languages,
            },
        }

    @staticmethod
    def _build_output_options() -> dict:
        return {
            "markdown": {
                "tables": {
                    "output_tables_as_markdown": True,
                    "merge_continued_tables": True,
                    "compact_markdown_tables": False,
                    "markdown_table_multiline_separator": "<br>",
                },
            },
        }

    @staticmethod
    async def _write_file(content: str, output_path: PathLike) -> None:
        """Async wrapper around a blocking file write."""
        target = Path(output_path)

        def _sync_write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        try:
            await asyncio.to_thread(_sync_write)
            logger.info("Saved → %s", target)
        except OSError as exc:
            logger.error("IO error writing to %s: %s", target, exc)
            raise