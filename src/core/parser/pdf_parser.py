from pathlib import Path
import asyncio
from typing import Union, Literal, Optional

from llama_cloud import AsyncLlamaCloud
from llama_cloud import omit
from llama_cloud.resources.parsing import ParsingGetResponse
from src.utils.logger import setup_logger

PathLike = Union[str, Path]
Tier = Literal["fast", "cost_effective", "agentic", "agentic_plus"]
ExpandType = Literal["text", "markdown", "items"]

logger = setup_logger("parser")


class PDFParser:
    """
    Async PDF parser backed by LlamaCloud.

    Responsibilities:
      - upload file
      - parse with optimal settings per tier
      - persist results to disk

    Not responsible for chunking, indexing, or RAG pipeline logic.
    """

    _AGENTIC_TIERS = frozenset({"agentic", "agentic_plus"})
    _CUSTOM_PROMPT = """
    Parse this document following these rules:
    1. For merged cells (rowspan/colspan), repeat the cell value in each row - do not leave empty cells.
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
    # API
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
            file_path:    Path to the PDF file.
            page_ranges:  LlamaCloud page range string, e.g. "1-5,8".
            expand_type:  Response field to expand: "text" | "markdown" | "items".
            tier:         Parsing tier. Use "agentic" for complex tables.
            languages:    OCR language hints, e.g. ["rus", "eng"].
                          Defaults to ["rus", "eng"] when None.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        languages = languages or ["ru", "en"]
        file_obj = await self._upload_file(path)
        logger.info("Uploaded '%s' → file_id=%s", path.name, file_obj.id)

        logger.info("Parsing pages='%s' tier='%s'", page_ranges, tier)
        logger.info("Processing options='%s'",self._build_processing_options(languages))
        logger.info("Output options='%s'",self._build_output_options())
        response = await self.client.parsing.parse(
            file_id=file_obj.id,
            tier=tier,
            version="latest",
            expand=[expand_type],
            page_ranges={"target_pages": page_ranges},
            processing_options=self._build_processing_options(languages=languages),
            output_options=self._build_output_options(),
            agentic_options=(
                {"custom_prompt": self._CUSTOM_PROMPT}
                if tier in self._AGENTIC_TIERS
                else omit
            )
        )
        logger.info("Parsing complete for '%s'", path.name)
        return response   


    async def save_markdown_full(self, response: ParsingGetResponse, output_path: PathLike) -> None:
        """Persist markdown_full from a parse response to disk."""
        content = response.markdown_full
        if not content:
            logger.warning("Response contains no markdown_full content - skipping save")
            return
        await self._write_file(response.markdown_full, output_path)

    async def save_text(self, text: str, output_path: PathLike) -> None:
        """Persist arbitrary text to disk."""
        await self._write_file(text, output_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _upload_file(self, file_path: Path):
        """Upload file to LlamaCloud and return the file object."""
        with file_path.open("rb") as f:
            return await self.client.files.create(file=f, purpose="parse")
    
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
        target = Path(output_path)

        def _sync_write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        try:
            await asyncio.to_thread(_sync_write)
            logger.info("Saved → %s", target)
        except OSError as e:
            logger.error("IO error writing to %s: %s", target, e)
            raise