"""MediaWiki reader for LlamaIndex.

Provides a LlamaIndex-compatible reader that fetches and converts pages from
any MediaWiki instance into LlamaIndex Documents.
"""

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

import html2text
import requests

from llama_index.core.bridge.pydantic import Field
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

_internal_logger = logging.getLogger(__name__)


class MediaWikiReader(BasePydanticReader):
    """LlamaIndex reader for MediaWiki instances.

    Fetches pages from a MediaWiki API endpoint, converts HTML content to clean
    text, and returns LlamaIndex Documents with metadata (title, URL,
    last_modified).

    Implements BasePydanticReader (for serialization / LlamaHub compatibility)
    and provides get_resource_info and load_resource for resource-based use.
    Additionally exposes get_resources_info for efficient batched timestamp/URL
    retrieval without N+1 API calls.
    """

    model_config = {"arbitrary_types_allowed": True}

    # -- Pydantic fields (serialisable config) --------------------------------

    api_url: str = Field(description="MediaWiki API endpoint URL")
    user_agent: str = Field(
        default="llama-index-readers-mediawiki/1.0",
        description="User-Agent header for HTTP requests",
    )
    request_delay: float = Field(
        default=0.1,
        description="Delay in seconds between API requests (rate limiting)",
    )
    page_limit: int = Field(
        default=500,
        description="Maximum number of pages per allpages API call",
    )
    batch_size: int = Field(
        default=50,
        description="Number of pages to batch for timestamp fetching",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts for API requests",
    )
    timeout: int = Field(
        default=30,
        description="HTTP request timeout in seconds",
    )
    namespaces: Optional[List[int]] = Field(
        default=None,
        description="List of namespace IDs to include (None = all namespaces)",
    )
    logger: logging.Logger = Field(
        default_factory=lambda: _internal_logger,
        description="Logger instance (injectable for tests or custom logging)",
        exclude=True,
    )

    # -- Non-serialised internal state ----------------------------------------
    _session: Optional[requests.Session] = None

    # -- Construction helpers -------------------------------------------------

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._validate_config()
        self.logger.info("Initialized MediaWikiReader for %s", self.api_url)

    def _validate_config(self) -> None:
        """Validate numeric config bounds — mirrors the original job checks."""
        if not self.api_url:
            raise ValueError("api_url is required")
        if self.request_delay < 0:
            raise ValueError("request_delay must be non-negative")
        if self.page_limit <= 0:
            raise ValueError("page_limit must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    # -- Session lifecycle ----------------------------------------------------

    @property
    def session(self) -> requests.Session:
        """Return the HTTP session, creating one if needed."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"User-Agent": self.user_agent})
        return self._session


    # -- Internal API helpers -------------------------------------------------

    def _make_api_request(
        self,
        params: Dict[str, Any],
        max_retries: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make a request to the MediaWiki API with retries and error handling.

        Args:
            params: Dictionary of API parameters (``format=json`` is added
                automatically).
            max_retries: Optional override for the number of retry attempts.

        Returns:
            Parsed JSON response, or ``None`` if the request failed.
        """
        params["format"] = "json"

        # Determine total attempts (at least 1 if max_retries=0)
        retries = max_retries if max_retries is not None else self.max_retries
        max_attempts = retries + 1

        for attempt in range(max_attempts):
            try:
                response = self.session.get(
                    self.api_url, params=params, timeout=self.timeout
                )

                if response.status_code == 429:
                    if attempt < max_attempts - 1:
                        retry_after = int(response.headers.get("Retry-After", 5))
                        self.logger.warning("Rate limited. Waiting %d seconds...", retry_after)
                        time.sleep(retry_after)
                        continue
                    else:
                        self.logger.error("Rate limited and no more retries left.")
                        return None

                response.raise_for_status()
                return response.json()

            except (requests.exceptions.RequestException, ValueError) as exc:
                if attempt < max_attempts - 1:
                    self.logger.warning(
                        "API request failed (attempt %d/%d): %s",
                        attempt + 1,
                        max_attempts,
                        exc,
                    )
                    time.sleep(2**attempt)
                    continue
                else:
                    self.logger.error("API request failed after %d attempts: %s", max_attempts, exc)
                    return None

        return None

    def _get_all_pages_generator(self) -> Iterator[Dict[str, Any]]:
        """Yield rich dictionaries for all pages using the generator API.

        Each yielded dict contains:
            - title (str)
            - url (str or None)
            - last_modified (datetime or None)

        This is much more efficient than fetching titles first and then
        querying metadata for each title.
        """
        # If namespaces is None, query all. If it's a list, we must iterate
        # because gapnamespace only supports a single value.
        namespaces = self.namespaces if self.namespaces is not None else [None]

        for ns in namespaces:
            continue_params: Dict[str, Any] = {}
            while True:
                params: Dict[str, Any] = {
                    "action": "query",
                    "generator": "allpages",
                    "gaplimit": self.page_limit,
                    "prop": "info|revisions",
                    "inprop": "url",
                    "rvprop": "timestamp",
                    "format": "json",
                    **continue_params,
                }
                if ns is not None:
                    params["gapnamespace"] = ns

                data = self._make_api_request(params)
                if not data:
                    break

                pages_dict = data.get("query", {}).get("pages", {})
                # generator=allpages returns a dict keyed by page ID
                for page_data in pages_dict.values():
                    title = page_data.get("title")
                    if not title:
                        continue

                    url = page_data.get("canonicalurl")

                    last_modified = None
                    revisions = page_data.get("revisions", [])
                    if revisions:
                        ts_str = revisions[0].get("timestamp")
                        if ts_str:
                            try:
                                # MediaWiki uses ISO8601 with Z
                                last_modified = datetime.fromisoformat(
                                    ts_str.replace("Z", "+00:00")
                                )
                            except (ValueError, TypeError):
                                pass

                    yield {
                        "title": title,
                        "url": url,
                        "last_modified": last_modified,
                    }

                continue_info = data.get("continue")
                if continue_info:
                    continue_params = continue_info
                    time.sleep(self.request_delay)
                else:
                    break


    def _get_page_contents(self, page_title: str) -> Optional[str]:
        """Fetch parsed content for a page.

        Returns:
            Clean text content or ``None``.
        """
        params = {
            "action": "parse",
            "page": page_title,
            "prop": "text",
            "disableeditsection": "true",
            "disabletoc": "true",
            "disablelimitreport": "true",
            "format": "json",
        }

        parsed_data = self._make_api_request(params)
        if not parsed_data:
            return None

        parse_result = parsed_data.get("parse", {})
        if not parse_result:
            self.logger.warning("No parse result for page '%s'", page_title)
            return None

        html_content = parse_result.get("text", {}).get("*", "")
        if not html_content:
            self.logger.warning("No content in parse result for page '%s'", page_title)
            return None

        return self._html_to_clean_text(html_content)

    def _html_to_clean_text(self, html_content: str) -> str:
        """Convert MediaWiki HTML to clean Markdown text."""
        try:
            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            h.body_width = 0
            h.ul_item_mark = "-"
            h.emphasis_mark = "*"
            h.strong_mark = "**"
            return h.handle(html_content).strip()
        except Exception as exc:
            self.logger.error("html2text conversion failed: %s", exc)
            clean_text = re.sub(r"<[^>]+>", "", html_content)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
            return clean_text

    # -- Resource API (get_resource_info, load_resource) ----------------------

    def get_resource_info(
        self, resource_id: str, *args: Any, **kwargs: Any
    ) -> Dict:
        """Return info for a single page.

        Returns:
            ``{"last_modified": datetime | None, "url": str | None}``
        """
        info = self.get_resources_info([resource_id])
        return info.get(resource_id, {"last_modified": None, "url": None})

    def load_resource(
        self,
        resource_id: str,
        resource_url: Optional[str] = None,
        last_modified: Optional[datetime] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Load a single page as a list containing one Document.

        Consolidated to minimize API calls. If resource_url and last_modified are
        provided, we only perform the 'parse' API call.

        Args:
            resource_id: The page title.
            resource_url: Optional pre-fetched canonical URL.
            last_modified: Optional pre-fetched last-modified timestamp.

        Returns:
            A one-element list with the page Document, or an empty list on failure.
        """
        if resource_url and last_modified:
            # We already have metadata, just need the content
            content = self._get_page_contents(resource_id)
            if not content:
                return []
        else:
            # Fallback: Fetch missing metadata first
            info_map = self.get_resources_info([resource_id])
            info = info_map.get(resource_id)

            if not info or not info.get("url"):
                self.logger.warning("Metadata not found for fallback page '%s'", resource_id)
                return []

            resource_url = info["url"]
            last_modified = info["last_modified"]

            content = self._get_page_contents(resource_id)
            if not content:
                return []

        # Build Document
        doc = Document(
            text=content,
            id_=f"mediawiki:{resource_id}",
            metadata={
                "title": resource_id,
                "url": resource_url,
                "last_modified": last_modified.isoformat() if last_modified else None,
            },
            excluded_llm_metadata_keys=["url", "last_modified"],
            excluded_embed_metadata_keys=["url", "last_modified"],
        )
        return [doc]

    # -- Custom batched method (NOT part of LlamaIndex API) -------------------

    def get_resources_info(
        self, page_titles: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Return info for multiple pages in batched API calls.

        Custom extension to retrieve ``last_modified`` and ``url`` for many pages
        in batched API requests (avoids N+1 round-trips).

        Args:
            page_titles: List of page titles.

        Returns:
            Dict mapping each title to
            ``{"last_modified": datetime | None, "url": str | None}``.
        """
        if not page_titles:
            return {}

        result: Dict[str, Dict[str, Any]] = {}
        # Batch-fetch both URLs and timestamps in one API request (prop=info|revisions)
        for i in range(0, len(page_titles), self.batch_size):
            batch = page_titles[i : i + self.batch_size]
            titles_param = "|".join(batch)
            params = {
                "action": "query",
                "titles": titles_param,
                "prop": "info|revisions",
                "inprop": "url",
                "rvprop": "timestamp",
            }
            data = self._make_api_request(params)
            if not data:
                for title in batch:
                    result[title] = {"last_modified": None, "url": None}
                continue

            pages = data.get("query", {}).get("pages", {})
            title_to_page = {
                pg.get("title"): pg for pg in pages.values() if pg.get("title")
            }

            for title in batch:
                page_data = title_to_page.get(title)
                last_modified = None
                url = None
                if page_data and "missing" not in page_data:
                    url = page_data.get("canonicalurl")
                    revisions = page_data.get("revisions", [])
                    if revisions:
                        ts_str = revisions[0].get("timestamp")
                        if ts_str:
                            try:
                                last_modified = datetime.fromisoformat(
                                    ts_str.replace("Z", "+00:00")
                                )
                            except (ValueError, TypeError):
                                pass

                result[title] = {
                    "last_modified": last_modified,
                    "url": url,
                }
        return result

    # -- BasePydanticReader / BaseReader interface -----------------------------

    def lazy_load_data(self, *args: Any, **kwargs: Any) -> Iterator[Document]:
        """Yield one Document per page in the wiki.

        Optimized to fetch content while traversing pages to avoid N+1 queries.
        Note: Content (the 'parse' action) still requires a separate call per page
        as the 'text' property is too large for the query generator.
        """
        for page_record in self._get_all_pages_generator():
            title = page_record["title"]
            url = page_record.get("url")
            last_modified = page_record.get("last_modified")

            docs = self.load_resource(
                title, resource_url=url, last_modified=last_modified
            )
            yield from docs
            time.sleep(self.request_delay)
