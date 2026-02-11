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
from llama_index.core.readers.base import BasePydanticReader, ResourcesReaderMixin
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


class MediaWikiReader(BasePydanticReader, ResourcesReaderMixin):
    """LlamaIndex reader for MediaWiki instances.

    Fetches pages from a MediaWiki API endpoint, converts HTML content to clean
    text, and returns LlamaIndex Documents with metadata (title, URL,
    last_modified).

    Implements both BasePydanticReader (for serialization / LlamaHub
    compatibility) and ResourcesReaderMixin (for the standard resource-based
    interface: list_resources, get_resource_info, load_resource).

    Additionally exposes a custom ``get_resources_info`` method for efficient
    batched timestamp/URL retrieval — this is *not* part of the LlamaIndex API
    but is used by downstream jobs to avoid N+1 API calls.
    """

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
    is_remote: bool = Field(default=True, description="Data is loaded from a remote API")

    # -- Non-serialised internal state ----------------------------------------
    _session: Optional[requests.Session] = None

    # -- Construction helpers -------------------------------------------------

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._validate_config()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.user_agent})
        logger.info("Initialized MediaWikiReader for %s", self.api_url)

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

    def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "MediaWikiReader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

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
            max_retries: Override for the configured max retry count.

        Returns:
            Parsed JSON response, or ``None`` if all retries failed.
        """
        if max_retries is None:
            max_retries = self.max_retries

        if max_retries <= 0:
            return None

        for attempt in range(max_retries):
            try:
                params["format"] = "json"
                response = self.session.get(
                    self.api_url, params=params, timeout=self.timeout
                )

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    logger.warning("Rate limited. Waiting %d seconds...", retry_after)
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as exc:
                logger.warning(
                    "API request failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries,
                    exc,
                )
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    logger.error("API request failed after %d attempts", max_retries)
                    return None

            except ValueError as exc:
                logger.error("Invalid JSON response: %s", exc)
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

    def _get_all_pages(self) -> Iterator[Dict[str, Any]]:
        """Deprecated: Use _get_all_pages_generator for efficient metadata.

        Maintained for backward compatibility with existing internal callers.
        """
        for page in self._get_all_pages_generator():
            yield {"title": page["title"]}

    def _get_page_data(
        self, page_title: str, **api_params: Any
    ) -> Optional[Dict[str, Any]]:
        """Query a single MediaWiki page and return its data dict.

        Returns ``None`` if the page is missing or the request failed.
        """
        params = {"action": "query", "titles": page_title, **api_params}

        data = self._make_api_request(params)
        if not data:
            return None

        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None

        page_data = next(iter(pages.values()))

        if page_data.get("pageid") == -1 or page_data.get("missing") is True:
            logger.warning("Page '%s' is missing", page_title)
            return None

        return page_data

    def _get_page_info(self, page_title: str) -> Optional[tuple]:
        """Fetch parsed content and canonical URL for a page.

        Returns:
            ``(clean_text, canonical_url)`` or ``None``.
        """
        url_data = self._get_page_data(page_title, prop="info", inprop="url")
        if not url_data:
            return None

        canonical_url = url_data.get("canonicalurl")
        if not canonical_url:
            logger.warning("No URL found for page '%s'", page_title)
            return None

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
            logger.warning("No parse result for page '%s'", page_title)
            return None

        html_content = parse_result.get("text", {}).get("*", "")
        if not html_content:
            logger.warning("No content in parse result for page '%s'", page_title)
            return None

        clean_content = self._html_to_clean_text(html_content)
        return clean_content, canonical_url

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
            logger.error("html2text conversion failed: %s", exc)
            clean_text = re.sub(r"<[^>]+>", "", html_content)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
            return clean_text

    def _get_pages_last_modified(
        self, page_titles: List[str]
    ) -> Dict[str, Optional[datetime]]:
        """Get last-modified timestamps for multiple pages in a single API call."""
        if not page_titles:
            return {}

        titles_param = "|".join(page_titles)
        params = {
            "action": "query",
            "titles": titles_param,
            "prop": "revisions",
            "rvprop": "timestamp",
        }

        data = self._make_api_request(params)
        if not data:
            return {title: None for title in page_titles}

        pages = data.get("query", {}).get("pages", {})
        title_to_page = {
            page_info.get("title"): page_info for page_info in pages.values()
        }

        result: Dict[str, Optional[datetime]] = {}
        for title in page_titles:
            page_data = title_to_page.get(title)
            if not page_data or "pageid" not in page_data:
                result[title] = None
                continue

            revisions = page_data.get("revisions", [])
            if revisions:
                timestamp_str = revisions[0].get("timestamp")
                if timestamp_str:
                    try:
                        result[title] = datetime.fromisoformat(timestamp_str)
                    except ValueError as exc:
                        logger.warning(
                            "Failed to parse timestamp '%s' for page '%s': %s",
                            timestamp_str,
                            title,
                            exc,
                        )
                        result[title] = None
                else:
                    result[title] = None
            else:
                result[title] = None

        return result

    def _get_page_url(self, page_title: str) -> Optional[str]:
        """Return the canonical URL for a page, or ``None``."""
        url_data = self._get_page_data(page_title, prop="info", inprop="url")
        if not url_data:
            return None
        return url_data.get("canonicalurl")

    # -- ResourcesReaderMixin implementation ----------------------------------

    def list_resources(self, *args: Any, **kwargs: Any) -> List[str]:
        """Return a list of all page titles in the wiki."""
        return [page["title"] for page in self._get_all_pages() if "title" in page]

    def get_resource_info(
        self, resource_id: str, *args: Any, **kwargs: Any
    ) -> Dict:
        """Return info for a single page (required by ResourcesReaderMixin).

        Returns:
            ``{"last_modified": datetime | None, "url": str | None}``
        """
        info = self.get_resources_info([resource_id])
        return info.get(resource_id, {"last_modified": None, "url": None})

    def load_resource(
        self, resource_id: str, *args: Any, **kwargs: Any
    ) -> List[Document]:
        """Load a single page as a list containing one Document.

        Args:
            resource_id: The page title.

        Returns:
            A one-element list with the page Document, or an empty list on failure.
        """
        page_info = self._get_page_info(resource_id)
        if page_info is None:
            return []

        clean_text, canonical_url = page_info

        # Fetch last_modified for metadata
        timestamps = self._get_pages_last_modified([resource_id])
        last_modified = timestamps.get(resource_id)

        doc = Document(
            text=clean_text,
            metadata={
                "url": canonical_url,
                "title": resource_id,
                "last_modified": last_modified.isoformat() if last_modified else None,
            },
        )
        return [doc]

    # -- Custom batched method (NOT part of LlamaIndex API) -------------------

    def get_resources_info(
        self, page_titles: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Return info for multiple pages in batched API calls.

        This is a **custom extension** — not part of the standard
        ``ResourcesReaderMixin`` interface.  It exists so that downstream jobs
        can retrieve ``last_modified`` and ``url`` for many pages without N+1
        API round-trips.

        Args:
            page_titles: List of page titles.

        Returns:
            Dict mapping each title to
            ``{"last_modified": datetime | None, "url": str | None}``.
        """
        timestamps = self._get_pages_last_modified(page_titles)

        # Batch-fetch URLs via the info API
        urls: Dict[str, Optional[str]] = {}
        if page_titles:
            titles_param = "|".join(page_titles)
            params = {
                "action": "query",
                "titles": titles_param,
                "prop": "info",
                "inprop": "url",
            }
            data = self._make_api_request(params)
            if data:
                pages = data.get("query", {}).get("pages", {})
                for page_data in pages.values():
                    title = page_data.get("title")
                    if title:
                        urls[title] = page_data.get("canonicalurl")

        result: Dict[str, Dict[str, Any]] = {}
        for title in page_titles:
            result[title] = {
                "last_modified": timestamps.get(title),
                "url": urls.get(title),
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
            docs = self.load_resource(title)
            yield from docs
            time.sleep(self.request_delay)
