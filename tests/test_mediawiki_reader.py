"""Tests for MediaWikiReader (Pytest version)."""

import logging

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import requests
from pydantic import ValidationError

from llama_index.readers.mediawiki import MediaWikiReader


def test_class():
    """MediaWikiReader must inherit from the LlamaIndex base reader."""
    names_of_base_classes = [b.__name__ for b in MediaWikiReader.__mro__]
    assert "BasePydanticReader" in names_of_base_classes


@pytest.fixture
def mock_session_cls():
    """Mock the requests.Session class."""
    with patch("llama_index.readers.mediawiki.base.requests.Session") as mocked:
        yield mocked


@pytest.fixture
def mock_session(mock_session_cls):
    """Provide a mock session instance."""
    session = Mock()
    mock_session_cls.return_value = session
    return session


def _make_reader(**overrides):
    """Create a MediaWikiReader with sensible defaults."""
    kwargs = {"api_url": "https://example.com/w/api.php"}
    kwargs.update(overrides)
    return MediaWikiReader(**kwargs)


@pytest.fixture
def reader(mock_session):
    """Provide a MediaWikiReader instance."""
    return _make_reader()


def _mock_response(status_code=200, json_data=None, json_exception=None):
    """Build a mock requests.Response."""
    resp = Mock()
    resp.status_code = status_code
    resp.raise_for_status = Mock()
    if json_exception:
        resp.json.side_effect = json_exception
    elif json_data is not None:
        resp.json.return_value = json_data
    return resp


class TestMediaWikiReaderInit:
    """Construction and config validation."""

    def test_missing_api_url_raises(self, mock_session_cls):
        with pytest.raises(ValidationError, match="api_url"):
            MediaWikiReader(api_url="")

    def test_negative_request_delay_raises(self, mock_session_cls):
        with pytest.raises(ValidationError, match="request_delay"):
            _make_reader(request_delay=-1)

    def test_zero_page_limit_raises(self, mock_session_cls):
        with pytest.raises(ValidationError, match="page_limit"):
            _make_reader(page_limit=0)

    def test_negative_batch_size_raises(self, mock_session_cls):
        with pytest.raises(ValidationError, match="batch_size"):
            _make_reader(batch_size=-1)

    def test_negative_max_retries_raises(self, mock_session_cls):
        with pytest.raises(ValidationError, match="max_retries"):
            _make_reader(max_retries=-1)

    def test_zero_timeout_raises(self, mock_session_cls):
        with pytest.raises(ValidationError, match="timeout"):
            _make_reader(timeout=0)

    def test_logger_injection(self, mock_session_cls):
        """Caller can inject a custom logger (e.g. for tests or logging config)."""
        custom = logging.getLogger("custom.mediawiki")
        reader = _make_reader(logger=custom)
        assert reader.logger is custom


class TestMakeApiRequest:
    """Low-level _make_api_request behaviour."""

    def test_success(self, reader, mock_session):
        mock_session.get.return_value = _mock_response(json_data={"ok": True})
        result = reader._make_api_request({"action": "query"})
        assert result == {"ok": True}

    def test_network_error_retries(self, reader, mock_session):
        mock_session.get.side_effect = requests.exceptions.RequestException("err")
        result = reader._make_api_request({"action": "query"})
        assert result is None
        assert mock_session.get.call_count == 4

    def test_rate_limiting_429(self, reader, mock_session):
        rate_resp = Mock()
        rate_resp.status_code = 429
        rate_resp.headers = {"Retry-After": "2"}
        rate_resp.raise_for_status = Mock(
            side_effect=requests.exceptions.HTTPError(response=rate_resp)
        )

        ok_resp = _mock_response(json_data={"test": "data"})
        mock_session.get.side_effect = [rate_resp, ok_resp]

        result = reader._make_api_request({"action": "query"})
        assert result == {"test": "data"}
        assert mock_session.get.call_count == 2

    @patch("llama_index.readers.mediawiki.base.time.sleep")
    def test_timeout(self, mock_sleep, reader, mock_session):
        mock_session.get.side_effect = requests.exceptions.Timeout("timeout")
        result = reader._make_api_request({"action": "query"})
        assert result is None
        assert mock_session.get.call_count == 4

    @patch("llama_index.readers.mediawiki.base.time.sleep")
    def test_exponential_backoff(self, mock_sleep, reader, mock_session):
        mock_session.get.side_effect = requests.exceptions.RequestException("err")
        reader._make_api_request({"action": "query"})
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert delays == [1, 2, 4]

    def test_empty_response(self, reader, mock_session):
        mock_session.get.return_value = _mock_response(json_data={})
        result = reader._make_api_request({"action": "query"})
        assert result == {}

    def test_invalid_json(self, reader, mock_session):
        mock_session.get.return_value = _mock_response(
            json_exception=ValueError("bad json")
        )
        result = reader._make_api_request({"action": "query"})
        assert result is None

    def test_zero_retries(self, reader, mock_session):
        """max_retries=0 should still allow 1 attempt."""
        mock_session.get.return_value = _mock_response(json_data={"ok": True})
        result = reader._make_api_request({"action": "query"}, max_retries=0)
        assert result == {"ok": True}
        assert mock_session.get.call_count == 1


class TestGetAllPages:
    """Page listing and pagination."""

    def test_generator_rich_response(self, mock_session):
        reader = _make_reader(namespaces=[0])  # explicit so we don't call siteinfo
        mock_session.get.return_value = _mock_response(json_data={
            "query": {"pages": {
                "1": {
                    "title": "Page 1",
                    "canonicalurl": "https://example.com/Page_1",
                    "revisions": [{"timestamp": "2024-01-01T12:00:00Z"}]
                }
            }}
        })

        pages = list(reader._get_all_pages_generator())
        assert len(pages) == 1
        assert pages[0]["title"] == "Page 1"
        assert pages[0]["url"] == "https://example.com/Page_1"
        assert pages[0]["last_modified"].year == 2024

    @patch("llama_index.readers.mediawiki.base.time.sleep")
    def test_pagination(self, mock_sleep, mock_session):
        reader = _make_reader(namespaces=[0])  # explicit so we don't call siteinfo
        first = _mock_response(json_data={
            "query": {"pages": {"1": {"title": "Page 1"}}},
            "continue": {"gapcontinue": "Page_2", "continue": "gapcontinue||"},
        })
        second = _mock_response(json_data={
            "query": {"pages": {"2": {"title": "Page 2"}}}
        })
        mock_session.get.side_effect = [first, second]

        pages = list(reader._get_all_pages_generator())
        assert len(pages) == 2
        assert mock_session.get.call_count == 2
        mock_sleep.assert_called_once()

    def test_namespace_iteration(self, mock_session):
        # Multiple namespaces should trigger multiple API call series
        reader = _make_reader(namespaces=[0, 1])

        resp_ns0 = _mock_response(json_data={"query": {"pages": {"1": {"title": "A"}}}})
        resp_ns1 = _mock_response(json_data={"query": {"pages": {"2": {"title": "Talk:A"}}}})
        mock_session.get.side_effect = [resp_ns0, resp_ns1]

        pages = list(reader._get_all_pages_generator())
        assert len(pages) == 2
        assert mock_session.get.call_count == 2

        # Verify gapnamespace was passed correctly for each call
        assert mock_session.get.call_args_list[0][1]["params"]["gapnamespace"] == 0
        assert mock_session.get.call_args_list[1][1]["params"]["gapnamespace"] == 1

    def test_content_namespaces_default(self, mock_session):
        """When namespaces is None, reader fetches content namespaces via siteinfo and uses them."""
        siteinfo_resp = _mock_response(json_data={
            "query": {
                "namespaces": {
                    "0": {"id": 0, "*": "", "content": True},
                    "1": {"id": 1, "*": "Talk", "content": False},
                    "4": {"id": 4, "*": "Project", "content": True},
                }
            }
        })
        allpages_ns0 = _mock_response(json_data={
            "query": {"pages": {"1": {"title": "Page1", "canonicalurl": "https://example.com/Page1", "revisions": [{"timestamp": "2024-01-01T00:00:00Z"}]}}}
        })
        allpages_ns4 = _mock_response(json_data={
            "query": {"pages": {"2": {"title": "Project:About", "canonicalurl": "https://example.com/Project:About", "revisions": [{"timestamp": "2024-01-02T00:00:00Z"}]}}}
        })
        mock_session.get.side_effect = [siteinfo_resp, allpages_ns0, allpages_ns4]

        reader = _make_reader(namespaces=None)
        pages = list(reader._get_all_pages_generator())

        assert mock_session.get.call_count == 3
        first_params = mock_session.get.call_args_list[0][1]["params"]
        assert first_params["action"] == "query"
        assert first_params["meta"] == "siteinfo"
        assert first_params["siprop"] == "namespaces"
        assert mock_session.get.call_args_list[1][1]["params"]["gapnamespace"] == 0
        assert mock_session.get.call_args_list[2][1]["params"]["gapnamespace"] == 4
        assert len(pages) == 2
        assert pages[0]["title"] == "Page1"
        assert pages[1]["title"] == "Project:About"

    def test_fetch_content_namespace_ids(self, mock_session):
        """_fetch_content_namespace_ids returns IDs where content is true; fallback to [0] on failure."""
        reader = _make_reader()

        mock_session.get.return_value = _mock_response(json_data={
            "query": {
                "namespaces": {
                    "0": {"id": 0, "*": "", "content": True},
                    "4": {"id": 4, "*": "Project", "content": True},
                }
            }
        })
        ids = reader._fetch_content_namespace_ids()
        assert ids == [0, 4]

        mock_session.get.return_value = _mock_response(json_data={})
        ids_empty = reader._fetch_content_namespace_ids()
        assert ids_empty == [0]

    def test_fetch_content_namespace_ids_returns_none(self, mock_session):
        """When _make_api_request returns None (e.g. network failure), fallback to [0]."""
        reader = _make_reader()
        mock_session.get.side_effect = requests.exceptions.RequestException("network error")
        ids = reader._fetch_content_namespace_ids()
        assert ids == [0]

    def test_content_namespaces_cache_fetched_once(self, mock_session):
        """Second call to _get_all_pages_generator() does not refetch siteinfo; cache is per instance."""
        siteinfo_resp = _mock_response(json_data={
            "query": {
                "namespaces": {
                    "0": {"id": 0, "*": "", "content": True},
                }
            }
        })
        allpages_resp = _mock_response(json_data={
            "query": {"pages": {"1": {"title": "A", "canonicalurl": "https://example.com/A", "revisions": [{"timestamp": "2024-01-01T00:00:00Z"}]}}}
        })
        mock_session.get.side_effect = [siteinfo_resp, allpages_resp, allpages_resp]

        reader = _make_reader(namespaces=None)
        list(reader._get_all_pages_generator())
        list(reader._get_all_pages_generator())

        call_params_list = [c[1]["params"] for c in mock_session.get.call_args_list]
        siteinfo_calls = [p for p in call_params_list if p.get("meta") == "siteinfo" and p.get("siprop") == "namespaces"]
        assert len(siteinfo_calls) == 1


class TestGetPageContents:
    """Content retrieval via parse action."""

    def test_success(self, reader, mock_session):
        """_get_page_contents returns raw HTML; caller converts via _html_to_clean_text."""
        parse_resp = _mock_response(json_data={
            "parse": {"text": {"*": "<p>Test page content with <a href='/wiki/Links'>links</a>.</p>"}}
        })
        mock_session.get.return_value = parse_resp

        result = reader._get_page_contents("Test Page")
        assert result is not None
        assert "Test page content" in result
        assert "<p>" in result  # raw HTML, not converted to text yet
        assert mock_session.get.call_count == 1

    def test_no_parse_result(self, reader, mock_session):
        mock_session.get.return_value = _mock_response(json_data={})
        assert reader._get_page_contents("Missing") is None


class TestHtmlToCleanText:
    """HTML-to-text conversion."""

    def test_basic_html(self, reader):
        result = reader._html_to_clean_text("<p>Hello <b>world</b></p>")
        assert "Hello" in result
        assert "world" in result

    def test_preserves_structure(self, reader):
        html = (
            "<h1>Title</h1>"
            "<p>Paragraph with <em>emphasis</em> and <strong>strong</strong>.</p>"
            "<ul><li>Item 1</li><li>Item 2</li></ul>"
        )
        result = reader._html_to_clean_text(html)
        assert "Title" in result
        assert "Item 1" in result
        assert "Item 2" in result


class TestResourcesInterface:
    """Public resource-based API."""

    def test_load_resource_with_prefetched_metadata(self, reader, mock_session):
        """load_resource should bypass metadata calls if url/timestamp are provided."""
        parse_resp = _mock_response(json_data={
            "parse": {"text": {"*": "<p>Content</p>"}}
        })
        mock_session.get.return_value = parse_resp

        timestamp = datetime(2024, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
        docs = reader.load_resource(
            "P", resource_url="https://wiki.com/P", last_modified=timestamp
        )

        assert len(docs) == 1
        assert docs[0].text == "Content"
        assert docs[0].metadata["url"] == "https://wiki.com/P"
        assert docs[0].metadata["last_modified"] == timestamp.isoformat()

        # Should only call 'parse', not 'info' or 'revisions'
        mock_session.get.assert_called_once()
        args, kwargs = mock_session.get.call_args
        assert kwargs["params"]["action"] == "parse"

    def test_load_resource_with_required_url_and_timestamp(self, reader, mock_session):
        """load_resource requires resource_url and last_modified; only calls parse."""
        parse_resp = _mock_response(json_data={
            "parse": {"text": {"*": "Content"}}
        })
        mock_session.get.return_value = parse_resp

        timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        docs = reader.load_resource(
            "Page",
            resource_url="https://example.com/wiki/Page",
            last_modified=timestamp,
        )
        assert len(docs) == 1
        assert "Content" in docs[0].text
        assert docs[0].metadata["url"] == "https://example.com/wiki/Page"
        assert docs[0].metadata["title"] == "Page"
        mock_session.get.assert_called_once()
        assert mock_session.get.call_args[1]["params"]["action"] == "parse"

    def test_load_resource_missing_page(self, reader, mock_session):
        """When parse returns no content, load_resource returns []."""
        mock_session.get.return_value = _mock_response(json_data={})
        docs = reader.load_resource(
            "Missing",
            resource_url="https://example.com/wiki/Missing",
            last_modified=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert docs == []

    def test_get_resource_info(self, reader, mock_session):
        ts_url_resp = _mock_response(json_data={
            "query": {"pages": {"1": {
                "pageid": 1, "title": "Page",
                "canonicalurl": "https://example.com/wiki/Page",
                "revisions": [{"timestamp": "2024-06-01T00:00:00Z"}],
            }}}
        })

        mock_session.get.side_effect = [ts_url_resp]

        info = reader.get_resource_info("Page")
        assert "last_modified" in info
        assert "url" in info
        assert info["url"] == "https://example.com/wiki/Page"
        assert mock_session.get.call_count == 1

    def test_get_resources_info_batched(self, reader, mock_session):
        ts_url_resp = _mock_response(json_data={
            "query": {"pages": {
                "1": {
                    "title": "A", "pageid": 1,
                    "canonicalurl": "https://example.com/wiki/A",
                    "revisions": [{"timestamp": "2024-01-01T00:00:00Z"}],
                },
                "2": {
                    "title": "B", "pageid": 2,
                    "canonicalurl": "https://example.com/wiki/B",
                    "revisions": [{"timestamp": "2024-02-01T00:00:00Z"}],
                },
            }}
        })
        mock_session.get.side_effect = [ts_url_resp]

        info = reader.get_resources_info(["A", "B"])
        assert len(info) == 2
        assert info["A"]["last_modified"] is not None
        assert info["B"]["url"] == "https://example.com/wiki/B"
        # Should only need 1 API call (consolidated)
        assert mock_session.get.call_count == 1

    def test_get_resources_info_chunking(self, mock_session):
        """Verify that get_resources_info respects batch_size."""
        reader = _make_reader(batch_size=1)

        # Mock responses for 2 separate batches
        resp1 = _mock_response(json_data={"query": {"pages": {"1": {"title": "A"}}}})
        resp2 = _mock_response(json_data={"query": {"pages": {"2": {"title": "B"}}}})
        mock_session.get.side_effect = [resp1, resp2]

        reader.get_resources_info(["A", "B"])

        assert mock_session.get.call_count == 2
