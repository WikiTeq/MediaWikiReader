"""Tests for MediaWikiReader."""

import unittest
from unittest.mock import Mock, patch

import requests

from llama_index.readers.mediawiki import MediaWikiReader


def _make_reader(**overrides):
    """Create a MediaWikiReader with sensible defaults and a mocked session."""
    kwargs = {"api_url": "https://example.com/w/api.php"}
    kwargs.update(overrides)
    return MediaWikiReader(**kwargs)


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


class TestMediaWikiReaderInit(unittest.TestCase):
    """Construction and config validation."""

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_missing_api_url_raises(self, _mock_sess):
        with self.assertRaises(ValueError):
            MediaWikiReader(api_url="")

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_negative_request_delay_raises(self, _mock_sess):
        with self.assertRaises(ValueError):
            _make_reader(request_delay=-1)

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_zero_page_limit_raises(self, _mock_sess):
        with self.assertRaises(ValueError):
            _make_reader(page_limit=0)

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_negative_batch_size_raises(self, _mock_sess):
        with self.assertRaises(ValueError):
            _make_reader(batch_size=-1)

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_negative_max_retries_raises(self, _mock_sess):
        with self.assertRaises(ValueError):
            _make_reader(max_retries=-1)

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_zero_timeout_raises(self, _mock_sess):
        with self.assertRaises(ValueError):
            _make_reader(timeout=0)

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_defaults(self, mock_sess_cls):
        reader = _make_reader()
        self.assertEqual(reader.request_delay, 0.1)
        self.assertEqual(reader.page_limit, 500)
        self.assertEqual(reader.batch_size, 50)
        self.assertEqual(reader.max_retries, 3)
        self.assertEqual(reader.timeout, 30)
        self.assertIsNone(reader.namespaces)
        self.assertTrue(reader.is_remote)


class TestMakeApiRequest(unittest.TestCase):
    """Low-level _make_api_request behaviour."""

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_success(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()
        mock_session.get.return_value = _mock_response(json_data={"ok": True})

        result = reader._make_api_request({"action": "query"})
        self.assertEqual(result, {"ok": True})

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_network_error_retries(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()
        mock_session.get.side_effect = requests.exceptions.RequestException("err")

        result = reader._make_api_request({"action": "query"})
        self.assertIsNone(result)
        self.assertEqual(mock_session.get.call_count, 3)

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_rate_limiting_429(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        rate_resp = Mock()
        rate_resp.status_code = 429
        rate_resp.headers = {"Retry-After": "2"}

        ok_resp = _mock_response(json_data={"test": "data"})
        mock_session.get.side_effect = [rate_resp, ok_resp]

        result = reader._make_api_request({"action": "query"})
        self.assertEqual(result, {"test": "data"})
        self.assertEqual(mock_session.get.call_count, 2)

    @patch("llama_index.readers.mediawiki.base.time.sleep")
    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_timeout(self, mock_sess_cls, _mock_sleep):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()
        mock_session.get.side_effect = requests.exceptions.Timeout("timeout")

        result = reader._make_api_request({"action": "query"})
        self.assertIsNone(result)
        self.assertEqual(mock_session.get.call_count, 3)

    @patch("llama_index.readers.mediawiki.base.time.sleep")
    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_exponential_backoff(self, mock_sess_cls, mock_sleep):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()
        mock_session.get.side_effect = requests.exceptions.RequestException("err")

        reader._make_api_request({"action": "query"})
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        self.assertEqual(delays, [1, 2])

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_empty_response(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()
        mock_session.get.return_value = _mock_response(json_data={})

        result = reader._make_api_request({"action": "query"})
        self.assertEqual(result, {})

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_invalid_json(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()
        mock_session.get.return_value = _mock_response(
            json_exception=ValueError("bad json")
        )

        result = reader._make_api_request({"action": "query"})
        self.assertIsNone(result)

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_zero_retries(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        result = reader._make_api_request({"action": "query"}, max_retries=0)
        self.assertIsNone(result)
        mock_session.get.assert_not_called()


class TestGetAllPages(unittest.TestCase):
    """Page listing and pagination."""

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_generator_rich_response(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        # Mock generator response format (query.pages)
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
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0]["title"], "Page 1")
        self.assertEqual(pages[0]["url"], "https://example.com/Page_1")
        self.assertEqual(pages[0]["last_modified"].year, 2024)

    @patch("llama_index.readers.mediawiki.base.time.sleep")
    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_pagination(self, mock_sess_cls, mock_sleep):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        first = _mock_response(json_data={
            "query": {"pages": {"1": {"title": "Page 1"}}},
            "continue": {"gapcontinue": "Page_2", "continue": "gapcontinue||"},
        })
        second = _mock_response(json_data={
            "query": {"pages": {"2": {"title": "Page 2"}}}
        })
        mock_session.get.side_effect = [first, second]

        pages = list(reader._get_all_pages())
        self.assertEqual(len(pages), 2)
        self.assertEqual(mock_session.get.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_namespace_iteration(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        # Multiple namespaces should trigger multiple API call series
        reader = _make_reader(namespaces=[0, 1])

        resp_ns0 = _mock_response(json_data={"query": {"pages": {"1": {"title": "A"}}}})
        resp_ns1 = _mock_response(json_data={"query": {"pages": {"2": {"title": "Talk:A"}}}})
        mock_session.get.side_effect = [resp_ns0, resp_ns1]

        pages = list(reader._get_all_pages())
        self.assertEqual(len(pages), 2)
        self.assertEqual(mock_session.get.call_count, 2)

        # Verify gapnamespace was passed correctly for each call
        self.assertEqual(mock_session.get.call_args_list[0][1]["params"]["gapnamespace"], 0)
        self.assertEqual(mock_session.get.call_args_list[1][1]["params"]["gapnamespace"], 1)


class TestGetPageInfo(unittest.TestCase):
    """Content + URL retrieval."""

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_success(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        url_resp = _mock_response(json_data={
            "query": {"pages": {"123": {
                "pageid": 123, "title": "Test Page",
                "canonicalurl": "https://example.com/wiki/Test_Page",
            }}}
        })
        parse_resp = _mock_response(json_data={
            "parse": {"text": {"*": "<p>Test page content with <a href='/wiki/Links'>links</a>.</p>"}}
        })
        mock_session.get.side_effect = [url_resp, parse_resp]

        result = reader._get_page_info("Test Page")
        self.assertIsNotNone(result)
        content, url = result
        self.assertIn("Test page content", content)
        self.assertEqual(url, "https://example.com/wiki/Test_Page")
        self.assertEqual(mock_session.get.call_count, 2)

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_missing_page(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        mock_session.get.return_value = _mock_response(json_data={
            "query": {"pages": {"-1": {"title": "Missing", "missing": True}}}
        })

        self.assertIsNone(reader._get_page_info("Missing"))

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_missing_canonical_url(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        mock_session.get.return_value = _mock_response(json_data={
            "query": {"pages": {"789": {"pageid": 789, "title": "No URL"}}}
        })

        self.assertIsNone(reader._get_page_info("No URL"))


class TestGetPagesLastModified(unittest.TestCase):
    """Batched timestamp retrieval."""

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_success(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        mock_session.get.return_value = _mock_response(json_data={
            "query": {"pages": {"123": {
                "title": "Test Page", "pageid": 123,
                "revisions": [{"timestamp": "2024-01-01T12:00:00Z"}],
            }}}
        })

        ts = reader._get_pages_last_modified(["Test Page"])
        self.assertIsNotNone(ts["Test Page"])
        self.assertEqual(ts["Test Page"].year, 2024)

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_missing_page(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        mock_session.get.return_value = _mock_response(json_data={
            "query": {"pages": {"-1": {"title": "Missing", "missing": True}}}
        })

        ts = reader._get_pages_last_modified(["Missing"])
        self.assertIsNone(ts["Missing"])

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_no_revisions(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        mock_session.get.return_value = _mock_response(json_data={
            "query": {"pages": {"123": {"pageid": 123, "title": "Test Page"}}}
        })

        ts = reader._get_pages_last_modified(["Test Page"])
        self.assertIsNone(ts["Test Page"])

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_timezone_offset(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        mock_session.get.return_value = _mock_response(json_data={
            "query": {"pages": {"123": {
                "title": "Test Page", "pageid": 123,
                "revisions": [{"timestamp": "2024-01-01T12:00:00-05:00"}],
            }}}
        })

        ts = reader._get_pages_last_modified(["Test Page"])
        self.assertIsNotNone(ts["Test Page"])
        self.assertEqual(ts["Test Page"].hour, 12)

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_invalid_timestamp(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        mock_session.get.return_value = _mock_response(json_data={
            "query": {"pages": {"123": {
                "title": "Test Page", "pageid": 123,
                "revisions": [{"timestamp": "invalid-timestamp"}],
            }}}
        })

        ts = reader._get_pages_last_modified(["Test Page"])
        self.assertIsNone(ts["Test Page"])

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_empty_list(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        ts = reader._get_pages_last_modified([])
        self.assertEqual(ts, {})


class TestHtmlToCleanText(unittest.TestCase):
    """HTML-to-text conversion."""

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_basic_html(self, mock_sess_cls):
        reader = _make_reader()
        result = reader._html_to_clean_text("<p>Hello <b>world</b></p>")
        self.assertIn("Hello", result)
        self.assertIn("world", result)

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_preserves_structure(self, mock_sess_cls):
        reader = _make_reader()
        html = (
            "<h1>Title</h1>"
            "<p>Paragraph with <em>emphasis</em> and <strong>strong</strong>.</p>"
            "<ul><li>Item 1</li><li>Item 2</li></ul>"
        )
        result = reader._html_to_clean_text(html)
        self.assertIn("Title", result)
        self.assertIn("Item 1", result)
        self.assertIn("Item 2", result)


class TestResourcesInterface(unittest.TestCase):
    """Public resource-based API."""

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_list_resources(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        mock_session.get.return_value = _mock_response(json_data={
            "query": {"pages": {"1": {"title": "A"}, "2": {"title": "B"}}}
        })

        titles = reader.list_resources()
        self.assertEqual(titles, ["A", "B"])

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_load_resource_success(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        url_resp = _mock_response(json_data={
            "query": {"pages": {"1": {
                "pageid": 1, "title": "Page",
                "canonicalurl": "https://example.com/wiki/Page",
            }}}
        })
        parse_resp = _mock_response(json_data={
            "parse": {"text": {"*": "<p>Content</p>"}}
        })
        ts_resp = _mock_response(json_data={
            "query": {"pages": {"1": {
                "title": "Page", "pageid": 1,
                "revisions": [{"timestamp": "2024-06-01T00:00:00Z"}],
            }}}
        })

        mock_session.get.side_effect = [url_resp, parse_resp, ts_resp]

        docs = reader.load_resource("Page")
        self.assertEqual(len(docs), 1)
        self.assertIn("Content", docs[0].text)
        self.assertEqual(docs[0].metadata["url"], "https://example.com/wiki/Page")
        self.assertEqual(docs[0].metadata["title"], "Page")

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_load_resource_missing_page(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        mock_session.get.return_value = _mock_response(json_data={
            "query": {"pages": {"-1": {"title": "Missing", "missing": True}}}
        })

        docs = reader.load_resource("Missing")
        self.assertEqual(docs, [])

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_get_resource_info(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        ts_resp = _mock_response(json_data={
            "query": {"pages": {"1": {
                "title": "Page", "pageid": 1,
                "revisions": [{"timestamp": "2024-01-01T00:00:00Z"}],
            }}}
        })
        url_resp = _mock_response(json_data={
            "query": {"pages": {"1": {
                "pageid": 1, "title": "Page",
                "canonicalurl": "https://example.com/wiki/Page",
            }}}
        })

        mock_session.get.side_effect = [ts_resp, url_resp]

        info = reader.get_resource_info("Page")
        self.assertIn("last_modified", info)
        self.assertIn("url", info)
        self.assertEqual(info["url"], "https://example.com/wiki/Page")

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_get_resources_info_batched(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()

        ts_resp = _mock_response(json_data={
            "query": {"pages": {
                "1": {
                    "title": "A", "pageid": 1,
                    "revisions": [{"timestamp": "2024-01-01T00:00:00Z"}],
                },
                "2": {
                    "title": "B", "pageid": 2,
                    "revisions": [{"timestamp": "2024-02-01T00:00:00Z"}],
                },
            }}
        })
        url_resp = _mock_response(json_data={
            "query": {"pages": {
                "1": {"pageid": 1, "title": "A", "canonicalurl": "https://example.com/wiki/A"},
                "2": {"pageid": 2, "title": "B", "canonicalurl": "https://example.com/wiki/B"},
            }}
        })
        mock_session.get.side_effect = [ts_resp, url_resp]

        info = reader.get_resources_info(["A", "B"])
        self.assertEqual(len(info), 2)
        self.assertIsNotNone(info["A"]["last_modified"])
        self.assertEqual(info["B"]["url"], "https://example.com/wiki/B")
        # Should only need 2 API calls (1 timestamps + 1 URLs), not 2*N
        self.assertEqual(mock_session.get.call_count, 2)


class TestSessionLifecycle(unittest.TestCase):
    """Session management and context manager."""

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_close(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        reader = _make_reader()
        reader.close()
        mock_session.close.assert_called_once()

    @patch("llama_index.readers.mediawiki.base.requests.Session")
    def test_context_manager(self, mock_sess_cls):
        mock_session = Mock()
        mock_sess_cls.return_value = mock_session
        with _make_reader() as reader:
            self.assertIsNotNone(reader)
        mock_session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
