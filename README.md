# LlamaIndex Readers Integration: MediaWiki

## Overview

The MediaWiki Reader loads pages from any [MediaWiki](https://www.mediawiki.org/)-based wiki (Wikipedia, Wikiversity, or your own instance) and returns them as LlamaIndex `Document` objects. It uses the wiki’s [Action API](https://www.mediawiki.org/wiki/API:Main_page): you provide the API URL, and the reader fetches page list, metadata (URL, last-modified), and parsed text, with optional rate limiting and namespace filtering.

### Features

- **Any MediaWiki instance** — Use `api_url` to point at any wiki (e.g. `https://en.wikipedia.org/w/api.php`).
- **Resource-based API** — Implements `load_resource` and `get_resource_info` for use with LlamaIndex ingestion and RAG pipelines.
- **Efficient listing** — Batched API calls for page metadata; optional `namespaces` filter.
- **HTML to text** — Converts wiki HTML to clean text via html2text (configurable).

### Installation

```bash
pip install llama-index-readers-mediawiki
```

### Usage

**Load a single page by title (requires URL and last_modified from get_resources_info):**

```python
from llama_index.readers.mediawiki import MediaWikiReader

reader = MediaWikiReader(
    api_url="https://en.wikipedia.org/w/api.php",
    user_agent="my-app/1.0",
)

# Get metadata for the page, then load content
info = reader.get_resources_info(["Python (programming language)"])
title = "Python (programming language)"
page_info = info.get(title, {})
if page_info.get("url"):
    docs = reader.load_resource(
        title,
        resource_url=page_info["url"],
        last_modified=page_info.get("last_modified"),
    )
# docs is a list of one Document with .text and .metadata (title, url, last_modified)
```

**Stream all pages (lazy):**

```python
for doc in reader.lazy_load_data():
    print(doc.metadata.get("title"), len(doc.text))
```

**Optional: filter by namespace and tune requests:**

```python
reader = MediaWikiReader(
    api_url="https://mywiki.example.com/w/api.php",
    request_delay=0.2,
    page_limit=100,
    batch_size=50,
    namespaces=[0],  # Main namespace only
)
```

### Configuration

| Parameter       | Type     | Default | Description |
|----------------|----------|---------|-------------|
| `api_url`      | `str`    | required | MediaWiki API endpoint (e.g. `https://en.wikipedia.org/w/api.php`). |
| `user_agent`   | `str`    | `"llama-index-readers-mediawiki/1.0"` | User-Agent header for API requests. |
| `request_delay`| `float`  | `0.1`   | Delay in seconds between API requests (rate limiting). |
| `page_limit`   | `int`    | `500`   | When listing pages: max titles per API call (allpages pagination). |
| `batch_size`   | `int`    | `50`    | When fetching metadata for multiple pages: titles per API call (get_resources_info). |
| `max_retries`  | `int`    | `3`     | Retry attempts for failed requests. |
| `timeout`      | `int`    | `30`    | HTTP request timeout in seconds. |
| `namespaces`   | `list[int] \| None` | `None` | Namespace IDs to list; `None` = wiki content namespaces from siteinfo API ([$wgContentNamespaces](https://www.mediawiki.org/wiki/Manual:$wgContentNamespaces)). |
| `logger`       | `logging.Logger` | module logger | Logger instance (injectable for tests or custom logging). Not serialized. |

### License

MIT.

---

This loader is designed to be used as a way to load data into [LlamaIndex](https://github.com/run-llama/llama_index) and/or subsequently as a Tool in a [LangChain](https://github.com/hwchase17/langchain) Agent.
