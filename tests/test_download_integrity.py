"""Guards against the two ways a download could silently return the wrong bytes.

1. A bot wall / captcha interstitial served from a URL ending in `.pdf`.
2. A repository returning a topically similar but different paper.
"""

import asyncio
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from paper_search_mcp import server

PDF_BODY = b"%PDF-1.7\n" + b"x" * server.MIN_PDF_BYTES
CAPTCHA_BODY = b"<!DOCTYPE html>\n<title>Client Challenge</title>" + b" " * 3000


class _FakeResponse:
    def __init__(self, content: bytes, content_type: str, status_code: int = 200):
        self.content = content
        self.headers = {"content-type": content_type}
        self.status_code = status_code


def _client_returning(response):
    client = AsyncMock()
    client.get = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _download(pdf_url: str, response) -> str | None:
    with tempfile.TemporaryDirectory() as tmp:
        with patch("paper_search_mcp.server.httpx.AsyncClient", return_value=_client_returning(response)):
            return asyncio.run(server._download_from_url(pdf_url, tmp, "out"))


class TestDownloadFromUrl(unittest.TestCase):
    def test_rejects_html_served_from_a_pdf_url(self):
        # The .pdf suffix used to be enough to accept the response on its own.
        result = _download(
            "https://link.springer.com/content/pdf/10.3758/BF03207484.pdf",
            _FakeResponse(CAPTCHA_BODY, "text/html; charset=utf-8"),
        )
        self.assertIsNone(result)

    def test_rejects_html_served_with_a_pdf_content_type(self):
        result = _download(
            "https://example.org/article",
            _FakeResponse(CAPTCHA_BODY, "application/pdf"),
        )
        self.assertIsNone(result)

    def test_rejects_implausibly_small_pdf(self):
        result = _download(
            "https://example.org/a.pdf",
            _FakeResponse(b"%PDF-1.4\nstub", "application/pdf"),
        )
        self.assertIsNone(result)

    def test_accepts_real_pdf_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            response = _FakeResponse(PDF_BODY, "application/octet-stream")
            with patch("paper_search_mcp.server.httpx.AsyncClient", return_value=_client_returning(response)):
                result = asyncio.run(server._download_from_url("https://example.org/a", tmp, "out"))
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(result))
            with open(result, "rb") as handle:
                self.assertTrue(handle.read().startswith(b"%PDF"))


class TestRepositoryFallbackIdentity(unittest.TestCase):
    WANTED_DOI = "10.2307/40285905"
    WANTED_TITLE = "The Effect of Key Color and Timbre on Absolute Pitch Recognition in Musical Contexts"

    def _run(self, papers, doi=WANTED_DOI, title=WANTED_TITLE):
        """Route every repository through one stub searcher returning `papers`."""
        hit = SimpleNamespace(search=lambda query, max_results=3: papers)
        empty = SimpleNamespace(search=lambda query, max_results=3: [])
        with patch.object(server, "_download_from_url",
                          new=AsyncMock(side_effect=lambda url, path, hint: f"/tmp/{hint}.pdf")), \
             patch.object(server, "europepmc_searcher", hit), \
             patch.object(server, "openaire_searcher", empty), \
             patch.object(server, "core_searcher", empty), \
             patch.object(server, "pmc_searcher", empty):
            return asyncio.run(server._try_repository_fallback(doi, title, "/tmp"))

    def test_rejects_different_paper_with_a_pdf(self):
        wrong = SimpleNamespace(
            paper_id="PMID_40727046",
            title="Augmenting art crossmodally: possibilities and pitfalls",
            doi="10.3389/fpsyg.2025.1605110",
            pdf_url="https://example.org/wrong.pdf",
        )
        result, error = self._run([wrong])
        self.assertIsNone(result)
        self.assertIn("different paper", error)

    def test_accepts_matching_doi(self):
        right = SimpleNamespace(
            paper_id="repo-1",
            title="The Effect of Key Color and Timbre on Absolute Pitch Recognition in Musical Contexts",
            doi="https://doi.org/10.2307/40285905",
            pdf_url="https://example.org/right.pdf",
        )
        result, _ = self._run([right])
        self.assertEqual(result, "/tmp/europepmc_repo-1.pdf")

    def test_matches_doi_case_insensitively(self):
        right = SimpleNamespace(
            paper_id="repo-2", title="irrelevant", doi="10.2307/40285905", pdf_url="https://example.org/r.pdf"
        )
        result, _ = self._run([right], doi="10.2307/40285905".upper())
        self.assertEqual(result, "/tmp/europepmc_repo-2.pdf")

    def test_title_only_requires_exact_title_match(self):
        near_miss = SimpleNamespace(
            paper_id="repo-3",
            title="Absolute pitch recognition in musical contexts: a replication",
            doi="",
            pdf_url="https://example.org/near.pdf",
        )
        result, error = self._run([near_miss], doi="")
        self.assertIsNone(result)
        self.assertIn("different paper", error)

    def test_title_only_accepts_exact_title_match(self):
        exact = SimpleNamespace(
            paper_id="repo-4", title=self.WANTED_TITLE.upper(), doi="", pdf_url="https://example.org/e.pdf"
        )
        result, _ = self._run([exact], doi="")
        self.assertEqual(result, "/tmp/europepmc_repo-4.pdf")


if __name__ == "__main__":
    unittest.main()
