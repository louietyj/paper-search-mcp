"""Sci-Hub mirror failover, captcha retry, and PDF validation.

A captcha page or a dead mirror must not look like a missing paper.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests

from paper_search_mcp.academic_platforms import sci_hub
from paper_search_mcp.academic_platforms.sci_hub import (
    MIN_PDF_BYTES,
    SciHubFetcher,
    SciHubRateLimitedError,
    SciHubUnavailableError,
)

PDF_BYTES = b"%PDF-1.7\n" + b"x" * MIN_PDF_BYTES
CAPTCHA_HTML = (
    '<html><body><h1>Some Paper Title</h1>'
    "<script>fetch('/captcha/solution/8529266')</script></body></html>"
)
# A served page carries the same captcha script, but also the PDF embed.
EMBED_WITH_CAPTCHA_SCRIPT_HTML = (
    '<html><body><embed type="application/pdf" src="/downloads/paper.pdf">'
    "<script>fetch('/captcha/solution/8529266')</script></body></html>"
)
EMBED_HTML = '<embed type="application/pdf" src="/downloads/paper.pdf">'
NOT_IN_DB_HTML = (
    "<html><body>Alas, the following paper is not yet available in my database"
    "</body></html>"
)


def _page(text, url="https://sci-hub.ru/10.1000/test", status=200):
    return SimpleNamespace(status_code=status, url=url, text=text, content=text.encode())


def _pdf(content=PDF_BYTES, url="https://sci-hub.ru/downloads/paper.pdf"):
    return SimpleNamespace(status_code=200, url=url, text="", content=content)


class TestNotInDatabase(unittest.TestCase):
    def test_raises_rather_than_returning_none(self):
        fetcher = SciHubFetcher(base_url="https://sci-hub.ru", output_dir=".")
        with patch.object(fetcher.session, "get", return_value=_page(NOT_IN_DB_HTML)):
            with self.assertRaises(SciHubUnavailableError):
                fetcher._get_direct_url("10.1027/1618-3169/a000153")

    def test_download_stops_immediately_without_retrying(self):
        fetcher = SciHubFetcher(output_dir=".")
        with patch.object(fetcher.session, "get", return_value=_page(NOT_IN_DB_HTML)) as get, \
             patch.object(sci_hub.time, "sleep"):
            self.assertIsNone(fetcher.download_pdf("10.1027/1618-3169/a000153"))

        # One request total: no captcha retries, no walking the other mirrors.
        self.assertEqual(get.call_count, 1)
        self.assertIn("database", fetcher.last_failure_reason)


class TestCaptchaRetry(unittest.TestCase):
    def test_captcha_page_is_reported_as_rate_limiting(self):
        fetcher = SciHubFetcher(base_url="https://sci-hub.ru", output_dir=".")
        with patch.object(fetcher.session, "get", return_value=_page(CAPTCHA_HTML)):
            with self.assertRaises(SciHubRateLimitedError):
                fetcher._get_direct_url("10.1000/test")

    def test_captcha_script_does_not_reject_a_page_that_has_the_pdf(self):
        """The widget ships on served pages too, so the link must win over the marker."""
        fetcher = SciHubFetcher(base_url="https://sci-hub.ru", output_dir=".")
        with patch.object(fetcher.session, "get", return_value=_page(EMBED_WITH_CAPTCHA_SCRIPT_HTML)):
            self.assertEqual(
                fetcher._get_direct_url("10.1000/test"),
                "https://sci-hub.ru/downloads/paper.pdf",
            )

    def test_retries_past_a_captcha_and_succeeds(self):
        fetcher = SciHubFetcher(base_url="https://sci-hub.ru", output_dir=".")
        responses = [_page(CAPTCHA_HTML), _page(EMBED_HTML), _pdf()]
        with patch.object(fetcher.session, "get", side_effect=responses), \
             patch.object(sci_hub.time, "sleep"), \
             patch("builtins.open", unittest.mock.mock_open()):
            result = fetcher.download_pdf("10.1000/test")

        self.assertIsNotNone(result)

    def test_gives_up_after_the_attempt_budget(self):
        fetcher = SciHubFetcher(base_url="https://sci-hub.ru", output_dir=".")
        with patch.object(fetcher.session, "get", return_value=_page(CAPTCHA_HTML)) as get, \
             patch.object(sci_hub.time, "sleep"):
            self.assertIsNone(fetcher.download_pdf("10.1000/test"))

        self.assertEqual(get.call_count, sci_hub.CAPTCHA_ATTEMPTS)
        self.assertIn("rate-limited", fetcher.last_failure_reason)

    def test_rejected_bytes_are_reported_distinctly(self):
        fetcher = SciHubFetcher(base_url="https://sci-hub.ru", output_dir=".")
        responses = [_page(EMBED_HTML), _pdf(content=b"<!DOCTYPE html>stub")]
        with patch.object(fetcher.session, "get", side_effect=responses), \
             patch.object(sci_hub.time, "sleep"):
            self.assertIsNone(fetcher.download_pdf("10.1000/test"))

        self.assertIn("not a valid PDF", fetcher.last_failure_reason)

    def test_empty_identifier_is_reported(self):
        fetcher = SciHubFetcher(base_url="https://sci-hub.ru", output_dir=".")
        self.assertIsNone(fetcher.download_pdf("   "))
        self.assertIn("no identifier", fetcher.last_failure_reason)

    def test_unparseable_page_is_not_retried(self):
        """Only a captcha is worth a retry; a page with no link will not grow one."""
        fetcher = SciHubFetcher(base_url="https://sci-hub.ru", output_dir=".")
        with patch.object(fetcher.session, "get", return_value=_page("<html>nothing</html>")) as get, \
             patch.object(sci_hub.time, "sleep"):
            self.assertIsNone(fetcher.download_pdf("10.1000/test"))

        self.assertEqual(get.call_count, 1)
        self.assertIn("no PDF link", fetcher.last_failure_reason)


class TestMirrorFailover(unittest.TestCase):
    def test_moves_to_the_next_mirror_when_one_is_unreachable(self):
        fetcher = SciHubFetcher(output_dir=".")
        fetcher.mirrors = ["https://dead.example", "https://live.example"]
        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            if "dead.example" in url:
                raise requests.ConnectionError("NXDOMAIN")
            if url.endswith("/10.1000/test"):
                return _page(EMBED_HTML, url="https://live.example/10.1000/test")
            return _pdf(url="https://live.example/downloads/paper.pdf")

        with patch.object(fetcher.session, "get", side_effect=fake_get), \
             patch.object(sci_hub.time, "sleep"), \
             patch("builtins.open", unittest.mock.mock_open()):
            result = fetcher.download_pdf("10.1000/test")

        self.assertIsNotNone(result)
        self.assertTrue(any("live.example" in url for url in calls))

    def test_dead_mirror_is_not_retried_for_captcha(self):
        fetcher = SciHubFetcher(output_dir=".")
        fetcher.mirrors = ["https://dead.example"]
        with patch.object(fetcher.session, "get", side_effect=requests.ConnectionError("x")) as get, \
             patch.object(sci_hub.time, "sleep"):
            self.assertIsNone(fetcher.download_pdf("10.1000/test"))

        self.assertEqual(get.call_count, 1)


class TestPdfValidation(unittest.TestCase):
    def _save(self, content):
        fetcher = SciHubFetcher(base_url="https://sci-hub.ru", output_dir=".")
        with patch.object(fetcher.session, "get", return_value=_pdf(content=content)), \
             patch("builtins.open", unittest.mock.mock_open()):
            return fetcher._save_pdf("https://sci-hub.ru/x.pdf", "10.1000/test")

    def test_rejects_html_body(self):
        self.assertIsNone(self._save(b"<!DOCTYPE html><title>Client Challenge</title>"))

    def test_rejects_truncated_pdf(self):
        self.assertIsNone(self._save(b"%PDF-1.4\nstub"))

    def test_accepts_real_pdf(self):
        self.assertIsNotNone(self._save(PDF_BYTES))


class TestConfiguredMirrors(unittest.TestCase):
    def test_env_overrides_the_default_list(self):
        with patch.object(sci_hub, "get_env", return_value=" https://a.example/, https://b.example "):
            self.assertEqual(
                sci_hub._configured_mirrors(), ["https://a.example", "https://b.example"]
            )

    def test_blank_env_falls_back_to_defaults(self):
        with patch.object(sci_hub, "get_env", return_value="   "):
            self.assertEqual(
                sci_hub._configured_mirrors(), [m.rstrip("/") for m in sci_hub.DEFAULT_MIRRORS]
            )


if __name__ == "__main__":
    unittest.main()
