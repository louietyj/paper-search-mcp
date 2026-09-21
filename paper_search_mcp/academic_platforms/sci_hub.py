"""Sci-Hub downloader integration.

Simple wrapper adapted from scihub.py for downloading PDFs via Sci-Hub.
"""
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from ..config import get_env
from ..utils import MIN_PDF_BYTES

logger = logging.getLogger(__name__)

# Mirrors go stale without warning: sci-hub.se stopped resolving in DNS and
# sci-hub.st answers 403, so a single hardcoded host fails every request.
DEFAULT_MIRRORS = (
    "https://sci-hub.ru",
    "https://sci-hub.st",
    "https://sci-hub.se",
)

# An isolated captcha clears on retry, but a rate-limited IP stays walled for far
# longer than any in-process backoff, so the budget stays small.
CAPTCHA_ATTEMPTS = 2
CAPTCHA_BACKOFF_SECONDS = 3

_NOT_IN_DATABASE_MARKERS = (
    "article not found",
    "not available through sci-hub",
    "not yet available in my database",
)

# A throttled client still gets the article page with its title rendered, only the
# PDF withheld. The widget ships on served pages too, so these classify a page that
# yielded no PDF link -- never one that did.
_CAPTCHA_MARKERS = (
    "/captcha/solution/",
    "g-recaptcha",
    "recaptcha/api.js",
)


class SciHubUnavailableError(RuntimeError):
    """Sci-Hub answered, but does not hold this paper. Retrying will not help."""


class SciHubRateLimitedError(RuntimeError):
    """Sci-Hub served a captcha instead of the paper. The client IP is throttled."""


def _configured_mirrors() -> List[str]:
    configured = get_env("SCIHUB_MIRRORS", "").strip()
    if configured:
        mirrors = [m.strip().rstrip("/") for m in configured.split(",") if m.strip()]
        if mirrors:
            return mirrors
    return [m.rstrip("/") for m in DEFAULT_MIRRORS]


class SciHubFetcher:
    """Simple Sci-Hub PDF downloader with mirror failover and captcha retry."""

    def __init__(self, base_url: Optional[str] = None, output_dir: str = "./downloads"):
        """`base_url` pins the fetcher to one mirror; unset tries all configured ones."""
        self.mirrors = [base_url.rstrip("/")] if base_url else _configured_mirrors()
        self.base_url = self.mirrors[0]
        self.last_failure_reason = ""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    def download_pdf(self, identifier: str) -> Optional[str]:
        """Download a PDF from Sci-Hub using a DOI, PMID, or URL.

        Walks the mirrors in order; sets `last_failure_reason` when it gives up.

        Args:
            identifier: DOI, PMID, or URL to the paper

        Returns:
            Path to saved PDF or None on failure
        """
        self.last_failure_reason = ""
        if not identifier.strip():
            self.last_failure_reason = "no identifier given"
            return None

        rate_limited = False
        rejected_bytes = False

        for mirror in self.mirrors:
            for attempt in range(CAPTCHA_ATTEMPTS):
                try:
                    pdf_url = self._get_direct_url(identifier, mirror)
                except SciHubUnavailableError as exc:
                    # Every mirror serves the same database, so no point trying another.
                    logger.warning("Sci-Hub does not hold %s: %s", identifier, exc)
                    self.last_failure_reason = "not in Sci-Hub's database"
                    return None
                except SciHubRateLimitedError as exc:
                    rate_limited = True
                    logger.warning("Sci-Hub captcha for %s: %s", identifier, exc)
                    if attempt < CAPTCHA_ATTEMPTS - 1:
                        time.sleep(CAPTCHA_BACKOFF_SECONDS)
                    continue
                except requests.RequestException as exc:
                    logger.warning("Sci-Hub mirror %s unreachable: %s", mirror, exc)
                    break

                if pdf_url:
                    saved = self._save_pdf(pdf_url, identifier)
                    if saved:
                        return saved
                    rejected_bytes = True

                break

        if rate_limited:
            self.last_failure_reason = (
                "Sci-Hub served a captcha on every mirror; this client's IP is rate-limited"
            )
        elif rejected_bytes:
            self.last_failure_reason = "resolved a PDF link but the response was not a valid PDF"
        else:
            self.last_failure_reason = "no PDF link found on any mirror"
        logger.error("Could not retrieve a PDF for %s: %s", identifier, self.last_failure_reason)
        return None

    def _save_pdf(self, pdf_url: str, identifier: str) -> Optional[str]:
        """Fetch a resolved PDF URL, rejecting anything that is not real PDF bytes."""
        try:
            response = self.session.get(pdf_url, verify=False, timeout=30)
        except requests.RequestException as exc:
            logger.warning("Failed to fetch resolved PDF %s: %s", pdf_url, exc)
            return None

        if response.status_code != 200:
            logger.warning("Failed to download PDF, status %s", response.status_code)
            return None

        content = response.content
        if not content.startswith(b"%PDF"):
            logger.warning(
                "Resolved Sci-Hub URL did not return PDF bytes: %s (first bytes=%r)",
                pdf_url, content[:16],
            )
            return None

        if len(content) < MIN_PDF_BYTES:
            logger.warning(
                "Resolved Sci-Hub URL returned an implausibly small PDF (%d bytes): %s",
                len(content), pdf_url,
            )
            return None

        file_path = self.output_dir / self._generate_filename(response, identifier)
        with open(file_path, "wb") as handle:
            handle.write(content)

        return str(file_path)

    def _get_direct_url(self, identifier: str, mirror: Optional[str] = None) -> Optional[str]:
        """Get the direct PDF URL from Sci-Hub.

        Raises:
            SciHubUnavailableError: Sci-Hub answered but does not hold the paper.
            SciHubRateLimitedError: the PDF was withheld behind a captcha.
            requests.RequestException: the mirror could not be reached.
        """
        # If it's already a direct PDF URL, return it
        if identifier.endswith('.pdf'):
            return identifier

        base_url = (mirror or self.base_url).rstrip("/")
        search_url = f"{base_url}/{identifier}"
        response = self.session.get(search_url, verify=False, timeout=20)

        if response.status_code != 200:
            return None

        effective_url = str(getattr(response, 'url', '') or search_url)
        body = response.text.lower()

        for marker in _NOT_IN_DATABASE_MARKERS:
            if marker in body:
                raise SciHubUnavailableError(marker)

        soup = BeautifulSoup(response.content, 'html.parser')

        # Look for embed tag with PDF (most common in modern Sci-Hub)
        embed = soup.find('embed', {'type': 'application/pdf'})
        if embed:
            src = embed.get('src') if hasattr(embed, 'get') else None
            if src and isinstance(src, str):
                return urljoin(effective_url, src)

        # Look for iframe with PDF (fallback)
        iframe = soup.find('iframe')
        if iframe:
            src = iframe.get('src') if hasattr(iframe, 'get') else None
            if src and isinstance(src, str):
                return urljoin(effective_url, src)

        # Look for download button with onclick
        for button in soup.find_all('button'):
            onclick = button.get('onclick', '') if hasattr(button, 'get') else ''
            if isinstance(onclick, str) and 'pdf' in onclick.lower():
                # Extract URL from onclick JavaScript
                url_match = re.search(r"location\.href='([^']+)'", onclick)
                if url_match:
                    return urljoin(effective_url, url_match.group(1))

        # Look for direct download links
        for link in soup.find_all('a'):
            href = link.get('href', '') if hasattr(link, 'get') else ''
            if isinstance(href, str) and href and ('pdf' in href.lower() or href.endswith('.pdf')):
                return urljoin(effective_url, href)

        # No link anywhere: a captcha widget means withheld, not absent.
        for marker in _CAPTCHA_MARKERS:
            if marker in body:
                raise SciHubRateLimitedError(f"captcha challenge from {base_url}")

        return None

    def _generate_filename(self, response: requests.Response, identifier: str) -> str:
        """Generate a unique filename for the PDF."""
        # Try to get filename from URL
        url_parts = response.url.split('/')
        if url_parts:
            name = url_parts[-1]
            # Remove view parameters
            name = re.sub(r'#view=(.+)', '', name)
            if name.endswith('.pdf'):
                # Generate hash for uniqueness
                pdf_hash = hashlib.md5(response.content).hexdigest()[:8]
                base_name = name[:-4]  # Remove .pdf
                return f"{pdf_hash}_{base_name}.pdf"

        # Fallback: use identifier
        clean_identifier = re.sub(r'[^\w\-_.]', '_', identifier)
        pdf_hash = hashlib.md5(response.content).hexdigest()[:8]
        return f"{pdf_hash}_{clean_identifier}.pdf"
