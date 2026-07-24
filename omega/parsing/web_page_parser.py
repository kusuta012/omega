import httpx
import trafilatura
import logging

logger = logging.getLogger("WebPageParser")

class WebPageParser:
    def __init__(self, timeout_seconds: int = 15):
        self.timeout_seconds = timeout_seconds
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    async def parse_url(self, url: str) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True, headers=self.headers) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise ValueError(f"HTTP error {e.response.status_code} while fetching {url}")
            except httpx.RequestError as e:
                raise ValueError(f"Network error while reaching {url} {str(e)}")

        html_content = response.text

        extracted_text = trafilatura.extract(
            html_content,
            include_links=True,
            include_images=False,
            output_format="txt"
        )

        if not extracted_text or len(extracted_text.strip()) == 0:
            raise ValueError(f"Could not extract meaningful text from {url}")

        metadata = trafilatura.extract_metadata(html_content)
        title = metadata.title if metadata and metadata.title else url

        return {
            "title": title,
            "raw_content": extracted_text,
            "source_ref": url
        }