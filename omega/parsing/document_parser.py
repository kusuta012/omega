import io
import httpx
import pdfplumber
import logging
from pathlib import Path

logger = logging.getLogger("PDFParser")

class PDFDocumentParser:
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds

    async def parse_pdf(self, source_ref: str) -> dict:
        pdf_bytes = None

        if source_ref.startswith(("http://", "https://")):
            async with httpx.AsyncClient(timeout=self.timeout_seconds, follow_redirects=True) as client:
                try:
                    response = await client.get(source_ref)
                    response.raise_for_status()
                    pdf_bytes = response.content
                except Exception as error:
                    raise ValueError(f"failed to download remote PDF from {source_ref}: {str(error)}")

        else:
            file_path = Path(source_ref)
            if not file_path.exists():
                raise ValueError(f"Local PDF file not found: {source_ref}")
            pdf_bytes = file_path.read_bytes()

        extracted_pages = []
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page_index, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        extracted_pages.append(f"--- Page {page_index + 1} ---\n{text.strip()}")
        except Exception as error:
            raise ValueError(f"corrupt or unreadable pdf file {str(error)}")

        if not extracted_pages:
            raise ValueError(f"PDF contains no extractable text (it may be scanned/image-only) {source_ref}")

        full_text = "\n\n".join(extracted_pages)
        title = Path(source_ref).stem if not source_ref.startswith("http") else source_ref.split("/")[-1]

        return {
            "title": title,
            "raw_content": full_text,
            "source_ref": source_ref
        }