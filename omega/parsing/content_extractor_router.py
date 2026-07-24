import logging
from omega.parsing.web_page_parser import WebPageParser
from omega.parsing.document_parser import PDFDocumentParser
from omega.parsing.code_text_parser import CodeAndTextParser

logger = logging.getLogger("ContentExtractorRouter")

class ContentExtractorRouter:
    def __init__(self):
        self.web_parser = WebPageParser()
        self.pdf_parser = PDFDocumentParser()
        self.text_parser = CodeAndTextParser()

    async def extract_content(self, source_type: str, source_ref: str | None = None, raw_content: str | None = None, title: str | None = None) -> dict:
        logger.info(f"routing extraction for source type='{source_type}' | ref='{source_ref}'")

        if source_type == "url":
            if not source_ref:
                raise ValueError("URL source requires a valid 'source_ref'")
            parsed_result = await self.web_parser.parse_url(source_ref)
            if title:
                parsed_result["title"] = title
            return parsed_result
        
        elif source_type == "pdf":
            if not source_ref:
                raise ValueError("PDF source requires a valid 'source_ref'")
            parsed_result = await self.pdf_parser.parse_pdf(source_ref)
            if title:
                parsed_result["title"] = title
            return parsed_result

        elif source_type in ["text", "code"]:
            if not raw_content:
                raise ValueError(f"'{source_type}' source requires 'raw_content'")
            return self.text_parser.parse_text_code(
                raw_content=raw_content,
                title=title,
                source_type=source_type
            )
        
        else:
            raise ValueError(f"Unsupported source_type {source_type}")