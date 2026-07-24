class CodeAndTextParser:
    def parse_text_code(self, raw_content: str, title: str = None, source_type: str = "text") -> dict:
        if not raw_content or not raw_content.strip():
            raise ValueError("Provided raw content is empty")

        cleaned_content = raw_content.strip()

        if not title:
            first_line = cleaned_content.split("\n")[0][:50]
            title = f"snippet: {first_line}..." if len(first_line) == 50 else f"snippet: {first_line}"

        return {
            "title": title,
            "raw_content": cleaned_content,
            "source_ref": None
        }