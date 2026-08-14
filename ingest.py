from pathlib import Path
import sys

from pypdf import PdfReader


# This is the folder where you will place study PDFs.
DOCUMENTS_DIR = Path("documents")


def extract_pdf_text(pdf_path: Path) -> str:
    """Read a PDF and return the text from all of its pages."""
    reader = PdfReader(pdf_path)
    print(f"Number of pages: {len(reader.pages)}")

    # Some PDF pages contain no selectable text, so use an empty string as a
    # safe fallback before joining all page text into one document string.
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def main() -> None:
    # Require an explicit filename so you always know which document is tested.
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <pdf-filename>")
        print("Example: python ingest.py notes.pdf")
        sys.exit(1)

    # Path.name prevents a command-line path from escaping documents/.
    pdf_path = DOCUMENTS_DIR / Path(sys.argv[1]).name

    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    if pdf_path.suffix.lower() != ".pdf":
        print(f"Error: expected a .pdf file, got: {pdf_path.name}")
        sys.exit(1)

    try:
        text = extract_pdf_text(pdf_path)
    except Exception as error:
        print(f"Error: could not read '{pdf_path.name}': {error}")
        sys.exit(1)

    if not text.strip():
        print("No extractable text was found. This may be a scanned/image-only PDF.")
        return

    preview_length = 500
    print(f"\nExtracted text preview (first {preview_length} characters):\n")
    print(text[:preview_length])


if __name__ == "__main__":
    main()
