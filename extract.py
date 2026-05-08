"""
PDF Extraction Script for Vulcan OmniPro 220 Welder Docs
=========================================================

Library choice: PyMuPDF (fitz) over pdfplumber
- pdfplumber is great for tables and clean text extraction, but weak on images —
  it can't reliably export embedded images with correct dimensions and color.
- PyMuPDF has direct access to the PDF's internal object stream, so it can
  extract raster images at native resolution, render pages to pixel maps,
  and give us precise bounding boxes for every text span and image on a page.
  For a manual full of diagrams and schematics, this is the right tool.

Chunking strategy: "semantic paragraph chunks with header inheritance"
- We don't split blindly by token count (that breaks mid-sentence mid-table).
- Instead we group the text blocks PyMuPDF gives us per-page, then split on
  blank-line boundaries to preserve natural paragraph/table units.
- Each chunk inherits the nearest preceding header (bold/large font) as its
  section label — this lets the RAG retriever know "this chunk is about duty cycles"
  without needing to re-read the whole page.
- Target: 300–800 characters per chunk. Too small = no context; too large = noise
  swamps the signal when doing vector similarity search.
- Overlap: none at this stage — we add overlap at RAG query time if needed.
"""

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
FILES_DIR = BASE_DIR / "files"
OUT_DIR = BASE_DIR / "extracted"
IMG_DIR = OUT_DIR / "images"

OUT_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

PDF_FILES = sorted(FILES_DIR.glob("*.pdf"))

# ── Tuning knobs ───────────────────────────────────────────────────────────────
MIN_CHUNK_CHARS = 80    # discard tiny fragments (page numbers, lone labels)
MAX_CHUNK_CHARS = 900   # split oversized blocks at sentence boundaries
MIN_IMAGE_BYTES = 2048  # skip tiny decorative images (bullets, borders)


# ── Helpers ────────────────────────────────────────────────────────────────────

def is_header(span: dict) -> bool:
    """Heuristic: bold or large font = section header."""
    flags = span.get("flags", 0)
    is_bold = bool(flags & 2**4)  # bit 4 = bold in PyMuPDF font flags
    is_large = span.get("size", 0) >= 11
    text = span.get("text", "").strip()
    # Must be short enough to be a header, not a body sentence
    return (is_bold or is_large) and len(text) < 120 and len(text) > 2


def split_to_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Split a block of text into chunks no larger than max_chars.
    Tries to break at sentence ends ('. ') before hard-splitting.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    while len(text) > max_chars:
        # Find the last sentence boundary within the window
        window = text[:max_chars]
        cut = window.rfind(". ")
        if cut == -1:
            cut = window.rfind(" ")  # fall back to word boundary
        if cut == -1:
            cut = max_chars          # hard cut as last resort
        else:
            cut += 1  # include the period
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()

    if text:
        chunks.append(text)
    return chunks


def extract_page_text_blocks(page: fitz.Page) -> list[dict]:
    """
    Extract structured text from a page.
    Returns list of {text, bbox, is_header, font_size} dicts.
    PyMuPDF's 'dict' mode gives us per-span font metadata we need for headers.
    """
    blocks_out = []
    raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

    for block in raw.get("blocks", []):
        if block.get("type") != 0:  # 0 = text block, 1 = image block
            continue

        block_text_parts = []
        block_is_header = False
        max_font_size = 0

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "")
                if t.strip():
                    block_text_parts.append(t)
                    if is_header(span):
                        block_is_header = True
                    max_font_size = max(max_font_size, span.get("size", 0))

        full_text = " ".join(block_text_parts).strip()
        # Normalise whitespace artifacts
        full_text = re.sub(r" {2,}", " ", full_text)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)

        if len(full_text) >= 3:  # skip single-char fragments
            blocks_out.append({
                "text": full_text,
                "bbox": block.get("bbox", []),
                "is_header": block_is_header,
                "font_size": round(max_font_size, 1),
            })

    return blocks_out


def extract_images_from_page(
    page: fitz.Page,
    doc: fitz.Document,
    source_stem: str,
    page_num: int,
    page_blocks: list[dict],
) -> list[dict]:
    """
    Extract all raster images from a page, save as PNG, and record surrounding
    text as caption context (the text block closest above the image bbox).
    Returns list of image metadata dicts.
    """
    images_out = []
    img_list = page.get_images(full=True)

    img_counter = 0
    for xref, smask, *_ in img_list:
        try:
            base_img = doc.extract_image(xref)
        except Exception:
            continue

        image_bytes = base_img.get("image", b"")
        if len(image_bytes) < MIN_IMAGE_BYTES:
            continue  # skip tiny decorative images

        img_counter += 1
        ext = base_img.get("ext", "png")
        filename = f"{source_stem}-p{page_num}-img{img_counter}.{ext}"
        img_path = IMG_DIR / filename

        img_path.write_bytes(image_bytes)

        # Find caption: text block whose bottom edge is closest above any image
        # on this page. We use the image's position on the page via page.get_image_rects.
        caption_text = ""
        try:
            rects = page.get_image_rects(xref)
            if rects:
                img_top = rects[0].y0  # top y-coordinate of image on page
                best_gap = float("inf")
                for blk in page_blocks:
                    blk_bottom = blk["bbox"][3] if blk["bbox"] else 0
                    gap = img_top - blk_bottom
                    if 0 < gap < best_gap:
                        best_gap = gap
                        caption_text = blk["text"]
        except Exception:
            pass

        images_out.append({
            "filename": filename,
            "path": str(img_path.relative_to(BASE_DIR)),
            "source": source_stem,
            "page": page_num,
            "width": base_img.get("width"),
            "height": base_img.get("height"),
            "caption_context": caption_text[:400],  # trim very long captions
        })

    return images_out


# ── Main extraction loop ───────────────────────────────────────────────────────

def extract_pdf(pdf_path: Path) -> tuple[list[dict], list[dict]]:
    """
    Process a single PDF. Returns (chunks, images).
    Each chunk: {id, source, page, chunk_index, section_header, text, char_count}
    Each image: {filename, path, source, page, width, height, caption_context}
    """
    source_stem = pdf_path.stem  # e.g. "owner-manual"
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*60}")

    doc = fitz.open(pdf_path)
    all_chunks: list[dict] = []
    all_images: list[dict] = []
    chunk_index = 0
    current_section = "Introduction"  # default section label

    for page_num in range(1, len(doc) + 1):
        page = doc[page_num - 1]
        print(f"  Page {page_num}/{len(doc)}", end="\r", flush=True)

        page_blocks = extract_page_text_blocks(page)

        # ── Track section headers as we walk down the page ──────────────────
        # Walk through blocks in reading order (PyMuPDF returns them top-to-bottom)
        for block in page_blocks:
            if block["is_header"]:
                current_section = block["text"].strip()

            raw_text = block["text"].strip()
            if len(raw_text) < MIN_CHUNK_CHARS:
                continue

            sub_chunks = split_to_chunks(raw_text)
            for sub in sub_chunks:
                if len(sub) < MIN_CHUNK_CHARS:
                    continue
                all_chunks.append({
                    "id": f"{source_stem}-p{page_num}-c{chunk_index}",
                    "source": source_stem,
                    "page": page_num,
                    "chunk_index": chunk_index,
                    "section_header": current_section,
                    "text": sub,
                    "char_count": len(sub),
                })
                chunk_index += 1

        # ── Extract images, referencing the text blocks we just parsed ───────
        page_images = extract_images_from_page(
            page, doc, source_stem, page_num, page_blocks
        )
        all_images.extend(page_images)

    print(f"  Done. {len(all_chunks)} chunks, {len(all_images)} images extracted.")
    doc.close()
    return all_chunks, all_images


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    if not PDF_FILES:
        print("No PDFs found in files/. Aborting.")
        sys.exit(1)

    all_chunks: list[dict] = []
    all_images: list[dict] = []

    for pdf_path in PDF_FILES:
        chunks, images = extract_pdf(pdf_path)
        all_chunks.extend(chunks)
        all_images.extend(images)

    # ── Save chunks.json ──────────────────────────────────────────────────────
    chunks_path = OUT_DIR / "chunks.json"
    with chunks_path.open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # ── Save images.json (separate index for the image assets) ───────────────
    images_path = OUT_DIR / "images.json"
    with images_path.open("w", encoding="utf-8") as f:
        json.dump(all_images, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Total text chunks : {len(all_chunks)}")
    print(f"  Total images      : {len(all_images)}")
    print()

    # Per-file breakdown
    for pdf_path in PDF_FILES:
        stem = pdf_path.stem
        n_chunks = sum(1 for c in all_chunks if c["source"] == stem)
        n_images = sum(1 for i in all_images if i["source"] == stem)
        print(f"  {pdf_path.name:<30}  {n_chunks:>4} chunks  {n_images:>3} images")

    print()
    print(f"  Output: {OUT_DIR.relative_to(BASE_DIR)}/")
    print(f"    chunks.json  ({len(all_chunks)} records)")
    print(f"    images.json  ({len(all_images)} records)")
    print(f"    images/      ({len(all_images)} PNG files)")

    # Show a sample chunk so you can see the structure
    if all_chunks:
        print("\n  Sample chunk:")
        sample = all_chunks[min(5, len(all_chunks) - 1)]
        for k, v in sample.items():
            val = str(v)[:80] + ("…" if len(str(v)) > 80 else "")
            print(f"    {k:<18}: {val}")

    if all_images:
        print("\n  Sample image record:")
        sample_img = all_images[0]
        for k, v in sample_img.items():
            val = str(v)[:80] + ("…" if len(str(v)) > 80 else "")
            print(f"    {k:<18}: {val}")


if __name__ == "__main__":
    main()
