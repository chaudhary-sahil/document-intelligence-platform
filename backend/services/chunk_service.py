"""Step 2: text -> overlapping chunks that remember their page."""
from backend.config import CHUNK_OVERLAP, CHUNK_SIZE


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into ~chunk_size pieces, preferring sentence/word boundaries.

    Fixed-size slicing (your first version) cuts words in half, which hurts
    embedding quality. Here each chunk ends at the last sentence end ('. ') or,
    failing that, the last space in the second half of the window, and the
    next chunk starts `overlap` characters earlier so a sentence that straddles
    a boundary still appears whole in at least one chunk.
    """
    if overlap >= chunk_size // 2:
        raise ValueError("overlap must be less than half of chunk_size")

    text = " ".join(text.split())  # collapse newlines/tabs/multiple spaces
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            window_start = start + chunk_size // 2
            cut = text.rfind(". ", window_start, end)
            if cut != -1:
                end = cut + 1  # keep the full stop
            else:
                cut = text.rfind(" ", window_start, end)
                if cut != -1:
                    end = cut

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break

        start = end - overlap
        # Don't start the next chunk in the middle of a word.
        if start > 0 and text[start - 1] != " ":
            next_space = text.find(" ", start, end)
            if next_space != -1:
                start = next_space + 1

    return chunks


def chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunk each page separately so every chunk has exactly one page number."""
    chunks = []
    index = 0
    for page in pages:
        for piece in split_text(page["text"], chunk_size, overlap):
            chunks.append(
                {"text": piece, "page": page["page"], "chunk_index": index}
            )
            index += 1
    return chunks
