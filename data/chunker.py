import re
def chunk_text(text, size=500, overlap=50):

    sections = re.split(r'(?=\n?\d+\.\s)', text)

    final_chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Step 2: If section is small, keep as is
        if len(section) <= size:
            final_chunks.append(section)
        else:
            # Step 3: Apply chunking within large sections
            start = 0
            while start < len(section):
                end = start + size
                chunk = section[start:end]

                final_chunks.append(chunk)
                start += size - overlap

    return final_chunks