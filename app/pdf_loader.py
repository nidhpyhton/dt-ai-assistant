import pdfplumber
import re
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter


#Extract and clean text from PDF
def extract_text_from_pdf(pdf_path):
    full_text = ""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = clean_text(page.extract_text())
            if text:
                full_text += f"\n\n[PAGE {i+1}]\n" + text
                pages.append((i+1, text))
    return full_text, pages

def clean_text(text):
    return text.replace("–", "-").replace("\xa0", " ").strip()


#Chunk the text extracted from the PDF

def chunk_text(text, chunk_size=1000, chunk_overlap=200):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

# Extract Page Range from the chunks
def extract_page_range_from_chunk(chunk_text):
    """Detects start and end page numbers from chunk using [PAGE x] markers."""
    page_matches = re.findall(r"\[PAGE (\d+)\]", chunk_text)
    if page_matches:
        page_numbers = [int(p) for p in page_matches]
        return min(page_numbers), max(page_numbers)
    return None, None

# save chunks to a jsonl file with meta data

def save_chunks_to_jsonl(chunks, output_path):
    """Saves chunks to a .jsonl file with metadata."""
    last_known_page = 1
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):

                start_page, end_page = extract_page_range_from_chunk(chunk)
                if start_page is None:
                    start_page = end_page = last_known_page
                last_known_page = end_page
                obj = {
                    "chunk_id": f"cg_{i:03}",
                    "section_title": "Unknown",
                    "text": chunk,
                    "start_page": start_page,
                    "end_page": end_page
                }
                f.write(json.dumps(obj) + "\n")


# Run the program

if __name__=="__main__":
    PDF_PATH =r"C:\Users\Dell\PycharmProjects\DT-AI-Assistant\data\capital_gains.pdf"
    OUTPUT_PATH = r"C:\Users\Dell\PycharmProjects\DT-AI-Assistant\data\capital_gains_chunks.jsonl"

    print("Extracting text from PDF...")
    full_text, pages = extract_text_from_pdf(PDF_PATH)

    print("Chunking extracted text...")
    chunks = chunk_text(full_text)

    print("Saving chunks to JSONL...")
    save_chunks_to_jsonl(chunks, OUTPUT_PATH)

    print("Process Complete!")
