import json
import os

# ----------- CONFIG -------------
JSONL_PATH = r'C:\Users\Dell\PycharmProjects\DT-AI-Assistant\data\capital_gains_chunks.jsonl'
LOG_FILE = r'C:\Users\Dell\PycharmProjects\DT-AI-Assistant\data\chunk_log.txt'
# --------------------------------


def load_chunks(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def log_issue(log, issue):
    print(issue)
    log.append(issue)


def check_chunks(chunks):
    log = []
    total = len(chunks)
    empty_count = 0
    page_issues = 0
    marker_issues = 0

    lengths = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "UNKNOWN")
        text = chunk.get("text", "").strip()
        start_page = chunk.get("start_page")
        end_page = chunk.get("end_page")

        # Length checks
        length = len(text)
        lengths.append(length)

        if length < 50:
            log_issue(log, f"⚠️  Chunk {chunk_id} is very short ({length} characters)")
            empty_count += 1

        # Page check
        if start_page is None or end_page is None:
            log_issue(log, f"❌  Missing page info in chunk {chunk_id}")
            page_issues += 1
        elif start_page > end_page:
            log_issue(log, f"❗ Invalid page range in chunk {chunk_id}: {start_page}–{end_page}")
            page_issues += 1

        # Check if [PAGE x] marker still exists
        if "[PAGE" in text:
            log_issue(log, f"📌  Chunk {chunk_id} contains leftover [PAGE x] marker")
            marker_issues += 1

    # Summary
    print("\n🔍 Summary:")
    print(f" - Total chunks: {total}")
    print(f" - Very short chunks (<50 chars): {empty_count}")
    print(f" - Page issues: {page_issues}")
    print(f" - Marker cleanup issues: {marker_issues}")
    print(f" - Avg length: {sum(lengths) // total} chars")
    print(f" - Max length: {max(lengths)} chars")

    # Write log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        for issue in log:
            f.write(issue + "\n")
    print(f"\n✅ Log written to {LOG_FILE}")


if __name__ == "__main__":
    if not os.path.exists(JSONL_PATH):
        print(f"❌ File not found: {JSONL_PATH}")
    else:
        chunks = load_chunks(JSONL_PATH)
        check_chunks(chunks)
