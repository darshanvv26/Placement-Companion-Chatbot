import os
import json

# Paths
CHUNKS_PATH = r"/home/darshan/darshan/darshan/chunks_by_filetype_new/chunks_all.json"
MMD_DIR = "_mmd_collected"
OUTPUT_PATH = "chunks_by_filetype_new/chunks_all_with_mmd.json"

def merge_mmd_results():
    # Load existing chunks_all.json
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"❌ chunks_all.json not found at {CHUNKS_PATH}")
    
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks_by_company = data.get("chunks", {})
    file_types = data.get("file_types", {})
    
    if not os.path.exists(MMD_DIR):
        print(f"⚠️ MMD directory not found: {MMD_DIR}")
        return
    
    mmd_added = 0
    company_added = set()

    # Traverse all .mmd files
    for root, _, files in os.walk(MMD_DIR):
        for name in files:
            if name.lower() != "result.mmd":
                continue

            # Extract company from folder structure
            rel_path = os.path.relpath(root, MMD_DIR)
            parts = rel_path.split(os.sep)
            company = parts[0] if parts else "Unknown"
            role = parts[1] if len(parts) > 1 else "General"

            mmd_path = os.path.join(root, name)
            try:
                with open(mmd_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
            except Exception as e:
                print(f"⚠️ Error reading {mmd_path}: {e}")
                continue

            # Create MMD chunk
            mmd_chunk = {
                "chunk_id": None,  # we'll fill below
                "section": "MMD",
                "confidence": 1.0,
                "content": content,
                "company": company,
                "role": role,
                "filename": name,
                "file_type": ".mmd"
            }

            # Add company if not present
            if company not in chunks_by_company:
                chunks_by_company[company] = []
                company_added.add(company)

            # Assign chunk_id sequentially for that company
            current_len = len(chunks_by_company[company])
            mmd_chunk["chunk_id"] = current_len + 1

            # Append to company list
            chunks_by_company[company].append(mmd_chunk)
            mmd_added += 1

    # Update metadata
    data["chunks"] = chunks_by_company
    data["total_chunks"] += mmd_added

    # Update file type counts
    if ".mmd" not in file_types:
        file_types[".mmd"] = {"files": 0, "chunks": 0}
    file_types[".mmd"]["files"] = len(company_added)
    file_types[".mmd"]["chunks"] += mmd_added
    data["file_types"] = file_types

    # Save updated file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Merged {mmd_added} .mmd chunks across {len(company_added)} companies.")
    print(f"💾 Saved updated JSON → {OUTPUT_PATH}")


if __name__ == "__main__":
    merge_mmd_results()
