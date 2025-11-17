from pathlib import Path
import shutil  # NEW

base = Path(r"/home/darshan/darshan/darshan/ocr_results_full")
files = list(base.rglob("result.mmd"))
print(f"Count: {len(files)}")
for f in files:
    print(f)

# NEW: Copy ALL result.mmd files into a new base directory, preserving subfolders
OUTPUT_BASE = Path(r"/home/darshan/darshan/darshan/_mmd_collected")
DRY_RUN = False
OVERWRITE = True

if not DRY_RUN:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

copied = 0
skipped = 0

for src in files:
    rel = src.relative_to(base)               # e.g., Company/Role - 1/result.mmd
    dest = OUTPUT_BASE / rel                  # keep company and role structure
    if dest.exists() and not OVERWRITE:
        print(f"SKIP (exists): {dest}")
        skipped += 1
        continue
    print(f"COPY: {src} -> {dest}")
    if not DRY_RUN:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    copied += 1

print(f"\nCopied: {copied}, Skipped: {skipped}, Output: {OUTPUT_BASE}")