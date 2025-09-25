import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import pdfplumber

# -------------------------
# Header Normalization Map
# -------------------------
HEADER_MAP = {
	"eligibility": "Eligibility Criteria",
	"eligibility criteria": "Eligibility Criteria",
	"criteria": "Eligibility Criteria",

	"job description": "Job Description",
	"about us": "Job Description",  # Some JDs start with this

	"skills": "Skill Set Requirements",
	"required skills and qualifications": "Skill Set Requirements",
	"preferred skills": "Skill Set Requirements",
	"skill set requirements": "Skill Set Requirements",

	"about the program": "Program Details",
	"key responsibilities": "Key Responsibilities",
	"who should apply": "Eligibility Criteria",
	"what we offer": "Benefits",
	"round details": "Selection Rounds",
	"profile & ctc": "Compensation",
	"stipend": "Compensation",
	"drive description": "Drive Details",
}


def normalize_header(header: str) -> str:
	"""Normalize headers to fixed relevant names using HEADER_MAP."""
	h = header.lower().strip()
	for key, value in HEADER_MAP.items():
		if key in h:
			return value
	return header  # fallback: keep original if no match


def extract_text_from_file(file_path: str) -> str:
	"""Extract text from txt, docx, pdf, png/jpg files."""
	ext = os.path.splitext(file_path)[-1].lower()

	if ext == ".txt":
		with open(file_path, "r", encoding="utf-8") as f:
			return f.read()

	elif ext == ".docx":
		doc = Document(file_path)
		return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

	elif ext in [".png", ".jpg", ".jpeg"]:
		image = Image.open(file_path)
		return pytesseract.image_to_string(image)

	elif ext == ".pdf":
		text = ""
		with pdfplumber.open(file_path) as pdf:
			for page in pdf.pages:
				text += page.extract_text() or ""
		if not text.strip():  # fallback OCR if scanned
			with pdfplumber.open(file_path) as pdf:
				for page in pdf.pages:
					pil_image = page.to_image().original
					text += pytesseract.image_to_string(pil_image)
		return text

	else:
		print(f"⚠️ Unsupported file type: {ext}")
		return ""


def split_into_sections(text: str, default_section="General"):
	"""Split raw text into normalized sections using known headers."""
	sections = {}
	current_section = default_section

	for line in text.splitlines():
		header_match = None
		for key in HEADER_MAP.keys():
			if key in line.lower():
				header_match = key
				break

		if header_match:
			normalized = normalize_header(header_match)
			current_section = normalized
			sections[current_section] = []
		else:
			sections.setdefault(current_section, []).append(line)

	for key in sections:
		sections[key] = "\n".join(sections[key]).strip()

	return sections


def chunk_sections(sections: dict):
	"""Split each section into smaller chunks with section metadata."""
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=400,
		chunk_overlap=50,
		separators=["\n\n", "\n", ".", " "]
	)

	all_chunks = []
	for section, content in sections.items():
		if not content.strip():
			continue
		chunks = text_splitter.split_text(content)
		for chunk in chunks:
			all_chunks.append({
				"section": section,
				"content": chunk
			})
	return all_chunks


def get_chunks(file_paths=None):
	"""Load multiple files and return structured text chunks grouped by file type."""
	if file_paths is None:
		file_paths = [
			r"E:\MiniProject_Placement_Companion\Placements_Data\Placements\Aditya Auto Products_MTech_2026\Info.txt",
			r"E:\MiniProject_Placement_Companion\Placements_Data\Placements\Aditya Auto Products_MTech_2026\JD Data Analyst.docx",
			r"E:\MiniProject_Placement_Companion\Placements_Data\Placements\Aditya Auto Products_MTech_2026\Aditya Auto Products_MTech_2026.png",
		]

	docx_chunks, txt_chunks, png_chunks, pdf_chunks = [], [], [], []

	for path in file_paths:
		raw_text = extract_text_from_file(path)
		if not raw_text.strip():
			continue

		default_section = os.path.basename(path)
		sections = split_into_sections(raw_text, default_section=default_section)
		chunks = chunk_sections(sections)

		ext = os.path.splitext(path)[-1].lower()
		if ext == ".docx":
			docx_chunks.extend(chunks)
		elif ext == ".txt":
			txt_chunks.extend(chunks)
		elif ext in [".png", ".jpg", ".jpeg"]:
			png_chunks.extend(chunks)
		elif ext == ".pdf":
			pdf_chunks.extend(chunks)

	return {
		"docx": docx_chunks,
		"txt": txt_chunks,
		"png": png_chunks,
		"pdf": pdf_chunks,
		"all": docx_chunks + txt_chunks + png_chunks + pdf_chunks
	}


if __name__ == "__main__":
	chunks_by_type = get_chunks()
	import json
	# print("DOCX:", json.dumps(chunks_by_type["docx"][:], indent=2, ensure_ascii=False))
	# print("TXT:", json.dumps(chunks_by_type["txt"][:2], indent=2, ensure_ascii=False))
	# print("PNG:", json.dumps(chunks_by_type["png"][:2], indent=2, ensure_ascii=False))
	print("all:", json.dumps(chunks_by_type["all"][:], indent=2, ensure_ascii=False))
