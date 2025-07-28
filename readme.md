# PDF Outline Extractor

### Adobe Hackathon Round 1A – Understand Your Document 🧠📄

## 📌 Problem Statement

You are provided with PDF files and are tasked with extracting a **structured document outline** from each:

- Document **Title**
- Section Headings: **H1**, **H2**, **H3**
- Page numbers

The extracted data should follow a strict hierarchical **JSON schema** and serve as the basis for semantic document understanding.

---

## 🚀 Approach

This solution performs rule-based heading detection using a combination of **layout analysis**, **font size clustering**, **text formatting**, and **heuristic merging**.

### Key Steps:

1. **Text Extraction**

   - Extract all selectable text using `PyMuPDF` (`fitz`), converting spans on each page into unified line entries with average font size and bounding boxes.

2. **Line Merging & Cleanup**

   - Merge adjacent lines likely forming a single heading (e.g., `3. Overview` + `Syllabus`) based on formatting, position, and gap thresholds.
   - Handles common heading continuation patterns using `should_merge_lines` and `post_process_merges`.

3. **Font Size Clustering**

   - Cluster font sizes using `KMeans` to distinguish heading candidates from body text.
   - Identify the body text cluster and treat outlier font sizes as potential headings.

4. **Title Detection**

   - Detect title on **page 0** as the line with the largest font in the **top 50%** of the page, unless it gets filtered out due to overlap or format.

5. **Heading Detection**

   - Score heading candidates based on:
     - Font size rank
     - Bold formatting
     - Vertical gap
     - Uppercase ratio
     - Section-number bonus (e.g., `2.1.3`)
   - Infer heading levels (H1, H2, H3) from numbering depth.

6. **Noise & Duplication Filter**

   - Remove lines that:
     - Appear too often across pages
     - Are pure numbers or codes
     - Exceed length or complexity thresholds
   - Eliminate duplicate headings across pages.

7. **Image Overlap Guard**

   - Remove text that overlaps significantly with detected images to avoid false positives.

8. **Fail-safe Heuristic**

   - If no heading or title is detected, fallback to using the max-font line as a `H1` heading.

---

## 📚 Libraries Used

- [`PyMuPDF`](https://pymupdf.readthedocs.io/en/latest/) (`fitz`) – for PDF parsing and text layout extraction
- [`scikit-learn`](https://scikit-learn.org/) – for KMeans clustering of font sizes
- `numpy`, `collections`, `re` – for numerical/statistical analysis and text filtering

---

## 🛠️ Build & Run Instructions

> 📁 Expected folder structure:

```
Challenge_1a/
├── sample_dataset/
│   ├── pdfs/                      # Place all input PDFs here
│   ├── outputs/                   # Extracted JSONs will be saved here
│   └── schema/output_schema.json # Provided JSON schema
├── scripts/
│   └── process_pdfs.py           # Main executable script (this file)
├── Dockerfile                    # Docker support for offline testing
```

### ✅ Run Locally

Ensure Python 3.9+ is installed, and run the following from the root directory:

```bash
pip install -r requirements.txt
python3 scripts/process_pdfs.py -i sample_dataset/pdfs -o sample_dataset/outputs
```

### 🐳 Docker Execution

Build and run the project in an isolated Docker environment:

```bash
# From root of repo
docker build -t adobe-demo-app .

docker run --rm \
  -v "$(pwd)/sample_dataset/pdfs:/app/input:ro" \
  -v "$(pwd)/sample_dataset/outputs:/app/output" \
  adobe-demo-app
```

---

## 📄 Output Format

Each input PDF generates a corresponding JSON with this schema:

```json
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "1. Section Title", "page": 2 },
    { "level": "H2", "text": "1.1 Subsection", "page": 3 },
    ...
  ]
}
```

---

## ✅ Features Summary

| Feature                        | Description                                              |
| ------------------------------ | -------------------------------------------------------- |
| ✅ Font-based Heading Detection | Clusters font sizes to infer heading candidates          |
| ✅ Line Merging Heuristics      | Combines multi-line headings with custom rules           |
| ✅ Title Detection              | Picks the top large-font line from Page 0                |
| ✅ Outline Level Inference      | Uses numbering to infer H1–H3 levels                     |
| ✅ Robust Filters               | Filters repeated, noisy, or image-overlapped text        |
| ✅ Docker Compatible            | Fully portable and self-contained for offline evaluation |

---

## 👨‍💻 Notes

- Designed for documents with English headings and numeric structures like `1.`, `2.3`, `3.1.4`.
- Not ML-based, but extensible to include classifier-based heading detection if needed.

---

