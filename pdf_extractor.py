#!/usr/bin/env python3
import fitz
import json
import re
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.cluster import KMeans

def extract_lines(doc):
    """
    1· Text extraction – ignore everything but selectable text
    Merge each span‐line into one record with avg font, bbox, flags, etc.
    """
    lines = []
    for pno in range(len(doc)):
        page = doc[pno]
        txt = page.get_text("dict")
        for block in txt["blocks"]:
            if block["type"] != 0:
                continue
            for row in block["lines"]:
                spans = row["spans"]
                if not spans:
                    continue
                # Merge spans
                # if spans are short segments, join with space; else concatenate
                if all(len(s["text"]) <= 5 for s in spans):
                    text = " ".join(s["text"] for s in spans).strip()
                else:
                    text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                sizes = [s["size"] for s in spans]
                avg_size = sum(sizes) / len(sizes)
                flags = 0
                for s in spans:
                    flags |= s["flags"]
                is_bold = bool(flags & 1)
                is_italic = bool(flags & 2)
                x0s = [s["bbox"][0] for s in spans]
                y0s = [s["bbox"][1] for s in spans]
                x1s = [s["bbox"][2] for s in spans]
                y1s = [s["bbox"][3] for s in spans]
                bbox = (min(x0s), min(y0s), max(x1s), max(y1s))
                lines.append({
                    "text": text,
                    "font_size": avg_size,
                    "is_bold": is_bold,
                    "is_italic": is_italic,
                    "page": pno,
                    "bbox": bbox,
                    "y0": bbox[1],
                    "y1": bbox[3],
                })
    return lines


def overlap_ratio(b1, b2):
    x0, y0, x1, y1 = b1
    u0, v0, u1, v1 = b2
    dx = min(x1, u1) - max(x0, u0)
    dy = min(y1, v1) - max(y0, v0)
    if dx <= 0 or dy <= 0:
        return 0.0
    inter = dx * dy
    area = (x1 - x0) * (y1 - y0)
    return inter / area if area > 0 else 0.0


def merge_similar_lines(lines):
    """
    Improved merge function to handle multi-line headings better
    """
    if not lines:
        return lines
    
    # Sort lines by page, then by vertical position
    lines = sorted(lines, key=lambda x: (x["page"], x["y0"]))
    
    merged = []
    i = 0
    
    while i < len(lines):
        current = lines[i].copy()  # Start with current line
        
        # Look ahead to find lines that should be merged
        j = i + 1
        while j < len(lines):
            next_line = lines[j]
            
            # Check if lines should be merged
            if should_merge_lines(current, next_line):
                # Merge the lines
                current = merge_two_lines(current, next_line)
                j += 1
            else:
                break
        
        merged.append(current)
        i = j if j > i + 1 else i + 1
    
    # Post-processing: try to catch missed merges
    merged = post_process_merges(merged)
    
    return merged


def post_process_merges(lines):
    """
    Additional pass to catch heading continuations that might have been missed
    """
    if len(lines) < 2:
        return lines
    
    final_merged = []
    i = 0
    
    while i < len(lines):
        current = lines[i]
        
        # Look for specific patterns that should be merged
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            
            # Special case: Numbered heading followed by single word on same page
            current_text = current["text"].strip()
            next_text = next_line["text"].strip()
            
            should_merge_post = False
            
            # Case 1: Line ends with "–" and next is single word/short phrase
            if (current_text.endswith("–") and 
                len(next_text.split()) <= 2 and
                current["page"] == next_line["page"] and
                abs(current["font_size"] - next_line["font_size"]) <= 2):
                should_merge_post = True
            
            # Case 2: Numbered section + single meaningful word
            elif (re.match(r'^\d+\.', current_text) and
                  len(next_text.split()) == 1 and
                  len(next_text) >= 3 and
                  next_text[0].isupper() and
                  current["page"] == next_line["page"] and
                  abs(current["font_size"] - next_line["font_size"]) <= 2):
                should_merge_post = True
                
            # Case 3: Any heading-like text + "Syllabus" specifically
            elif (next_text in ["Syllabus", "Overview", "Introduction", "Summary"] and
                  current["page"] == next_line["page"] and
                  abs(current["font_size"] - next_line["font_size"]) <= 2 and
                  len(current_text.split()) >= 3):
                should_merge_post = True
            
            if should_merge_post:
                print(f"POST-MERGE: '{current_text}' + '{next_text}'")
                merged_line = merge_two_lines(current, next_line)
                final_merged.append(merged_line)
                i += 2  # Skip both lines
                continue
        
        final_merged.append(current)
        i += 1
    
    return final_merged


def should_merge_lines(line1, line2):
    """
    Determine if two lines should be merged based on various criteria
    """
    # Must be on same page
    if line1["page"] != line2["page"]:
        return False
    
    # Calculate vertical gap
    vertical_gap = line2["y0"] - line1["y1"]
    
    # Calculate average font size and height
    avg_font_size = (line1["font_size"] + line2["font_size"]) / 2
    height1 = line1["y1"] - line1["y0"]
    height2 = line2["y1"] - line2["y0"]
    avg_height = (height1 + height2) / 2
    
    # Font size should be similar (within 15% tolerance - increased)
    font_size_diff = abs(line1["font_size"] - line2["font_size"])
    similar_font_size = font_size_diff <= 0.15 * avg_font_size
    
    # Check horizontal alignment (left edges should be close)
    horizontal_alignment_threshold = max(20, 0.1 * line1["bbox"][2])  # Further increased threshold
    left_aligned = abs(line1["bbox"][0] - line2["bbox"][0]) <= horizontal_alignment_threshold
    
    # Gap should be reasonable (not too large)
    reasonable_gap = 0 <= vertical_gap <= 3.0 * avg_height  # Further increased gap tolerance
    
    # Both should have similar formatting (bold/italic) - but be more lenient
    similar_formatting = (line1["is_bold"] == line2["is_bold"] and 
                         line1["is_italic"] == line2["is_italic"])
    
    # Allow some formatting difference if other conditions are strong
    lenient_formatting = (
        (line1["is_bold"] or line2["is_bold"]) and  # At least one is bold
        line1["is_italic"] == line2["is_italic"]    # Same italic status
    )
    
    # Check if this looks like a heading continuation
    text1 = line1["text"].strip()
    text2 = line2["text"].strip()
    
    # Debug print for the specific case
    if "Agile Tester" in text1 and "Syllabus" in text2:
        print(f"DEBUG: Checking merge for '{text1}' + '{text2}'")
        print(f"  - similar_font_size: {similar_font_size} (diff: {font_size_diff}, threshold: {0.15 * avg_font_size})")
        print(f"  - left_aligned: {left_aligned} (diff: {abs(line1['bbox'][0] - line2['bbox'][0])}, threshold: {horizontal_alignment_threshold})")
        print(f"  - reasonable_gap: {reasonable_gap} (gap: {vertical_gap}, threshold: {3.0 * avg_height})")
        print(f"  - similar_formatting: {similar_formatting}")
        print(f"  - lenient_formatting: {lenient_formatting}")
    
    # First line characteristics that suggest it might be start of heading
    looks_like_heading_start = (
        len(text1.split()) <= 20 and  # Increased word limit
        (not text1.endswith('.') or re.match(r'^\d+\.', text1)) and  # No period or numbered
        any(c.isupper() for c in text1) and  # Has some uppercase
        line1["font_size"] >= 8  # Reduced minimum font size
    )
    
    # Second line characteristics that suggest it's continuation
    looks_like_continuation = (
        len(text2.split()) <= 15 and  # Increased word limit for continuation
        not text2.startswith(('•', '-', '○', '▪')) and  # Not a bullet point
        not re.match(r'^\d+\.', text2) and  # Not a new numbered section
        any(c.isalpha() for c in text2) and  # Contains letters
        not text2.lower().startswith(('note:', 'example:', 'figure', 'table'))  # Not special content
    )
    
    # Special case: if first line ends with "–" or "-", it's likely continued
    ends_with_dash = text1.endswith(('–', '-'))
    
    # Special case: single word continuation (like "Syllabus")
    single_word_continuation = (
        len(text2.split()) <= 2 and  # 1-2 words
        len(text2) <= 25 and  # Not too long
        not text2.islower() and  # Not all lowercase
        any(c.isupper() for c in text2)  # Has uppercase letters
    )
    
    # Special case: Numbered section with continuation
    numbered_section_continuation = (
        re.match(r'^\d+\.', text1) and  # First line starts with number
        not re.match(r'^\d+\.', text2) and  # Second line doesn't start with number
        looks_like_continuation
    )
    
    # Combine all conditions - use lenient formatting if other conditions are met
    formatting_ok = similar_formatting or lenient_formatting
    
    should_merge = (
        similar_font_size and
        left_aligned and
        reasonable_gap and
        formatting_ok and
        (
            (looks_like_heading_start and looks_like_continuation) or
            ends_with_dash or
            single_word_continuation or
            numbered_section_continuation
        )
    )
    
    # Additional debug for the specific case
    if "Agile Tester" in text1 and "Syllabus" in text2:
        print(f"  - Final decision: {should_merge}")
    
    return should_merge


def merge_two_lines(line1, line2):
    """
    Merge two lines into one, combining their properties appropriately
    """
    merged = line1.copy()
    
    # Combine text with space
    merged["text"] = line1["text"].strip() + " " + line2["text"].strip()
    
    # Update bounding box to encompass both lines
    merged["bbox"] = (
        min(line1["bbox"][0], line2["bbox"][0]),  # min x0
        min(line1["bbox"][1], line2["bbox"][1]),  # min y0
        max(line1["bbox"][2], line2["bbox"][2]),  # max x1
        max(line1["bbox"][3], line2["bbox"][3])   # max y1
    )
    
    # Update y coordinates
    merged["y0"] = merged["bbox"][1]
    merged["y1"] = merged["bbox"][3]
    
    # Use average font size (weighted by text length)
    len1, len2 = len(line1["text"]), len(line2["text"])
    total_len = len1 + len2
    if total_len > 0:
        merged["font_size"] = (line1["font_size"] * len1 + line2["font_size"] * len2) / total_len
    
    # Preserve formatting if both lines have it
    merged["is_bold"] = line1["is_bold"] or line2["is_bold"]
    merged["is_italic"] = line1["is_italic"] or line2["is_italic"]
    
    return merged


def build_outline(lines, doc):
    # Apply the improved merging first
    lines = merge_similar_lines(lines)

    n_pages = len(doc)
    if not lines:
        return "", []

    # 2· Global font statistics (k‐means clustering)
    all_sizes = np.array([L["font_size"] for L in lines]).reshape(-1, 1)
    unique = np.unique(all_sizes)
    k = min(6, len(unique))
    if k > 1:
        km = KMeans(n_clusters=k, random_state=0).fit(all_sizes)
        labels = km.labels_
        cents = km.cluster_centers_.flatten()
    else:
        labels = np.zeros(len(lines), int)
        cents = unique
    order = np.argsort(-cents)
    cluster_rank = {int(old): r for r, old in enumerate(order)}
    freq = Counter(labels)
    body_cluster = freq.most_common(1)[0][0]
    body_mode_size = cents[body_cluster]
    text_count = Counter(L["text"] for L in lines)
    # Compute typical heading left-x position (based on numeric headings)
    heading_xs = [
        L["bbox"][0]
        for L in lines
        if re.match(r'^\d+(\.\d+)*', L["text"])
        and L["font_size"] >= 0.95 * body_mode_size
    ]
    typical_heading_left_x = np.median(heading_xs) if heading_xs else None


    # 3· Title detection (page 0 only) – join multiple top candidates
    # 3· Title detection (page 0 only) – only maximum font size in top 50%
    page0_lines = [L for L in lines if L["page"] == 0]
    title = ""
    
    if page0_lines:
        # Find maximum font size on page 0
        max_font_size = max(L["font_size"] for L in page0_lines)
        
        # Get lines with maximum font size (within small tolerance)
        max_font_lines = [L for L in page0_lines if abs(L["font_size"] - max_font_size) < 0.5]
        
        # Check if any max font lines are in top 50% of page
        page_height = doc[0].rect.height
        top_half_lines = [L for L in max_font_lines if L["y0"] < 0.5 * page_height]
        
        if top_half_lines:
            # Use the topmost line as title
            topmost_line = min(top_half_lines, key=lambda L: L["y0"])
            title = topmost_line["text"].strip()
        else:
            # Max font lines are not in top 50%, so they become headings
            title = ""



    # 4· Heading-candidate pool
    per_page = defaultdict(list)
    for idx, L in enumerate(lines): per_page[L["page"]].append((idx, L))
    gaps = [0.0]*len(lines)
    max_gap = {}
    for p, items in per_page.items():
        items.sort(key=lambda t: lines[t[0]]["y0"])
        prev_y1 = None
        page_gaps = []
        for idx, L in items:
            g = L["y0"] - prev_y1 if prev_y1 is not None else 9999.0
            gaps[idx] = g
            page_gaps.append(g)
            prev_y1 = L["y1"]
        max_gap[p] = max(page_gaps) if page_gaps else 0.0

    candidates = []
    for i, L in enumerate(lines):
        is_section_numbered = bool(re.match(r'^\d+(\.\d+)*', L["text"]))
        if (
            L["font_size"] >= 1.15 * body_mode_size
            or (L["is_bold"] and gaps[i] >= 0.6 * max_gap[L["page"]])
            or (gaps[i] >= 0.9 * max_gap[L["page"]])  # ADD: isolated large gap
        ):
            candidates.append(i)


    # 5· Hard noise filter
    filtered = []
    for i in candidates:
        t = lines[i]["text"].strip()

    # 🚫 skip purely numeric headings like "0.1" or "3.2.1"
        if re.fullmatch(r'\d+(?:\.\d+)*', t):
            continue

    # now all your other noise filters
        if len(t) > 110: 
            continue
        if t.endswith(":"):
            continue
        if re.search(r"\d{3,}", t) and re.search(r"\b[A-Z]{2}\b", t):
            continue
        if re.match(r"^\d+(?:[.)])$", t):
            continue
        if len(t) > 120 or t.count(" ") > 20:
            continue

        pages_with = {L2["page"] for L2 in lines if L2["text"] == t}
        if len(pages_with) / n_pages > 0.25:
            continue

        filtered.append(i)


    # 6· Soft scoring
    scored = []
    for i in filtered:
        L = lines[i]
        rank = cluster_rank[labels[i]]
        b = 1 if L["is_bold"] else 0
        ng = gaps[i]/max_gap[L["page"]] if max_gap[L["page"]]>0 else 0
        letters = sum(c.isalpha() for c in L["text"])
        ac = 1 if letters and sum(c.isupper() for c in L["text"]) / letters >= 0.8 else 0
        section_bonus = 1.5 if re.match(r'^\d+(\.\d+)+', L["text"].strip()) else 0

        score = (
            rank + 0.4 * b + 0.2 * ng + 0.2 * ac + 0.6 * section_bonus + (2 - rank) * 0.4
        )

        scored.append((i, score))
    

    def infer_level_from_numbered(text: str) -> str:
        m = re.match(r'^(\d+(?:\.\d+)*)', text.strip())
        if not m:
            return None
        parts = m.group(1).split('.')
        if len(parts) == 1:
            return "H1"
        elif len(parts) == 2:
            return "H2"
        else:
            return "H3"
    outline = []
    for idx, _score in scored:
        text = lines[idx]["text"]
    # first try to infer from "2.3", "3.1.4", etc.
        lvl = infer_level_from_numbered(text)
    # fallback to H1 for anything else
        if lvl is None:
            lvl = "H1"
        outline.append({
            "level": lvl,
            "text":  text,
            "page":  lines[idx]["page"]
        })

    # remove duplicates
    unique = []
    seen = set()
    for h in outline:
        key = (h['text'], h['page'])
        if key not in seen:
            unique.append(h)
            seen.add(key)
    outline = unique
    for h in outline:
        match = re.match(r"^(\d+(?:\.\d+)*)(\s|:|$)", h["text"])
        if match:
            h["section"] = match.group(1)

    # 8· Image guard
    safe = []
    for h in outline:
        match = next((L for L in lines if L['text']==h['text'] and L['page']==h['page']), None)
        if not match: continue
        bbox = match['bbox']
        reject = False
        for img in doc[h['page']].get_images(full=True):
            _, x0,y0,x1,y1 = img[:5]
            if overlap_ratio(bbox, (x0,y0,x1,y1))>0.8:
                reject = True
                break
        if not reject:
            safe.append(h)
    outline = safe

    # 9· Fail-safe: If no title and no outline, promote max font line to H1
    if not title and not outline and lines:
        # Find the line with maximum font size
        max_font_line = max(lines, key=lambda L: L["font_size"])
        
        # Check if it's not inside a shape/image/table
        bbox = max_font_line['bbox']
        reject = False
        
        # Check for image overlap
        for img in doc[max_font_line['page']].get_images(full=True):
            _, x0,y0,x1,y1 = img[:5]
            if overlap_ratio(bbox, (x0,y0,x1,y1)) > 0.8:
                reject = True
                break
        
        if not reject:
            outline = [{
                "level": "H1",
                "text": max_font_line["text"],
                "page": max_font_line["page"]
            }]

    # keep even if <3
    return title, outline


def process_pdf(path, out_dir):
    doc = fitz.open(path)
    lines = extract_lines(doc)
    title, outline = build_outline(lines, doc)
    result = {"title": title, "outline": outline}
    out_path = out_dir / f"{path.stem}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[✓] {path.name} → {out_path.name}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input", type=Path, default=Path("input"))
    parser.add_argument("-o","--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(args.input.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in", args.input)
        return
    for pdf in pdfs:
        try:
            process_pdf(pdf, args.output)
        except Exception as e:
            print(f"[!] Error processing {pdf.name}: {e}")

if __name__ == "_main_":
    main()