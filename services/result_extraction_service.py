"""
Result Extraction Service
- Parses result PDFs using hybrid deterministic+LLM approach
- NIT KKR result PDFs show only SGPA per student (not per-subject marks)
- Indexes result content in ChromaDB for RAG queries
- Supports natural language queries like "mera result kya hai semester 3 mein"
"""
import warnings
warnings.filterwarnings("ignore")
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from config import GROQ_API_KEY
from datetime import datetime
import re, json

RESULT_VECTORSTORE_DIR = "vectorstore_results"
ANNOUNCEMENT_VECTORSTORE_DIR = "vectorstore_announcements"

_embeddings = None
_llm = None
_result_vs = None
_announcement_vs = None
_pdf_cache = {}  # file_path -> (mtime, documents)
_extraction_cache = {}  # (file_path, roll_number, student_name) -> result_dict

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(api_key=GROQ_API_KEY, model="openai/gpt-oss-120b")
    return _llm

def get_result_vectorstore():
    global _result_vs
    if _result_vs is None:
        _result_vs = Chroma(
            persist_directory=RESULT_VECTORSTORE_DIR,
            embedding_function=get_embeddings()
        )
    return _result_vs

def get_announcement_vectorstore():
    global _announcement_vs
    if _announcement_vs is None:
        _announcement_vs = Chroma(
            persist_directory=ANNOUNCEMENT_VECTORSTORE_DIR,
            embedding_function=get_embeddings()
        )
    return _announcement_vs

def get_cached_pdf_documents(file_path: str):
    """Cache PDF loaded documents in RAM based on file modification time."""
    if not os.path.exists(file_path):
        return []
    mtime = os.path.getmtime(file_path)
    if file_path in _pdf_cache:
        cached_mtime, docs = _pdf_cache[file_path]
        if cached_mtime == mtime:
            return docs
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    _pdf_cache[file_path] = (mtime, docs)
    return docs

import os, csv

def parse_excel_or_csv(file_path: str) -> list:
    """Parse .xlsx, .xls, or .csv file into a standardized list of dict rows."""
    ext = os.path.splitext(file_path)[1].lower()
    rows = []

    if ext in ['.xlsx', '.xls']:
        import openpyxl
        wb = openpyxl.load_workbook(file_path, data_only=True)
        sheet = wb.active
        headers = []
        for i, row in enumerate(sheet.iter_rows(values_only=True)):
            if i == 0:
                headers = [str(cell).strip().lower() if cell is not None else "" for cell in row]
                continue
            if not any(row):
                continue
            row_dict = {}
            for h, cell in zip(headers, row):
                row_dict[h] = str(cell).strip() if cell is not None else ""
            rows.append(row_dict)
    elif ext == '.csv':
        with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            reader = csv.DictReader(f)
            for r in reader:
                clean_r = {str(k).strip().lower(): str(v).strip() for k, v in r.items() if k}
                rows.append(clean_r)
    return rows

def _extract_field(row: dict, possible_keys: list, default="") -> str:
    for k in possible_keys:
        for rk in row.keys():
            if k in rk:
                return row[rk]
    return default

def extract_from_excel_or_csv(file_path: str, roll_number: str = None, student_name: str = None) -> dict:
    """Extract student result from an Excel or CSV file."""
    rows = parse_excel_or_csv(file_path)
    roll_keys = ["roll", "rollno", "roll_no", "roll_number", "id"]
    name_keys = ["name", "student_name", "studentname", "full_name"]
    sgpa_keys = ["sgpa", "gpa", "cgpa", "marks"]
    status_keys = ["status", "result", "pass_fail"]
    remark_keys = ["reappear", "remarks", "detail"]

    search_roll = str(roll_number).strip().lower() if roll_number else None
    search_name = str(student_name).strip().lower() if student_name else None

    for r in rows:
        r_roll = _extract_field(r, roll_keys)
        r_name = _extract_field(r, name_keys)

        match = False
        if search_roll and search_roll in r_roll.lower():
            match = True
        elif search_name and search_name in r_name.lower():
            match = True

        if match:
            sgpa = _extract_field(r, sgpa_keys, "N/A")
            status = _extract_field(r, status_keys, "Pass")
            remarks = _extract_field(r, remark_keys, "")

            return {
                "found": True,
                "student_name": r_name or student_name or "Student",
                "roll_number": r_roll or roll_number,
                "sgpa": sgpa,
                "result_status": status,
                "remarks": remarks,
                "subjects": []
            }

    return {"found": False, "message": "Student not found in Excel/CSV file."}

def index_result_excel_or_csv(file_path: str, degree: str, branch: str, semester: int, year: int, result_type: str, result_id: int):
    """Index Excel/CSV result rows as ChromaDB Document chunks."""
    from langchain_core.documents import Document
    rows = parse_excel_or_csv(file_path)
    if not rows:
        return

    docs = []
    roll_keys = ["roll", "rollno", "roll_no", "roll_number", "id"]
    name_keys = ["name", "student_name", "studentname", "full_name"]
    sgpa_keys = ["sgpa", "gpa", "cgpa", "marks"]
    status_keys = ["status", "result", "pass_fail"]
    remark_keys = ["reappear", "remarks", "detail"]

    for r in rows:
        r_roll = _extract_field(r, roll_keys)
        r_name = _extract_field(r, name_keys)
        sgpa = _extract_field(r, sgpa_keys, "N/A")
        status = _extract_field(r, status_keys, "Pass")
        remarks = _extract_field(r, remark_keys, "")

        text = f"Student: {r_name}, Roll Number: {r_roll}, SGPA: {sgpa}, Status: {status}, Degree: {degree}, Branch: {branch}, Semester: {semester}, Year: {year}, Type: {result_type}"
        if remarks:
            text += f", Remarks: {remarks}"

        doc = Document(
            page_content=text,
            metadata={
                "type": "result",
                "degree": degree,
                "branch": branch,
                "semester": semester,
                "year": year,
                "result_type": result_type,
                "result_id": result_id,
                "roll_number": r_roll,
                "student_name": r_name,
                "indexed_at": datetime.now().strftime("%Y-%m-%d"),
                "source_label": f"{degree} {branch} Sem {semester} {result_type} {year}"
            }
        )
        docs.append(doc)

    vs = get_result_vectorstore()
    vs.add_documents(docs)
    print(f"✅ Indexed {len(docs)} Excel/CSV student rows for result: {degree} {branch} Sem {semester} {year}")

def index_result_pdf(file_path: str, degree: str, branch: str, semester: int, year: int, result_type: str, result_id: int):
    """Index result PDF, Excel, or CSV content in vectorstore for RAG queries."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.xlsx', '.xls', '.csv']:
        return index_result_excel_or_csv(file_path, degree, branch, semester, year, result_type, result_id)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata.update({
            "type": "result",
            "degree": degree,
            "branch": branch,
            "semester": semester,
            "year": year,
            "result_type": result_type,
            "result_id": result_id,
            "indexed_at": datetime.now().strftime("%Y-%m-%d"),
            "source_label": f"{degree} {branch} Sem {semester} {result_type} {year}"
        })

    vs = get_result_vectorstore()
    vs.add_documents(chunks)
    print(f"✅ Indexed {len(chunks)} chunks for result: {degree} {branch} Sem {semester} {year} ({result_type})")

def index_announcement_pdf(file_path: str, title: str, ann_id: int):
    """Index announcement PDF content in vectorstore."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata.update({
            "type": "announcement",
            "title": title,
            "ann_id": ann_id,
            "indexed_at": datetime.now().strftime("%Y-%m-%d"),
        })

    vs = get_announcement_vectorstore()
    vs.add_documents(chunks)
    print(f"✅ Indexed announcement: {title}")



# ─────────────────────────────────────────────────────────────
# NIT KKR Subject Code → Name Map
# ─────────────────────────────────────────────────────────────
SUBJECT_NAME_MAP = {
    # Sem 3 CSE
    "MAIC-201": "Discrete Mathematics",
    "MARC-201": "Discrete Mathematics",
    "CSPC-201": "Computer Programming",
    "CSPC-203": "Data Structures",
    "CSPC-205": "Object-Oriented Programming",
    "CSPC-207": "Software Engineering",
    "CSPC-209": "IoT Programming",
    # Sem 1 CSE/ECE/IIoT
    "CSPC-101": "Introduction to Computing",
    "CSPC-103": "Programming Fundamentals",
    "CSPC-105": "Digital Logic Design",
    "MATH-101": "Mathematics I",
    "PHYS-101": "Engineering Physics",
    "CHEM-101": "Engineering Chemistry",
    # Sem 4 Re-appear
    "CSPC-401": "Theory of Computation",
    "CSPC-403": "Computer Organisation",
    "CSPC-405": "Operating Systems",
    "CSPC-407": "DBMS",
    "CSPC-409": "Computer Networks",
    # ECE
    "ECPC-201": "Electronic Devices & Circuits",
    "ECPC-203": "Signals & Systems",
    "ECPC-205": "Digital Electronics",
}


def _parse_sgpa_value(raw: str) -> str:
    """Convert '8 2000' or '8.2000' to '8.2000'."""
    raw = raw.strip()
    # Handle "8 2000" (space instead of decimal point — OCR artifact)
    m = re.match(r'^(\d{1,2})\s+(\d{4})$', raw)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    # Handle "8.2000" directly
    m2 = re.match(r'^(\d{1,2}\.\d{1,4})$', raw)
    if m2:
        try:
            val = float(m2.group(1))
            return f"{val:.4f}"
        except:
            pass
    return raw


def _deterministic_extract_page(page_text: str, search_term: str):
    """
    Parse a single PDF page to find a student's SGPA and reappear info.
    
    NIT KKR PDFs use multi-column layout. PyPDF extracts column-by-column, so:
    - Column 1: Sr.No + Roll No rows (like "17 124102014 Sanj")
    - Column 2: Full names
    - Column 3: Father names / Re. entries
    - Column 4: SGPA values (one per student, in same order as Sr.No column)
    
    Algorithm:
    1. Find all (sr_no, roll_no) pairs on this page → ordered student list
    2. Find all SGPA-like values on this page → ordered SGPA list
    3. Match by index: student[i] → sgpa[i]
    4. Find Re. entries for this specific student
    """
    lines = page_text.split('\n')
    search_lower = search_term.lower()
    
    # ── Step 1: Extract all student rows (Sr.No + Roll) ──────────────
    # Pattern: "17 124102014" or "17 124102014 Sanj" at start of line
    student_rows = []  # list of (line_idx, sr_no, roll_no)
    for idx, line in enumerate(lines):
        m = re.match(r'^\s*(\d{1,3})\s+(\d{9,12})', line.strip())
        if m:
            sr_no = int(m.group(1))
            roll_no = m.group(2)
            student_rows.append((idx, sr_no, roll_no))
    
    # ── Step 2: Find target student ──────────────────────────────────
    target_row = None
    target_pos = None  # index in student_rows list
    for i, (idx, sr_no, roll_no) in enumerate(student_rows):
        if search_lower in roll_no.lower():
            target_row = (idx, sr_no, roll_no)
            target_pos = i
            break
    
    if target_row is None:
        return None
    
    target_line_idx, target_sr_no, found_roll = target_row
    
    # ── Step 3: Extract all SGPA-like values from the page ───────────
    # SGPA patterns: "8.2000", "8 2000", "10 0000", "I 4000" (I=Incomplete), "R,..." (result late)
    sgpa_values = []  # list of (line_idx, raw_value, normalized)
    re_entries = []   # list of (line_idx, subject_code, raw_line)
    
    sgpa_re = re.compile(r'^\s*(\d{1,2}[\s.]\d{4})\s*$')
    ten_re = re.compile(r'^\s*(10[\s.]\d{4})\s*$')
    incomplete_re = re.compile(r'^\s*I[\s.]\d{4}\s*$')
    reappear_re = re.compile(r'Re\.?\s+([A-Z]{2,6}-\d{3}[A-Z]?)', re.IGNORECASE)
    
    for idx, line in enumerate(lines):
        ls = line.strip()
        # SGPA value line
        if sgpa_re.match(ls) or ten_re.match(ls):
            m = re.match(r'(\d{1,2})[\s.](\d{4})', ls)
            if m:
                normalized = f"{m.group(1)}.{m.group(2)}"
                try:
                    val = float(normalized)
                    if 0.5 <= val <= 10.0:
                        sgpa_values.append((idx, ls, normalized))
                except:
                    pass
        elif '.' in ls:
            # Handle "8.2000" format
            m = re.match(r'^(\d{1,2}\.\d{4})$', ls)
            if m:
                try:
                    val = float(m.group(1))
                    if 0.5 <= val <= 10.0:
                        sgpa_values.append((idx, ls, f"{val:.4f}"))
                except:
                    pass
        # Incomplete / special
        elif incomplete_re.match(ls):
            sgpa_values.append((idx, ls, "Incomplete"))
        # Re. entry
        m_re = reappear_re.search(ls)
        if m_re:
            re_entries.append((idx, m_re.group(1), ls))
    
    # ── Step 4: Match SGPA by position ──────────────────────────────
    # target_pos is 0-indexed position in student_rows on this page
    found_sgpa = None
    if 0 <= target_pos < len(sgpa_values):
        sgpa_raw = sgpa_values[target_pos][2]
        if sgpa_raw != "Incomplete":
            found_sgpa = sgpa_raw

    # ── Step 5: Find Re. entries for this student ────────────────────
    # Re. entries appear between consecutive SGPA values.
    # Student at target_pos: SGPA at sgpa_values[target_pos-1] (line idx) < Re. line < sgpa_values[target_pos] (line idx)
    reappear_codes = []
    if re_entries:
        # Get the SGPA line index boundaries for this student's slot
        prev_sgpa_idx = sgpa_values[target_pos - 1][0] if target_pos > 0 else -1
        curr_sgpa_idx = sgpa_values[target_pos][0] if target_pos < len(sgpa_values) else 99999
        
        for re_line_idx, re_code, re_raw in re_entries:
            if prev_sgpa_idx < re_line_idx < curr_sgpa_idx:
                reappear_codes.append(re_code)
    
    # ── Step 6: Find student full name ──────────────────────────────
    found_name = None
    window_start = max(0, target_line_idx - 20)
    window_end = min(len(lines), target_line_idx + 30)
    window_text = "\n".join(lines[window_start:window_end])
    
    name_candidates = re.findall(r'\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,3})\b', window_text)
    SKIP_KW = {"Discrete", "Computer", "Software", "Object", "Data", "National", "Institute",
               "Notification", "Subjects"}
    student_names = [n for n in name_candidates
                     if 2 <= len(n.split()) <= 4
                     and not any(kw in n for kw in SKIP_KW)]
    if student_names:
        roll_rel = target_line_idx - window_start
        window_lines = lines[window_start:window_end]
        best_name = None
        best_dist = 9999
        for sn in student_names:
            for li, ln in enumerate(window_lines):
                if sn in ln:
                    dist = abs(li - roll_rel)
                    if dist < best_dist:
                        best_dist = dist
                        best_name = sn
        found_name = best_name
    
    return {
        "found_roll": found_roll,
        "found_student_name": found_name,
        "found_sgpa": found_sgpa,
        "reappear_codes": reappear_codes,
    }


def _deterministic_extract(documents: list, search_term: str, is_roll: bool):
    """
    Iterate over PDF pages and use page-level extraction to find the student.
    Returns the first successful result, or None if not found.
    """
    search_lower = search_term.lower()
    for doc in documents:
        result = _deterministic_extract_page(doc.page_content, search_lower)
        if result is not None:
            return result
    return None


# ─────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────
async def extract_student_result(file_path: str, roll_number: str = None, student_name: str = None) -> dict:
    """
    Hybrid extraction:
    1. Excel/CSV structured row parsing (100% exact match).
    2. Deterministic parser for NIT KKR SGPA-only PDFs.
    3. LLM fallback for complex PDF formats.
    """
    cache_key = (file_path, str(roll_number or "").lower().strip(), str(student_name or "").lower().strip())
    if cache_key in _extraction_cache:
        return _extraction_cache[cache_key]

    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.xlsx', '.xls', '.csv']:
        res = extract_from_excel_or_csv(file_path, roll_number=roll_number, student_name=student_name)
        if res and res.get("found"):
            _extraction_cache[cache_key] = res
        return res

    documents = get_cached_pdf_documents(file_path)

    if not roll_number and not student_name:
        return {"error": "Please provide roll number or student name"}

    search_term = roll_number if roll_number else student_name
    identifier_type = "Roll Number" if roll_number else "Student Name"
    search_lower = search_term.lower()
    is_roll = bool(roll_number)

    # Combine all page text
    full_text = "\n".join([doc.page_content for doc in documents])

    # ── STEP 1: Extract subject codes from header (page 1 usually) ──
    all_subject_codes = re.findall(r'\b([A-Z]{2,6}-\d{3}[A-Z]?)\b', full_text)
    # Dedupe while preserving order, ignore repeat occurrences from student rows
    seen = set()
    header_subjects = []
    for sc in all_subject_codes:
        if sc not in seen:
            seen.add(sc)
            header_subjects.append(sc)
    # Keep only the first N unique codes (header), subsequent ones come from Re. entries
    # Heuristic: take the first ≤8 unique codes as the semester subjects
    semester_subjects = header_subjects[:8] if len(header_subjects) > 8 else header_subjects

    # ── STEP 2: Try deterministic extraction ────────────────────────
    parsed = _deterministic_extract(documents, search_lower, is_roll)

    if parsed:
        found_roll = parsed["found_roll"] or search_term
        found_name = parsed["found_student_name"] or search_term
        found_sgpa = parsed["found_sgpa"]
        reappear_codes = parsed["reappear_codes"]

        # Build subjects list — only show subjects if there are reappear entries
        subjects_list = []
        if reappear_codes:
            for sc in semester_subjects:
                sname = SUBJECT_NAME_MAP.get(sc, sc)
                is_re = sc in reappear_codes
                subjects_list.append({
                    "subject_code": sc,
                    "subject_name": sname,
                    "marks_obtained": "N/A",
                    "max_marks": "N/A",
                    "grade": "Re" if is_re else "Pass"
                })

        result_status = "Reappear" if reappear_codes else "Pass"
        remarks = None
        if reappear_codes:
            remarks = f"Reappear in: {', '.join(reappear_codes)}"
        else:
            remarks = "Individual subject marks not available in this PDF — only overall SGPA is declared."

        result = {
            "found": True,
            "student_name": found_name,
            "roll_number": found_roll,
            "subjects": subjects_list,
            "total_marks": None,
            "percentage": None,
            "result_status": result_status,
            "sgpa": found_sgpa,
            "cgpa": None,
            "remarks": remarks,
        }
        _extraction_cache[cache_key] = result
        return result

    # ── STEP 3: Deterministic failed — try LLM fallback ─────────────
    relevant_pages = []
    for i, doc in enumerate(documents):
        if search_lower in doc.page_content.lower():
            start = max(0, i - 1)
            end = min(len(documents), i + 2)
            for j in range(start, end):
                if j not in [p[0] for p in relevant_pages]:
                    relevant_pages.append((j, documents[j].page_content))

    if not relevant_pages:
        name_parts = search_lower.split()
        for i, doc in enumerate(documents):
            if any(part in doc.page_content.lower() for part in name_parts if len(part) > 3):
                start = max(0, i - 1)
                end = min(len(documents), i + 2)
                for j in range(start, end):
                    if j not in [p[0] for p in relevant_pages]:
                        relevant_pages.append((j, documents[j].page_content))
            if len(relevant_pages) >= 6:
                break

    if not relevant_pages:
        return {
            "found": False,
            "message": f"Student with {identifier_type} '{search_term}' not found in this result."
        }

    relevant_pages.sort(key=lambda x: x[0])
    context_text = "\n\n--- PAGE BREAK ---\n\n".join([p[1] for p in relevant_pages])
    if len(context_text) > 1200:
        context_text = context_text[:1200]

    llm = get_llm()
    prompt = f"""You are a university result extraction assistant for NIT Kurukshetra.

Find and extract the result for the student with {identifier_type}: {search_term}

CRITICAL NOTE: This PDF shows SGPA per student, NOT individual subject marks.
The column after student names contains SGPA (e.g., 8.2000, 9.5000) — NOT marks.
Do NOT put SGPA values into the marks_obtained field.

Result PDF Content:
{context_text}

Return a JSON object:
{{
  "found": true,
  "student_name": "full name",
  "roll_number": "roll number",
  "subjects": [],
  "total_marks": null,
  "percentage": null,
  "result_status": "Pass or Reappear",
  "sgpa": "SGPA value like 8.2000",
  "cgpa": null,
  "remarks": "Reappear in: SUBJECTCODE if reappear, else null"
}}

If student not found: {{"found": false, "message": "Not found"}}
Return ONLY valid JSON, no markdown.
"""

    response = llm.invoke(prompt)
    raw = response.content.strip()
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    raw = raw.strip()

    try:
        parsed_result = json.loads(raw)
        if parsed_result.get("found"):
            _extraction_cache[cache_key] = parsed_result
        return parsed_result
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                parsed_result = json.loads(match.group())
                if parsed_result.get("found"):
                    _extraction_cache[cache_key] = parsed_result
                return parsed_result
            except:
                pass
        return {"found": False, "message": f"Could not parse result. Raw: {raw[:200]}"}


async def query_result_rag(question: str, degree: str = None, branch: str = None, semester: int = None, year: int = None) -> str:
    """RAG query over indexed result PDFs & Excel files."""
    vs = get_result_vectorstore()

    filter_dict = {}
    if branch: filter_dict["branch"] = branch
    if semester: filter_dict["semester"] = semester
    if year: filter_dict["year"] = year
    if degree: filter_dict["degree"] = degree

    try:
        if filter_dict:
            docs = vs.similarity_search(question, k=2, filter=filter_dict)
        else:
            docs = vs.similarity_search(question, k=2)
    except Exception as e:
        docs = vs.similarity_search(question, k=2)

    if not docs:
        return "No matching result records found in the database."

    context_text = "\n".join([f"• {d.page_content}" for d in docs])
    if len(context_text) > 1500:
        context_text = context_text[:1500]

    prompt = f"""You are an official NIT Kurukshetra Result Assistant.
Answer the question accurately based on the result documents provided below.

Guidelines:
- If the question asks for a count, percentage, list, or aggregate statistics (e.g. "number of students with CGPA/SGPA more than 8.5" or "toppers"), analyze the context provided and give an accurate count and summary.
- List student names, roll numbers, and SGPA where helpful.
- Present data in clean Markdown format (bullet points or tables).
- Be concise and clear.

Context from result database:
{context_text}

Question: {question}

Answer:"""

    llm = get_llm()
    res = llm.invoke(prompt)
    return res.content.strip()

def execute_direct_range_query(question: str):
    """
    Directly scans all uploaded Excel/CSV files in DB/filesystem for numerical range conditions
    (e.g., SGPA > 8.5, SGPA >= 9.0, SGPA < 6.0, toppers, reappear list).
    Returns 100% accurate, complete student list & exact counts across all student records.
    """
    q_lower = question.lower()

    # Extract numerical condition
    num_match = re.search(r'(?:sgpa|cgpa|marks|score)?\s*(?:more than|greater than|above|>|>=|exceeds|over)\s*(\d+(?:\.\d+)?)', q_lower)
    op = None
    target_val = None
    if num_match:
        op = ">"
        target_val = float(num_match.group(1))
    else:
        num_match_less = re.search(r'(?:sgpa|cgpa|marks|score)?\s*(?:less than|below|<|<=|under)\s*(\d+(?:\.\d+)?)', q_lower)
        if num_match_less:
            op = "<"
            target_val = float(num_match_less.group(1))

    if target_val is None and not any(k in q_lower for k in ["reappear", "fail", "topper", "top"]):
        return None

    from db import SessionLocal
    from models import Result
    import os

    db = SessionLocal()
    matching_students = []
    total_scanned = 0

    try:
        results_in_db = db.query(Result).order_by(Result.semester.asc()).all()
        for r_file in results_in_db:
            if not os.path.exists(r_file.file_path):
                continue

            rows = parse_excel_or_csv(r_file.file_path)
            for row in rows:
                total_scanned += 1
                s_name = row.get('student name') or row.get('name') or row.get('full name') or 'N/A'
                s_roll = row.get('roll number') or row.get('roll no') or row.get('roll') or row.get('rollno') or 'N/A'
                sgpa_str = row.get('sgpa') or row.get('cgpa') or row.get('gpa') or '0.0'
                status = row.get('result status') or row.get('status') or row.get('remarks') or 'Pass'

                try:
                    sgpa_val = float(sgpa_str)
                except:
                    sgpa_val = 0.0

                is_match = False
                if op == ">" and sgpa_val > target_val:
                    is_match = True
                elif op == "<" and sgpa_val < target_val:
                    is_match = True
                elif "reappear" in q_lower or "fail" in q_lower:
                    if "reappear" in status.lower() or "fail" in status.lower() or "reappear" in str(row).lower():
                        is_match = True
                elif "top" in q_lower or "topper" in q_lower:
                    if sgpa_val >= 9.0:
                        is_match = True

                if is_match:
                    matching_students.append({
                        "name": s_name,
                        "roll": s_roll,
                        "sgpa": sgpa_str,
                        "sgpa_val": sgpa_val,
                        "status": status,
                        "sem": f"Sem {r_file.semester} ({r_file.branch})",
                        "file": r_file.original_filename
                    })

    finally:
        db.close()

    if not matching_students and total_scanned == 0:
        return None

    # Deduplicate by (roll, sem)
    unique_matches = []
    seen = set()
    for item in matching_students:
        key = (item["roll"], item["sem"])
        if key not in seen:
            seen.add(key)
            unique_matches.append(item)

    unique_matches.sort(key=lambda x: x["sgpa_val"], reverse=True)

    cond_desc = f"SGPA / CGPA > {target_val}" if op == ">" else (f"SGPA / CGPA < {target_val}" if op == "<" else "Matching criteria")
    ans = f"""🎓 **Direct Database & Excel Analytical Results**

• **Query Filter:** `{cond_desc}`
• **Total Matching Records Found:** **{len(unique_matches)}** out of **{total_scanned}** scanned student records across all uploaded result files.

📊 **Complete List of Matching Students:**

| # | Roll Number | Student Name | SGPA / CGPA | Branch & Semester | Status |
|---|-------------|--------------|-------------|-------------------|--------|
"""
    display_limit = 50
    for idx, s in enumerate(unique_matches[:display_limit], start=1):
        ans += f"| {idx} | `{s['roll']}` | **{s['name']}** | `{s['sgpa']}` | {s['sem']} | {s['status']} |\n"

    if len(unique_matches) > display_limit:
        ans += f"\n*(Showing top {display_limit} of {len(unique_matches)} total matching students)*\n"

    return ans


async def unified_chat(question: str, user_id: str = None) -> dict:
    """
    Unified NLP chat endpoint — auto-detects intent and routes to appropriate service:
    - Result queries & Roll Numbers → Direct PDF extraction + RAG vectorstore
    - Policy/notification queries → notification vectorstore  
    - Announcement queries → announcement vectorstore
    - General queries → LLM direct
    Returns: {answer, source_type, sources}
    """
    q_lower = question.lower()

    # Detect Roll Number (e.g. 124102014, 12112045) or Student Name pattern
    roll_pattern = re.search(r'\b(\d{7,12})\b', question)
    found_roll = roll_pattern.group(1) if roll_pattern else None

    # Detect intent keywords
    result_keywords = ["result", "marks", "grade", "sgpa", "cgpa", "pass", "fail", "semester result",
                       "score", "subject marks", "mera result", "my result", "percentage", "reappear"]
    notification_keywords = ["policy", "attendance", "internship", "scholarship", "rule", "regulation",
                              "notice", "circular", "guideline", "leave", "exam policy", "fee", "holiday",
                              "raksha", "rakshabandhan", "diwali", "holi", "vacation", "off"]
    announcement_keywords = ["announcement", "admission", "merit", "scholarship list"]

    is_result = bool(found_roll) or any(k in q_lower for k in result_keywords)
    is_notification = any(k in q_lower for k in notification_keywords)
    is_announcement = any(k in q_lower for k in announcement_keywords)

    # Detect aggregate / analytics queries (e.g. "how many students", "cgpa more than 8.5", "topper")
    aggregate_keywords = ["number of", "how many", "count", "list", "top", "topper", "average", "highest",
                          "more than", "greater than", "above", "less than", "below", "pass percentage"]
    is_aggregate = any(ak in q_lower for ak in aggregate_keywords)

    # ── 1. RESULT QUERY (Direct PDF/Excel Extraction + Multi-Sem Aggregation) ──────
    if is_result:
        # Step A: Check numerical range query (e.g., SGPA > 8.5) -> execute direct Excel/DB scanning
        if is_aggregate:
            direct_ans = execute_direct_range_query(question)
            if direct_ans:
                return {"answer": direct_ans, "source_type": "result", "sources": ["Result Database (Excel/CSV Scan)"]}

            try:
                answer = await query_result_rag(question)
                if answer and len(answer.strip()) > 10:
                    return {"answer": answer, "source_type": "result", "sources": ["Result Database"]}
            except Exception as e:
                print(f"RAG aggregate query error: {e}")

        # Detect student name if roll number is missing
        search_name = None
        if not found_roll:
            name_match = re.search(r'(?:result of|result for|marks of|marks for)\s+([a-zA-Z]{3,20})', question, re.IGNORECASE)
            if name_match:
                search_name = name_match.group(1).strip()
            elif not any(k in q_lower for k in notification_keywords + announcement_keywords):
                clean_q = re.sub(r'(?:result|marks|score|ka|batao|check|dekhna|show|get)', '', q_lower).strip()
                if 2 <= len(clean_q) <= 25 and not clean_q.isdigit():
                    search_name = clean_q

        target_roll = found_roll
        target_name = search_name

        if target_roll or target_name:
            try:
                from db import SessionLocal
                from models import Result
                import os

                db = SessionLocal()
                found_records = []
                try:
                    results_in_db = db.query(Result).order_by(Result.semester.asc()).all()
                    for r_file in results_in_db:
                        if not os.path.exists(r_file.file_path):
                            continue
                        extracted = await extract_student_result(
                            r_file.file_path,
                            roll_number=target_roll,
                            student_name=target_name
                        )
                        if extracted.get("found"):
                            found_records.append({
                                "file_info": r_file,
                                "data": extracted
                            })

                    if found_records:
                        # Deduplicate by semester if needed
                        sem_map = {}
                        for item in found_records:
                            sem = item["file_info"].semester
                            sem_map[sem] = item
                        sorted_items = [sem_map[s] for s in sorted(sem_map.keys())]

                        first_item = sorted_items[0]
                        s_name = first_item["data"].get("student_name", "Student")
                        s_roll = first_item["data"].get("roll_number", target_roll or "N/A")
                        degree = first_item["file_info"].degree
                        branch = first_item["file_info"].branch

                        if len(sorted_items) == 1:
                            item = sorted_items[0]
                            r_file = item["file_info"]
                            data = item["data"]
                            ans = f"""🎓 **Student Result Found**

• **Student Name:** {s_name}
• **Roll Number:** {s_roll}
• **Degree & Branch:** {degree} {branch}
• **Semester:** Semester {r_file.semester} ({r_file.type} {r_file.year})
• **Overall SGPA:** {data.get('sgpa', 'N/A')}
• **Result Status:** {data.get('result_status', 'N/A')}
"""
                            if data.get('remarks'):
                                ans += f"\n📌 **Details:** {data.get('remarks')}\n"
                            ans += f"\n📄 *Source File:* `{r_file.original_filename}`"
                            return {"answer": ans, "source_type": "result", "sources": [r_file.original_filename]}

                        else:
                            # Multi-Semester Academic Transcript
                            sgpas = []
                            ans = f"""🎓 **Student Academic Report (Multi-Semester)**

• **Student Name:** {s_name}
• **Roll Number:** {s_roll}
• **Degree & Branch:** {degree} {branch}
• **Total Semesters Found:** {len(sorted_items)}

📊 **Semester-Wise Performance Breakdown:**
─────────────────────────────────────────────
"""
                            sources = []
                            for idx, item in enumerate(sorted_items, start=1):
                                r_file = item["file_info"]
                                data = item["data"]
                                sgpa_str = data.get("sgpa", "N/A")
                                status_str = data.get("result_status", "N/A")
                                remarks_str = data.get("remarks", "")
                                sources.append(r_file.original_filename)

                                try:
                                    sgpa_val = float(sgpa_str)
                                    sgpas.append(sgpa_val)
                                except:
                                    pass

                                ans += f"""{idx}️⃣ **Semester {r_file.semester} ({r_file.type} {r_file.year}):**
   • **SGPA:** {sgpa_str} | **Status:** {status_str}
"""
                                if remarks_str:
                                    ans += f"   • **Details:** {remarks_str}\n"
                                ans += "\n"

                            if sgpas:
                                cgpa = sum(sgpas) / len(sgpas)
                                ans += f"⭐ **Cumulative CGPA (CGPA):** `{cgpa:.4f}`\n"

                            ans += f"\n📄 *Source Files:* {', '.join(set(sources))}"
                            return {"answer": ans, "source_type": "result", "sources": list(set(sources))}

                finally:
                    db.close()
            except Exception as ex:
                print(f"Direct extraction attempt error: {ex}")

        # Step B: Fallback to RAG vectorstore query
        try:
            answer = await query_result_rag(question)
            if answer and len(answer.strip()) > 10:
                return {"answer": answer, "source_type": "result", "sources": ["Result Database"]}
        except Exception as e:
            pass

        # Step C: If roll number was missing, prompt student to provide Roll Number
        if not found_roll:
            return {
                "answer": "🎓 **NIT Kurukshetra Result Assistant**\n\nKripya apna **Roll Number** (e.g. `124102014` ya `12112045`) type karein, main aapka SGPA, status aur result details instantly search karke bata dunga!",
                "source_type": "result",
                "sources": ["Result Assistant"]
            }

    # ── 2. POLICY / NOTIFICATION QUERY ──────────────────────────────
    if is_notification:
        try:
            from services.rag_service import query_notifications
            answer = await query_notifications(question)
            return {"answer": answer, "source_type": "notification", "sources": ["Policy Documents"]}
        except Exception as e:
            pass

    # ── 3. ANNOUNCEMENT QUERY ────────────────────────────────────────
    if is_announcement:
        try:
            vs = get_announcement_vectorstore()
            docs = vs.similarity_search(question, k=3)
            context_str = "\n".join([f"• {d.page_content}" for d in docs])
            if len(context_str) > 1500:
                context_str = context_str[:1500]
            prompt = f"""You are a helpful NIT Kurukshetra assistant.
Answer the student's query based on official announcements.

Context:
{context_str}

Question: {question}

Answer:"""
            res = get_llm().invoke(prompt)
            return {"answer": res.content, "source_type": "announcement", "sources": ["Announcements"]}
        except Exception as e:
            pass

    # ── 4. GENERAL FALLBACK LLM ─────────────────────────────────────
    llm = get_llm()
    fallback_prompt = f"""You are a helpful AI assistant for NIT Kurukshetra students and faculty.
Answer the following question in a friendly, helpful way.
If you don't have specific information about NIT KKR, provide general helpful information.

Question: {question}

Answer:"""

    response = llm.invoke(fallback_prompt)
    return {
        "answer": response.content,
        "source_type": "general",
        "sources": ["AI Assistant"]
    }