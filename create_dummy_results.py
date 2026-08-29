"""
Script to create sample Excel (.xlsx) and CSV (.csv) result files for NIT Kurukshetra,
insert records into PostgreSQL database, and index in ChromaDB vectorstore.
"""
import os, sys
import openpyxl
from datetime import datetime

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db import SessionLocal, engine
from models import Base, Result
from services.result_extraction_service import index_result_pdf

# Ensure DB tables exist
Base.metadata.create_all(bind=engine)

os.makedirs("uploads/results", exist_ok=True)

# ── 1. Create Dummy Excel File ──────────────────────────────────────
excel_path = "uploads/results/BTech_CSE_Sem3_2024_Results.xlsx"

wb = openpyxl.Workbook()
ws = wb.active
ws.title = "CSE Sem 3 Results"

# Header
ws.append(["Roll Number", "Student Name", "SGPA", "Result Status", "Reappear Subjects / Remarks"])

# Dummy Students Data (including Deepak 12212193 and Sanjeev 124102014)
students = [
    ["12212193", "Deepak Sharma", "8.7500", "Pass", "Clear"],
    ["124102014", "Sanjeev Kumar", "8.2000", "Pass", "Clear"],
    ["124102015", "Rahul Verma", "7.9000", "Pass", "Clear"],
    ["124102016", "Ananya Gupta", "9.1000", "Pass", "Clear"],
    ["124102017", "Vikram Singh", "6.4000", "Reappear", "Reappear in: CSPC-203 (Data Structures)"],
    ["124102018", "Priya Patel", "8.4500", "Pass", "Clear"],
]

for s in students:
    ws.append(s)

wb.save(excel_path)
print(f"✅ Created dummy Excel file at: {excel_path}")

# ── 2. Add to Database & Vectorstore ────────────────────────────────
db = SessionLocal()
try:
    # Check if already exists in DB
    existing = db.query(Result).filter(Result.original_filename == "BTech_CSE_Sem3_2024_Results.xlsx").first()
    if not existing:
        res_record = Result(
            degree="BTech",
            branch="CSE",
            semester=3,
            year=2024,
            type="Regular",
            file_path=excel_path,
            original_filename="BTech_CSE_Sem3_2024_Results.xlsx"
        )
        db.add(res_record)
        db.commit()
        db.refresh(res_record)
        print(f"✅ Saved Result record in DB (ID: {res_record.id})")
        res_id = res_record.id
    else:
        res_id = existing.id
        print(f"ℹ️ Result record already in DB (ID: {res_id})")

    # Index in ChromaDB
    index_result_pdf(excel_path, "BTech", "CSE", 3, 2024, "Regular", res_id)
    print("✅ Indexed Excel result in ChromaDB Vectorstore!")

finally:
    db.close()
