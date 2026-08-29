"""
Generate multi-semester Excel result files for NIT Kurukshetra,
enabling complete semester-wise academic report cards for students like Akshat Sharma, Deepak Sharma, etc.
"""
import os, sys, random
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

# Shared multi-semester students (they appear in Sem 1, Sem 2, and Sem 3)
MULTI_SEM_STUDENTS = [
    {"roll": "124101005", "name": "Akshat Sharma", "sem1_sgpa": "8.5000", "sem2_sgpa": "8.9000", "sem3_sgpa": "9.1000"},
    {"roll": "12212193", "name": "Deepak Sharma", "sem1_sgpa": "8.4000", "sem2_sgpa": "8.6000", "sem3_sgpa": "8.7500"},
    {"roll": "124102014", "name": "Sanjeev Kumar", "sem1_sgpa": "8.0000", "sem2_sgpa": "8.1000", "sem3_sgpa": "8.2000"},
    {"roll": "124101006", "name": "Rohan Verma", "sem1_sgpa": "7.5000", "sem2_sgpa": "7.8000", "sem3_sgpa": "8.1000"},
    {"roll": "124101007", "name": "Priya Gupta", "sem1_sgpa": "9.2000", "sem2_sgpa": "9.4000", "sem3_sgpa": "9.6000"},
]

FIRST_NAMES = ["Aarav", "Aditi", "Aditya", "Akash", "Aman", "Ananya", "Aniket", "Anjali", "Ankit", "Arjun", "Ayush", "Bhavya", "Dev", "Divya", "Gaurav", "Harsh", "Isha", "Jatin", "Karan", "Kavya", "Khushi", "Kunal", "Mehak", "Mohit", "Neha", "Nikhil", "Parth", "Pooja", "Pranav", "Rahul", "Riya", "Rohan", "Sachin", "Sahil", "Sakshi", "Sameer", "Sarthak", "Shivam", "Shreya", "Siddharth", "Sneha", "Sumit", "Tanmay", "Utkarsh", "Vaibhav", "Varun", "Yash"]
LAST_NAMES = ["Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Joshi", "Mehta", "Aggarwal", "Chawla", "Mishra", "Pandey", "Yadav", "Kapoor", "Saxena"]

COURSES = [
    {
        "degree": "BTech", "branch": "CSE", "semester": 1, "year": 2023, "type": "Regular",
        "filename": "BTech_CSE_Sem1_2023_Results.xlsx", "sem_key": "sem1_sgpa"
    },
    {
        "degree": "BTech", "branch": "CSE", "semester": 2, "year": 2024, "type": "Regular",
        "filename": "BTech_CSE_Sem2_2024_Results.xlsx", "sem_key": "sem2_sgpa"
    },
    {
        "degree": "BTech", "branch": "CSE", "semester": 3, "year": 2024, "type": "Regular",
        "filename": "BTech_CSE_Sem3_2024_Results.xlsx", "sem_key": "sem3_sgpa"
    },
    {
        "degree": "BTech", "branch": "ECE", "semester": 3, "year": 2024, "type": "Regular",
        "filename": "BTech_ECE_Sem3_2024_Results.xlsx", "sem_key": None
    },
    {
        "degree": "BTech", "branch": "IT", "semester": 5, "year": 2024, "type": "Regular",
        "filename": "BTech_IT_Sem5_2024_Results.xlsx", "sem_key": None
    },
]

random.seed(100)

db = SessionLocal()
try:
    for course in COURSES:
        file_path = os.path.join("uploads/results", course["filename"])
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = f"{course['branch']} Sem {course['semester']}"
        ws.append(["Roll Number", "Student Name", "SGPA", "Result Status", "Reappear Subjects / Remarks"])

        records = []
        # Add shared multi-sem students if this course has sem_key
        if course["sem_key"]:
            for ms in MULTI_SEM_STUDENTS:
                records.append([ms["roll"], ms["name"], ms[course["sem_key"]], "Pass", "Clear"])

        # Fill remaining up to 100 students
        count = 100 - len(records)
        base_roll = 124100000 + course["semester"] * 1000
        for i in range(count):
            roll = str(base_roll + i + 10)
            name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
            is_pass = random.random() > 0.10
            sgpa = f"{random.uniform(7.2, 9.8):.4f}" if is_pass else f"{random.uniform(5.5, 6.8):.4f}"
            status = "Pass" if is_pass else "Reappear"
            remarks = "Clear" if is_pass else "Reappear in: CSPC-201"
            records.append([roll, name, sgpa, status, remarks])

        for row in records:
            ws.append(row)

        wb.save(file_path)
        print(f"✅ Generated {course['filename']} with {len(records)} student records.")

        # Register DB
        existing = db.query(Result).filter(Result.original_filename == course["filename"]).first()
        if not existing:
            res_record = Result(
                degree=course["degree"],
                branch=course["branch"],
                semester=course["semester"],
                year=course["year"],
                type=course["type"],
                file_path=file_path,
                original_filename=course["filename"]
            )
            db.add(res_record)
            db.commit()
            db.refresh(res_record)
            res_id = res_record.id
            print(f"   💾 Saved in DB (ID: {res_id})")
        else:
            res_id = existing.id
            print(f"   ℹ️ Record already in DB (ID: {res_id})")

        # Index ChromaDB
        index_result_pdf(
            file_path,
            course["degree"],
            course["branch"],
            course["semester"],
            course["year"],
            course["type"],
            res_id
        )
        print(f"   ⚡ Indexed {len(records)} rows in Vectorstore!")

finally:
    db.close()

print("\n🎉 MULTI-SEMESTER DATASET SEEDED SUCCESSFULLY FOR AKSHAT SHARMA, DEEPAK SHARMA, SANJEEV KUMAR ETC.!")
