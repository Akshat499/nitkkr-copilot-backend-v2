import os
import re
from db import SessionLocal
from models import Result
from services.result_extraction_service import index_result_pdf

def sync_all_results():
    db = SessionLocal()
    try:
        existing_files = set(os.path.basename(r.file_path).lower() for r in db.query(Result).all())
        all_files = [f for f in os.listdir("uploads/results") if f.endswith(('.xlsx', '.xls', '.csv', '.pdf'))]
        added = 0

        for f in all_files:
            if f.lower() not in existing_files:
                sem_m = re.search(r'sem(?:ester)?\s*(\d+)', f, re.I)
                sem = int(sem_m.group(1)) if sem_m else 1
                yr_m = re.search(r'(20\d{2})', f)
                yr = int(yr_m.group(1)) if yr_m else 2024
                branch = 'ECE' if 'ece' in f.lower() else ('IT' if 'it' in f.lower() else 'CSE')
                type_ = 'Reappear' if 're' in f.lower() else 'Regular'
                path = os.path.join("uploads/results", f)

                r = Result(
                    degree='BTech',
                    branch=branch,
                    semester=sem,
                    year=yr,
                    type=type_,
                    original_filename=f,
                    file_path=path
                )
                db.add(r)
                db.commit()
                db.refresh(r)
                index_result_pdf(path, 'BTech', branch, sem, yr, type_, r.id)
                added += 1
                print(f"✅ Added & indexed: {f}")

        print(f"🎉 Total newly registered result files: {added}")
    finally:
        db.close()

if __name__ == "__main__":
    sync_all_results()
