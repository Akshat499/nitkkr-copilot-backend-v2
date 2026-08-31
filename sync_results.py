import os
import re
from db import SessionLocal
from models import Result, Announcement, Notification
from services.result_extraction_service import index_result_pdf, index_announcement_pdf
from services.rag_service import index_notification

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

def sync_all_announcements():
    db = SessionLocal()
    try:
        announcements = db.query(Announcement).all()
        synced = 0
        for ann in announcements:
            if os.path.exists(ann.file_path):
                try:
                    index_announcement_pdf(ann.file_path, ann.title, ann.id)
                    synced += 1
                except Exception as e:
                    print(f"⚠️ Error syncing announcement {ann.id}: {e}")
        print(f"✅ Synced & indexed {synced} announcements into vectorstore.")
    finally:
        db.close()

def sync_all_notifications():
    db = SessionLocal()
    try:
        notifications = db.query(Notification).all()
        synced = 0
        for notif in notifications:
            if os.path.exists(notif.file_path):
                try:
                    year = notif.year or (notif.uploaded_at.year if notif.uploaded_at else 2024)
                    index_notification(notif.file_path, notif.title, year)
                    synced += 1
                except Exception as e:
                    print(f"⚠️ Error syncing notification {notif.id}: {e}")
        print(f"✅ Synced & indexed {synced} policy notifications into vectorstore.")
    finally:
        db.close()

if __name__ == "__main__":
    sync_all_results()
    sync_all_announcements()
    sync_all_notifications()
