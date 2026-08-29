from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
from db import get_db
from models import Result, User
from services.dependencies import admin_only, get_current_user
from services.file_service import save_file
from services.result_extraction_service import index_result_pdf
import os, time, asyncio

router = APIRouter(prefix="/admin", tags=["admin"])

DEGREES  = ["BTech", "MTech", "BCA", "MCA", "BSc", "MSc"]
BRANCHES = ["CSE", "ECE", "ME", "CE", "IT", "Electrical", "IOT"]
TYPES    = ["Regular", "Reappear"]

# ── In-process TTL caches ───────────────────────────────────────────────────
_stats_cache = {"data": None, "ts": 0.0}
_STATS_TTL = 30.0          # seconds — invalidated by upload/delete too

# /options is pure static data — cache forever (process lifetime)
_options_cache = None

def _invalidate_stats_cache():
    _stats_cache["data"] = None
    _stats_cache["ts"] = 0.0


@router.get("/options")
def get_options():
    global _options_cache
    if _options_cache is None:
        _options_cache = {
            "degrees": DEGREES,
            "branches": BRANCHES,
            "types": TYPES,
            "semesters": list(range(1, 9)),
            "years": list(range(2018, 2030))
        }
    return _options_cache


@router.post("/upload-result")
async def upload_result(
    degree:   str = Form(...),
    branch:   str = Form(...),
    semester: int = Form(...),
    year:     int = Form(...),
    type:     str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(admin_only)
):
    ext = os.path.splitext(file.filename)[1].lower()
    allowed_exts = [".pdf", ".xlsx", ".xls", ".csv"]
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail="Only PDF, Excel (.xlsx/.xls), and CSV (.csv) files allowed")

    file_path = await save_file(file)

    result = Result(
        degree=degree, branch=branch, semester=semester,
        year=year, type=type, file_path=file_path,
        original_filename=file.filename
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    result_id = result.id

    # Invalidate stats cache so counts update on next /stats call
    _invalidate_stats_cache()

    # Run vectorstore indexing in a background thread — response returns immediately
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        lambda: _background_index(file_path, degree, branch, semester, year, type, result_id)
    )

    return {
        "message": "Result uploaded successfully",
        "id": result_id,
        "rag_indexed": "indexing_in_background"
    }


def _background_index(file_path, degree, branch, semester, year, result_type, result_id):
    """Run vectorstore indexing in a background thread."""
    try:
        index_result_pdf(file_path, degree, branch, semester, year, result_type, result_id)
        print(f"✅ Background indexing complete for result_id={result_id}")
    except Exception as e:
        print(f"⚠️ Background indexing failed for result_id={result_id}: {e}")


@router.get("/results")
def get_all_results(
    db: Session = Depends(get_db),
    current_user: dict = Depends(admin_only)
):
    results = db.query(Result).order_by(Result.uploaded_at.desc()).all()
    return [
        {
            "id": r.id, "degree": r.degree, "branch": r.branch,
            "semester": r.semester, "year": r.year, "type": r.type,
            "original_filename": r.original_filename, "file_path": r.file_path,
            "uploaded_at": r.uploaded_at.isoformat() if r.uploaded_at else None,
        }
        for r in results
    ]


@router.delete("/results/{result_id}")
def delete_result(
    result_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(admin_only)
):
    result = db.query(Result).filter(Result.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    # Delete file from disk
    if os.path.exists(result.file_path):
        os.remove(result.file_path)

    db.delete(result)
    db.commit()
    _invalidate_stats_cache()  # Keep counts accurate
    return {"message": "Result deleted successfully"}


@router.get("/stats")
def get_stats(
    db: Session = Depends(get_db),
    current_user: dict = Depends(admin_only)
):
    """Admin dashboard statistics — cached for 30 seconds."""
    now = time.time()
    if _stats_cache["data"] is not None and (now - _stats_cache["ts"]) < _STATS_TTL:
        return _stats_cache["data"]

    total_results = db.query(Result).count()
    total_students = db.query(User).filter(User.role == "student").count()
    total_admins = db.query(User).filter(User.role == "admin").count()

    from models import Notification, Announcement
    total_notifications = db.query(Notification).count()
    total_announcements = db.query(Announcement).count()

    data = {
        "results": total_results,
        "students": total_students,
        "admins": total_admins,
        "notifications": total_notifications,
        "announcements": total_announcements,
    }
    _stats_cache["data"] = data
    _stats_cache["ts"] = now
    return data


@router.post("/create-admin")
def create_admin(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(admin_only)
):
    """Create another admin account."""
    from models import User
    from services.auth_service import hash_password

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_admin = User(
        name=name, email=email,
        password_hash=hash_password(password),
        role="admin"
    )
    db.add(new_admin)
    db.commit()
    return {"message": f"Admin {name} created successfully"}


@router.post("/create-teacher")
def create_teacher(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(admin_only)
):
    """Create a teacher account — only admins can do this."""
    from services.auth_service import hash_password

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_teacher = User(
        name=name, email=email,
        password_hash=hash_password(password),
        role="teacher"
    )
    db.add(new_teacher)
    db.commit()
    db.refresh(new_teacher)
    return {"message": f"Teacher '{name}' created successfully", "email": email, "role": "teacher"}


@router.get("/users")
def list_users(
    role: str = Query(None),
    status: str = Query(None),
    db: Session = Depends(get_db),
    current_user: dict = Depends(admin_only)
):
    """List all users, optionally filter by role (student/teacher/admin) or status (pending/approved)."""
    q = db.query(User)
    if role:
        q = q.filter(User.role == role)
    if status == "pending":
        q = q.filter(User.is_approved == False)
    elif status == "approved":
        q = q.filter(User.is_approved == True)

    return [
        {
            "id": u.id, "name": u.name, "email": u.email, "role": u.role,
            "is_approved": bool(u.is_approved),
            "created_at": u.created_at.isoformat() if u.created_at else None,
        }
        for u in q.order_by(User.created_at.desc()).all()
    ]


@router.get("/pending-students")
def list_pending_students(
    db: Session = Depends(get_db),
    current_user: dict = Depends(admin_only)
):
    """List all student registration requests awaiting admin approval."""
    pending = db.query(User).filter(User.role == "student", User.is_approved == False).order_by(User.created_at.desc()).all()
    return [
        {
            "id": u.id, "name": u.name, "email": u.email, "role": u.role,
            "is_approved": False,
            "created_at": u.created_at.isoformat() if u.created_at else None,
        }
        for u in pending
    ]


@router.post("/approve-student/{user_id}")
def approve_student(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(admin_only)
):
    """Approve a student account registration."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Student not found")

    user.is_approved = True
    db.commit()
    return {"message": f"Student '{user.name}' ({user.email}) approved successfully!"}


@router.post("/reject-student/{user_id}")
def reject_student(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(admin_only)
):
    """Reject and delete a student registration request."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Student not found")

    user_name = user.name
    db.delete(user)
    db.commit()
    return {"message": f"Student registration request for '{user_name}' rejected and removed."}


@router.post("/create-student")
def create_student(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(admin_only)
):
    """Admin directly creates an approved student account."""
    from services.auth_service import hash_password

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_student = User(
        name=name,
        email=email,
        password_hash=hash_password(password),
        role="student",
        is_approved=True  # Directly approved when created by Admin
    )
    db.add(new_student)
    db.commit()
    db.refresh(new_student)
    return {"message": f"Student '{name}' created and approved successfully!", "email": email}


@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(admin_only)
):
    """Delete a user. Cannot delete your own account."""
    if str(user_id) == str(current_user.get("sub")):
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {"message": f"User '{user.name}' deleted successfully"}