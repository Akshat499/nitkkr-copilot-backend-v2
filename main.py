from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from db import engine, Base
import models
from routers import auth, admin, student, notifications, announcements

app = FastAPI(title="NIT KKR Smart Edu Copilot", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

# Ensure upload directories exist for static mounting and uploads
for folder in ["uploads", "uploads/results", "uploads/notifications", "uploads/announcements"]:
    os.makedirs(folder, exist_ok=True)

Base.metadata.create_all(bind=engine)

# Startup routine: Ensure existing user accounts in database are approved
@app.on_event("startup")
async def startup_db_check():
    import asyncio
    from db import SessionLocal
    from models import User
    from sync_results import sync_all_results
    db = SessionLocal()
    try:
        # Approve existing users that may have NULL or False from before approval requirement
        existing_users = db.query(User).filter((User.is_approved == None) | (User.role == "admin")).all()
        for u in existing_users:
            u.is_approved = True
        db.commit()
        print("✅ Database user approval check completed.")
    except Exception as e:
        print(f"⚠️ DB startup check error: {e}")
        db.rollback()
    finally:
        db.close()

    # Run sync_all_results in background after server is live
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, sync_all_results)
    print("✅ Async result sync background task scheduled.")


# Include all routers
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(student.router)
app.include_router(notifications.router)
app.include_router(announcements.router)

# Serve uploaded files statically
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
def read_root():
    return {
        "message": "NIT KKR Smart Edu Copilot API v2.0",
        "endpoints": {
            "auth": ["/auth/signup", "/auth/login", "/auth/me", "/auth/verify"],
            "student": [
                "/student/chat", "/student/chat/guest",
                "/student/extract-result", "/student/extract-result/guest",
                "/student/results/list", "/student/query"
            ],
            "admin": [
                "/admin/upload-result", "/admin/results", "/admin/stats",
                "/admin/create-admin", "/admin/create-teacher",
                "/admin/users", "/admin/options"
            ],
            "notifications": ["/notifications/upload", "/notifications/all", "/notifications/query"],
            "announcements": ["/announcements/upload", "/announcements/all", "/announcements/query"],
        }
    }

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)