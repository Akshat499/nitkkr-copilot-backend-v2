import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta
from config import JWT_SECRET, JWT_EXPIRE_MINUTES

def hash_password(password: str) -> str:
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not plain_password or not hashed_password:
        return False
    try:
        pw_bytes = plain_password.encode('utf-8')
        hash_bytes = hashed_password.encode('utf-8') if isinstance(hashed_password, str) else hashed_password
        return bcrypt.checkpw(pw_bytes, hash_bytes)
    except Exception:
        try:
            from passlib.context import CryptContext
            ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
            return ctx.verify(plain_password, hashed_password)
        except Exception:
            return False

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    minutes = int(JWT_EXPIRE_MINUTES) if JWT_EXPIRE_MINUTES else 1440
    expire = datetime.utcnow() + timedelta(minutes=minutes)
    to_encode.update({"exp": expire})
    secret = str(JWT_SECRET) if JWT_SECRET else "nitkkr-copilot-secret-key-2026-prod-jwt-token"
    return jwt.encode(to_encode, secret, algorithm="HS256")

def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except JWTError:
        return None