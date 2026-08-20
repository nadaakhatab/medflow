import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt

from backend import models, schemas
from backend.database import get_db

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# Environment configuration for JWT authentication secrets
SECRET_KEY = os.getenv("SECRET_KEY") or os.getenv("AUTH_SECRET_KEY")
# Never use a source-controlled default JWT secret. A deployment without a Space
# secret still receives a cryptographically strong process-local key; sessions
# safely expire on a restart. Set SECRET_KEY in Space Secrets for stable sessions.
if not SECRET_KEY:
    SECRET_KEY = secrets.token_urlsafe(48)
ALGORITHM = os.getenv("AUTH_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 7 days
SECURE_COOKIE = os.getenv("SECURE_COOKIE", "false").lower() == "true" or os.getenv("APP_ENV") == "production"

import bcrypt

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            return False

def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(request: Request, db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    # Read token from HttpOnly cookie or Bearer header
    token = request.cookies.get("access_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        
    if not token:
        # This is intentionally opt-in and is never available in Public Mode.
        initial_admin_email = os.getenv("INITIAL_ADMIN_EMAIL", "").strip()
        dev_bypass_enabled = os.getenv("DEV_AUTH_BYPASS", "false").lower() == "true"
        local_client = request.client and request.client.host in {"127.0.0.1", "::1", "localhost"}
        if initial_admin_email and dev_bypass_enabled and os.getenv("APP_ENV") != "production" and local_client:
            dev_user = db.query(models.User).filter(models.User.email == initial_admin_email.lower()).first()
            if dev_user:
                return dev_user
        raise credentials_exception
        
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = db.query(models.User).filter(models.User.email == email.lower()).first()
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user

async def get_current_admin(current_user: models.User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges"
        )
    return current_user

@router.post("/register", response_model=schemas.UserResponse)
def register(user_in: schemas.UserCreate, response: Response, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == user_in.email.lower()).first()
    if user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists."
        )
    
    hashed_password = get_password_hash(user_in.password)
    db_user = models.User(
        full_name=user_in.full_name,
        email=user_in.email.lower(),
        password_hash=hashed_password,
        role="user",
        profession=user_in.profession,
        is_active=True
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.email}, expires_delta=access_token_expires
    )
    
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        expires=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=SECURE_COOKIE,
    )

    return db_user

@router.post("/login", response_model=schemas.UserResponse)
def login(login_data: schemas.UserLogin, response: Response, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == login_data.email.lower()).first()
    if not user or not verify_password(login_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        expires=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=SECURE_COOKIE,
    )

    return user

@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(
        key="access_token",
        httponly=True,
        samesite="lax",
        secure=SECURE_COOKIE,
    )
    return {"message": "Logged out successfully"}

@router.get("/me", response_model=schemas.UserResponse)
def read_users_me(current_user: models.User = Depends(get_current_user)):
    return current_user
