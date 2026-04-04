from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.auth.utils import verify_password, hash_password, create_access_token
from app.auth.users import get_user, user_exists, create_user
from app.models.schemas import Token, UserCreate

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(req: UserCreate):
    if user_exists(req.username):
        raise HTTPException(status_code=400, detail="Username already taken.")
    create_user(req.username, hash_password(req.password))
    return {"message": "Account created. You can now log in."}


@router.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = get_user(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(form_data.username)
    return Token(access_token=token, token_type="bearer")
