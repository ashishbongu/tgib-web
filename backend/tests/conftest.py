from app.main import app
from app.auth.deps import get_current_user

def override_get_current_user():
    return {"user_id": "test"}

app.dependency_overrides[get_current_user] = override_get_current_user