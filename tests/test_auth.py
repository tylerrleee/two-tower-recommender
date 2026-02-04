from datetime import timedelta
from jose import jwt
from api.auth import create_access_token, verify_password, get_password_hash, SECRET_KEY, ALGORITHM

def test_password_hashing():
    pwd = "secret_password"
    hashed = get_password_hash(pwd)
    
    assert hashed != pwd
    assert verify_password(pwd, hashed) is True
    assert verify_password("wrong_password", hashed) is False

def test_jwt_creation_and_subject():
    """Test that 'email' is correctly mapped to 'sub'"""
    data = {"email": "test@ufl.edu", "role": "admin"}
    token = create_access_token(data, expires_delta=timedelta(minutes=15))
    
    # Decode manually
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    
    assert payload["sub"] == "test@ufl.edu"  # Important: Check email -> sub mapping
    assert payload["role"] == "admin"
    assert "exp" in payload