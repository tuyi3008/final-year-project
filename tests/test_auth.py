# tests/test_auth.py
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock


class TestAuthFunctions(unittest.TestCase):
    """Authentication module unit tests"""
    
    def setUp(self):
        self.test_user = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "SecurePass123"
        }
    
    @patch('auth.get_users_collection')
    @patch('auth.mongodb')
    @patch('auth.get_db')
    @patch('auth.pwd_context.hash')
    def test_register_user_success(self, mock_hash, mock_get_db, mock_mongodb, mock_get_users_collection):
        """Test: User registration success"""
        mock_hash.return_value = "hashed_password"
        
        mock_mongodb.database = MagicMock()
        mock_mongodb.client = AsyncMock()
        mock_admin = AsyncMock()
        mock_admin.command = AsyncMock(return_value={"ok": 1})
        mock_mongodb.client.admin = mock_admin
        
        mock_users_collection = AsyncMock()
        mock_users_collection.find_one = AsyncMock(return_value=None)
        mock_insert_result = MagicMock()
        mock_insert_result.inserted_id = "123"
        mock_users_collection.insert_one = AsyncMock(return_value=mock_insert_result)
        mock_get_users_collection.return_value = mock_users_collection
        
        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db
        
        from auth import register_user, UserCreate
        
        user = UserCreate(
            username="testuser",
            email="test@example.com",
            password="SecurePass123"
        )
        
        result = asyncio.run(register_user(user))
        
        self.assertEqual(result["code"], 200)
        self.assertEqual(result["msg"], "Registration successful")
        self.assertEqual(result["username"], "testuser")
        self.assertEqual(result["email"], "test@example.com")
        mock_users_collection.insert_one.assert_called_once()
    
    @patch('auth.get_users_collection')
    @patch('auth.mongodb')
    @patch('auth.get_db')
    def test_register_user_duplicate(self, mock_get_db, mock_mongodb, mock_get_users_collection):
        """Test: Duplicate registration fails"""
        mock_mongodb.database = MagicMock()
        mock_mongodb.client = AsyncMock()
        mock_admin = AsyncMock()
        mock_admin.command = AsyncMock(return_value={"ok": 1})
        mock_mongodb.client.admin = mock_admin
        
        mock_users_collection = AsyncMock()
        mock_users_collection.find_one = AsyncMock(return_value={
            "_id": "existing_id",
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": "hashed"
        })
        mock_get_users_collection.return_value = mock_users_collection
        
        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db
        
        from auth import register_user, UserCreate
        
        user = UserCreate(
            username="testuser",
            email="test@example.com",
            password="SecurePass123"
        )
        
        with self.assertRaises(Exception) as context:
            asyncio.run(register_user(user))
        
        self.assertIn("Email already registered", str(context.exception))
        mock_users_collection.insert_one.assert_not_called()
    
    @patch('auth.get_users_collection')
    @patch('auth.mongodb')
    @patch('auth.get_db')
    def test_get_user_by_email(self, mock_get_db, mock_mongodb, mock_get_users_collection):
        """Test: Retrieve user by email"""
        mock_mongodb.database = MagicMock()
        mock_mongodb.client = AsyncMock()
        
        mock_users_collection = AsyncMock()
        mock_users_collection.find_one = AsyncMock(return_value={
            "_id": "user123",
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": "hashed",
            "total_xp": 100
        })
        mock_get_users_collection.return_value = mock_users_collection
        
        mock_db = AsyncMock()
        mock_get_db.return_value = mock_db
        
        from auth import get_user
        
        result = asyncio.run(get_user("test@example.com"))
        
        self.assertIsNotNone(result)
        self.assertEqual(result.username, "testuser")
        self.assertEqual(result.email, "test@example.com")
        self.assertEqual(result.total_xp, 100)
    
    def test_password_hashing(self):
        """Test: Password hashing and verification"""
        import auth
        
        password = "mysecret123"
        hashed = auth.pwd_context.hash(password)
        
        self.assertNotEqual(hashed, password)
        self.assertTrue(auth.pwd_context.verify(password, hashed))
        self.assertFalse(auth.pwd_context.verify("wrongpassword", hashed))
    
    def test_verify_password(self):
        """Test: Password verification function"""
        import auth
        
        password = "test123"
        hashed = auth.get_password_hash(password)
        
        self.assertTrue(auth.verify_password(password, hashed))
        self.assertFalse(auth.verify_password("wrong", hashed))


if __name__ == '__main__':
    unittest.main()