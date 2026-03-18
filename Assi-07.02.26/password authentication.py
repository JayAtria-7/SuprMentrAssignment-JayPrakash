"""
install bcrypt==4.1.2
Secure Password Authentication System
======================================
This module provides a robust password authentication system with:
- Password hashing using bcrypt
- Password strength validation
- Account lockout after failed attempts
- Secure password storage
- Password reset functionality
"""

import bcrypt
import re
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple


class PasswordAuthenticator:
    """
    A secure password authentication system with best practices.
    """
    
    def __init__(self, storage_file: str = "users.json", max_attempts: int = 5):
        """
        Initialize the password authenticator.
        
        Args:
            storage_file: Path to the JSON file for storing user data
            max_attempts: Maximum login attempts before account lockout
        """
        self.storage_file = storage_file
        self.max_attempts = max_attempts
        self.lockout_duration = timedelta(minutes=15)
        self._load_users()
    
    def _load_users(self) -> None:
        """Load users from storage file."""
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r') as f:
                self.users = json.load(f)
        else:
            self.users = {}
    
    def _save_users(self) -> None:
        """Save users to storage file."""
        with open(self.storage_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def validate_password_strength(self, password: str) -> Tuple[bool, str]:
        """
        Validate password strength against security requirements.
        
        Args:
            password: The password to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if len(password) > 128:
            return False, "Password must not exceed 128 characters"
        
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        if not re.search(r'\d', password):
            return False, "Password must contain at least one digit"
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"
        
        # Check for common passwords (basic check)
        common_passwords = ['password', '12345678', 'qwerty', 'admin123', 'password123']
        if password.lower() in common_passwords:
            return False, "Password is too common. Please choose a stronger password"
        
        return True, "Password is strong"
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt with salt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password as string
        """
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Plain text password
            hashed_password: Hashed password to compare against
            
        Returns:
            True if password matches, False otherwise
        """
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def is_account_locked(self, username: str) -> Tuple[bool, Optional[str]]:
        """
        Check if an account is locked due to failed login attempts.
        
        Args:
            username: Username to check
            
        Returns:
            Tuple of (is_locked, unlock_time_message)
        """
        if username not in self.users:
            return False, None
        
        user_data = self.users[username]
        
        if user_data.get('locked_until'):
            locked_until = datetime.fromisoformat(user_data['locked_until'])
            if datetime.now() < locked_until:
                remaining = locked_until - datetime.now()
                minutes = int(remaining.total_seconds() / 60)
                return True, f"Account locked. Try again in {minutes} minutes"
            else:
                # Unlock the account
                user_data['failed_attempts'] = 0
                user_data['locked_until'] = None
                self._save_users()
        
        return False, None
    
    def register_user(self, username: str, password: str) -> Tuple[bool, str]:
        """
        Register a new user with username and password.
        
        Args:
            username: Username for the new account
            password: Password for the new account
            
        Returns:
            Tuple of (success, message)
        """
        # Validate username
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters long"
        
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return False, "Username can only contain letters, numbers, and underscores"
        
        # Check if user already exists
        if username in self.users:
            return False, "Username already exists"
        
        # Validate password strength
        is_valid, message = self.validate_password_strength(password)
        if not is_valid:
            return False, message
        
        # Hash password and store user
        hashed_password = self.hash_password(password)
        self.users[username] = {
            'password': hashed_password,
            'created_at': datetime.now().isoformat(),
            'failed_attempts': 0,
            'locked_until': None,
            'last_login': None
        }
        self._save_users()
        
        return True, "User registered successfully"
    
    def authenticate(self, username: str, password: str) -> Tuple[bool, str]:
        """
        Authenticate a user with username and password.
        
        Args:
            username: Username to authenticate
            password: Password to verify
            
        Returns:
            Tuple of (success, message)
        """
        # Check if user exists
        if username not in self.users:
            return False, "Invalid username or password"
        
        # Check if account is locked
        is_locked, lock_message = self.is_account_locked(username)
        if is_locked:
            return False, lock_message
        
        user_data = self.users[username]
        
        # Verify password
        if self.verify_password(password, user_data['password']):
            # Successful login
            user_data['failed_attempts'] = 0
            user_data['last_login'] = datetime.now().isoformat()
            self._save_users()
            return True, "Authentication successful"
        else:
            # Failed login
            user_data['failed_attempts'] = user_data.get('failed_attempts', 0) + 1
            
            if user_data['failed_attempts'] >= self.max_attempts:
                # Lock the account
                user_data['locked_until'] = (datetime.now() + self.lockout_duration).isoformat()
                self._save_users()
                return False, f"Account locked due to {self.max_attempts} failed attempts. Try again in 15 minutes"
            
            self._save_users()
            remaining = self.max_attempts - user_data['failed_attempts']
            return False, f"Invalid username or password. {remaining} attempts remaining"
    
    def change_password(self, username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
        """
        Change a user's password.
        
        Args:
            username: Username of the account
            old_password: Current password
            new_password: New password to set
            
        Returns:
            Tuple of (success, message)
        """
        # Authenticate with old password
        success, message = self.authenticate(username, old_password)
        if not success:
            return False, "Current password is incorrect"
        
        # Validate new password
        is_valid, message = self.validate_password_strength(new_password)
        if not is_valid:
            return False, message
        
        # Check if new password is same as old
        if old_password == new_password:
            return False, "New password must be different from old password"
        
        # Update password
        user_data = self.users[username]
        user_data['password'] = self.hash_password(new_password)
        user_data['failed_attempts'] = 0
        self._save_users()
        
        return True, "Password changed successfully"
    
    def delete_user(self, username: str) -> Tuple[bool, str]:
        """
        Delete a user account.
        
        Args:
            username: Username to delete
            
        Returns:
            Tuple of (success, message)
        """
        if username not in self.users:
            return False, "User not found"
        
        del self.users[username]
        self._save_users()
        return True, "User deleted successfully"


def main():
    """
    Interactive demo of the password authentication system.
    """
    auth = PasswordAuthenticator()
    
    print("=" * 60)
    print("SECURE PASSWORD AUTHENTICATION SYSTEM")
    print("=" * 60)
    
    while True:
        print("\n--- MENU ---")
        print("1. Register new user")
        print("2. Login")
        print("3. Change password")
        print("4. Delete user")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            print("\n--- REGISTER NEW USER ---")
            username = input("Enter username: ").strip()
            password = input("Enter password: ").strip()
            confirm_password = input("Confirm password: ").strip()
            
            if password != confirm_password:
                print("❌ Passwords do not match!")
                continue
            
            success, message = auth.register_user(username, password)
            if success:
                print(f"✅ {message}")
            else:
                print(f"❌ {message}")
        
        elif choice == '2':
            print("\n--- LOGIN ---")
            username = input("Enter username: ").strip()
            password = input("Enter password: ").strip()
            
            success, message = auth.authenticate(username, password)
            if success:
                print(f"✅ {message}")
                print(f"Welcome, {username}!")
            else:
                print(f"❌ {message}")
        
        elif choice == '3':
            print("\n--- CHANGE PASSWORD ---")
            username = input("Enter username: ").strip()
            old_password = input("Enter current password: ").strip()
            new_password = input("Enter new password: ").strip()
            confirm_password = input("Confirm new password: ").strip()
            
            if new_password != confirm_password:
                print("❌ New passwords do not match!")
                continue
            
            success, message = auth.change_password(username, old_password, new_password)
            if success:
                print(f"✅ {message}")
            else:
                print(f"❌ {message}")
        
        elif choice == '4':
            print("\n--- DELETE USER ---")
            username = input("Enter username to delete: ").strip()
            confirm = input(f"Are you sure you want to delete '{username}'? (yes/no): ").strip().lower()
            
            if confirm == 'yes':
                success, message = auth.delete_user(username)
                if success:
                    print(f"✅ {message}")
                else:
                    print(f"❌ {message}")
            else:
                print("Deletion cancelled")
        
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()