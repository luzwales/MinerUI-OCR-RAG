# Create auth/windows_auth.py
"""
Windows Domain Authentication Integration
"""

import ldap3
from typing import Optional, Tuple
from loguru import logger

def authenticate_domain_user(username: str, password: str, domain: str = None) -> Optional[dict]:
    """
    Authenticate user against Windows domain
    
    Args:
        username: Domain username
        password: User password
        domain: Domain name (optional, can be part of username)
        
    Returns:
        dict: User information if authentication successful, None otherwise
    """
    try:
        # Parse username (could be in format domain\\username or username@domain.com)
        if '\\' in username:
            domain, username = username.split('\\', 1)
        elif '@' in username:
            username, domain = username.split('@', 1)
            
        # Configure LDAP connection
        server = ldap3.Server(f"ldap://{domain}" if domain else "ldap://domain-controller")
        conn = ldap3.Connection(server, user=f"{domain}\\{username}", password=password, auto_bind=True)
        
        # Search for user details
        conn.search(
            search_base='DC=domain,DC=com',  # Adjust to your domain
            search_filter=f'(sAMAccountName={username})',
            attributes=['displayName', 'mail', 'sAMAccountName']
        )
        
        if conn.entries:
            entry = conn.entries[0]
            return {
                'username': str(entry.sAMAccountName),
                'email': str(entry.mail) if 'mail' in entry else f"{username}@{domain}",
                'full_name': str(entry.displayName) if 'displayName' in entry else username
            }
            
        return None
        
    except Exception as e:
        logger.error(f"Domain authentication failed for {username}: {e}")
        return None