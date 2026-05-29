import os
import secrets
from dotenv import load_dotenv, set_key

# Load environment variables
load_dotenv()

ENV_FILE = '.env'

# Generate a new API key
def generate_api_key():
    return secrets.token_hex(16)

# Add the new API key to the ALLOWED_API_KEYS in the .env file
def add_api_key_to_env(new_key):
    allowed_keys = os.getenv('ALLOWED_API_KEYS', '')
    if allowed_keys:
        updated_keys = f"{allowed_keys},{new_key}"
    else:
        updated_keys = new_key
    set_key(ENV_FILE, 'ALLOWED_API_KEYS', updated_keys)

if __name__ == '__main__':
    new_api_key = generate_api_key()
    add_api_key_to_env(new_api_key)
    print(f"New API key generated and added to .env: {new_api_key}")
