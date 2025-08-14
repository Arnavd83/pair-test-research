"""
Simple .env file loader for JailbreakingLLMs project.
This loads environment variables from a .env file without requiring python-dotenv.
"""
import os
from pathlib import Path


def load_env_file(env_path=".env"):
    """Load environment variables from a .env file."""
    env_file = Path(env_path)
    
    if not env_file.exists():
        print(f"Warning: {env_path} file not found. Please create one from .env.example")
        return
    
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
                
            # Parse KEY=VALUE format
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # Set environment variable if not already set
                if key and value and value != f"your_{key.lower()}_here":
                    os.environ[key] = value
                    print(f"Loaded {key}")


def check_required_env_vars():
    """Check if required environment variables are set."""
    required_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY", 
        "GEMINI_API_KEY",
        "TOGETHER_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Warning: Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file or as system environment variables.")
    else:
        print("All required API keys are configured!")


if __name__ == "__main__":
    # Load .env file
    load_env_file()
    
    # Check if all required variables are set
    check_required_env_vars()
