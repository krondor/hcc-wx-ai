"""
watsonx.ai Setup Verification Script
Houston Community College - watsonx.ai Learning Hub

This script verifies that your watsonx.ai environment is properly configured.
Run this after completing the setup instructions in GETTING_STARTED.md

Usage:
    python test_setup.py
"""

import sys
import os
from typing import Tuple

def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_status(check: str, passed: bool, message: str = "") -> None:
    """Print a status message with checkmark or X."""
    symbol = "✅" if passed else "❌"
    print(f"{symbol} {check}")
    if message:
        print(f"   → {message}")

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is 3.9 or higher."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor}.{version.micro} (3.9+ required)"

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, f"{package_name} {version}"
    except ImportError:
        return False, f"{package_name} not installed"

def check_env_file() -> Tuple[bool, str]:
    """Check if .env file exists."""
    if os.path.exists('.env'):
        return True, ".env file found"
    return False, ".env file not found (copy from .env.example)"

def check_credentials() -> Tuple[bool, str]:
    """Check if credentials are configured."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('WATSONX_API_KEY')
        project_id = os.getenv('WATSONX_PROJECT_ID')
        url = os.getenv('WATSONX_URL')
        
        if not api_key or api_key == 'your_api_key_here':
            return False, "WATSONX_API_KEY not configured in .env"
        if not project_id or project_id == 'your_project_id_here':
            return False, "WATSONX_PROJECT_ID not configured in .env"
        if not url:
            return False, "WATSONX_URL not configured in .env"
        
        return True, "All credentials configured"
    except Exception as e:
        return False, f"Error loading credentials: {str(e)}"

def test_watsonx_connection() -> Tuple[bool, str]:
    """Test connection to watsonx.ai."""
    try:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from dotenv import load_dotenv
        
        load_dotenv()
        
        credentials = Credentials(
            url=os.getenv('WATSONX_URL'),
            api_key=os.getenv('WATSONX_API_KEY')
        )
        
        model = ModelInference(
            model_id="ibm/granite-13b-instruct-v2",
            credentials=credentials,
            project_id=os.getenv('WATSONX_PROJECT_ID')
        )
        
        # Try a simple generation
        response = model.generate_text(
            prompt="Say 'Hello from watsonx.ai!'",
            params={"max_new_tokens": 10}
        )
        
        if response:
            return True, "Successfully connected and generated text"
        return False, "Connection succeeded but no response received"
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "Unauthorized" in error_msg:
            return False, "Authentication failed - check your API key"
        elif "404" in error_msg or "not found" in error_msg.lower():
            return False, "Project not found - check your Project ID"
        else:
            return False, f"Connection error: {error_msg[:100]}"

def main():
    """Run all setup verification checks."""
    print_header("watsonx.ai Setup Verification")
    print("Houston Community College - watsonx.ai Learning Hub")
    print("\nThis script will verify your setup is correct.\n")
    
    all_passed = True
    
    # Check Python version
    print_header("1. Python Environment")
    passed, message = check_python_version()
    print_status("Python version", passed, message)
    all_passed = all_passed and passed
    
    # Check required packages
    print_header("2. Required Packages")
    
    packages = [
        ("ibm-watsonx-ai", "ibm_watsonx_ai"),
        ("python-dotenv", "dotenv"),
        ("jupyter", "jupyter"),
    ]
    
    for package_name, import_name in packages:
        passed, message = check_package(package_name, import_name)
        print_status(package_name, passed, message)
        all_passed = all_passed and passed
    
    # Check configuration files
    print_header("3. Configuration Files")
    
    passed, message = check_env_file()
    print_status(".env file", passed, message)
    all_passed = all_passed and passed
    
    if passed:
        passed, message = check_credentials()
        print_status("Credentials", passed, message)
        all_passed = all_passed and passed
    
    # Test watsonx.ai connection
    if all_passed:
        print_header("4. watsonx.ai Connection Test")
        print("Testing connection to watsonx.ai... (this may take a moment)")
        passed, message = test_watsonx_connection()
        print_status("Connection test", passed, message)
        all_passed = all_passed and passed
    else:
        print_header("4. watsonx.ai Connection Test")
        print("⏭️  Skipping connection test (fix previous issues first)")
    
    # Final summary
    print_header("Summary")
    
    if all_passed:
        print("\n🎉 SUCCESS! Your watsonx.ai environment is properly configured!")
        print("\n📚 Next Steps:")
        print("   1. Read GETTING_STARTED.md for your first tutorial")
        print("   2. Explore notebooks in the notebooks/ directory")
        print("   3. Start with the Beginner learning path")
        print("\n🚀 Ready to start learning!")
        return 0
    else:
        print("\n⚠️  SETUP INCOMPLETE - Please fix the issues above")
        print("\n📖 Troubleshooting:")
        print("   1. Review GETTING_STARTED.md for setup instructions")
        print("   2. Check the Troubleshooting section in README.md")
        print("   3. Ensure all packages are installed: pip install -r requirements.txt")
        print("   4. Verify your .env file has correct credentials")
        print("\n💡 Need help? Contact your instructor or check the documentation")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup verification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("Please report this issue to your instructor")
        sys.exit(1)

# Made with Bob
