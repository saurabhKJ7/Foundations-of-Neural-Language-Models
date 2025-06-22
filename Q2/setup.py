#!/usr/bin/env python3
"""
Setup script for Hallucination Detection System
Helps users configure their OpenAI API key
"""

import os
import shutil

def setup_api_key():
    """Interactive setup for OpenAI API key"""
    print("🔧 Hallucination Detection System Setup")
    print("=" * 40)
    
    # Check if .env already exists
    if os.path.exists('.env'):
        print("✅ .env file already exists")
        with open('.env', 'r') as f:
            content = f.read()
            if 'OPENAI_API_KEY=' in content and 'your-openai-api-key-here' not in content:
                print("✅ API key appears to be configured")
                return True
    
    # Copy template if it doesn't exist
    if not os.path.exists('.env') and os.path.exists('.env.template'):
        print("📋 Creating .env from template...")
        shutil.copy('.env.template', '.env')
    
    print("\n🔑 OpenAI API Key Setup")
    print("You need an OpenAI API key to use this system.")
    print("Get one at: https://platform.openai.com/api-keys")
    
    api_key = input("\nEnter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided")
        return False
    
    if not api_key.startswith('sk-'):
        print("⚠️  Warning: API key should start with 'sk-'")
        confirm = input("Continue anyway? (y/N): ").strip().lower()
        if confirm != 'y':
            return False
    
    # Update .env file
    env_content = f"""# OpenAI API Configuration
OPENAI_API_KEY={api_key}

# Optional: Specify which model to use
# MODEL=gpt-3.5-turbo

# Optional: Set temperature for responses (0.0 to 1.0)  
# TEMPERATURE=0.1

# Optional: Maximum tokens per response
# MAX_TOKENS=150
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ API key saved to .env file")
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n📦 Checking dependencies...")
    
    missing_packages = []
    
    try:
        import openai
        print("✅ openai package found")
    except ImportError:
        missing_packages.append('openai')
    
    try:
        import dotenv
        print("✅ python-dotenv package found")
    except ImportError:
        missing_packages.append('python-dotenv')
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """Main setup function"""
    print("Starting setup process...\n")
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Setup failed: Missing dependencies")
        return
    
    # Setup API key
    if not setup_api_key():
        print("\n❌ Setup failed: API key not configured")
        return
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Run the system: python run.py")
    print("2. Or run components individually:")
    print("   - python ask_model.py")
    print("   - python validator.py")
    print("\nFiles you can check:")
    print("- kb.json (knowledge base)")
    print("- .env (your configuration)")

if __name__ == "__main__":
    main()