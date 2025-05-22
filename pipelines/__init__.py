import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()