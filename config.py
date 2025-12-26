import os
from dotenv import load_dotenv

load_dotenv()

NAME = "WhisperFlow"
VERSION = "v0.1.0(dev)"
AUTHOR = "Rupnil Codes"

GROQ_API = os.environ['GROQ_API']