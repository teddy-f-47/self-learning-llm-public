from huggingface_hub import login as hf_login
from dotenv import load_dotenv
import os


"""
Might be useful if you have your HF access token in the .env file.
Just run this Python script and it will keep the container logged in, so that you can access gated repos.
"""

load_dotenv()
hf_login(token=os.getenv('hf_personal_access_token'))
