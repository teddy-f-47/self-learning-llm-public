from dotenv import load_dotenv
import wandb
import os


"""
Might be useful if you have your wandb access token in the .env file.
Just run this Python script and it will keep the container logged in.
"""

load_dotenv()
wandb.login(key=os.getenv('wandb_key'))
