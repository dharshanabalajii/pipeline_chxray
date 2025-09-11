import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]:%(message)s:')

project_name = "cnnClassifier"

list_of_files =[
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html" #The templates folder is a common convention for storing HTML files that are rendered by the backend
]

for filepath in list_of_files:
    filepath=Path(filepath) #converts string to path object 
    filedir,filename=os.path.split(filepath)

    if filedir!="":
        os.makedirs(filedir,exist_ok=True) #do nothing if the folder already exists
        logging.info(f"Creating directory;{filedir} for file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0): #create the file inside the created folder , if the file does not exist or it is empty
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating file;{filename}")

    else:
        logging.info(f"{filename} already exists")