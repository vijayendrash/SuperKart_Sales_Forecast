from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "vijayendras/superkart-sales-data"
repo_type = "dataset"


token = os.getenv("HF_TOKEN")
api = HfApi(token=token)

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset...")

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=token
    )

    print("Dataset created.")

api.upload_folder(
    folder_path="superkart_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
    token=token
)

print("Upload completed.")
