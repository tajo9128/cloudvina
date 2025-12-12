import os
from huggingface_hub import HfApi, hf_hub_download
from typing import Optional

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "tajo9128/biodockify-data")

api = HfApi(token=HF_TOKEN)

def upload_file_to_hub(
    file_path: str,
    path_in_repo: str,
    repo_id: str = HF_DATASET_REPO,
    repo_type: str = "dataset"
) -> str:
    """
    Upload a file to Hugging Face Hub.
    Returns the URL of the uploaded file.
    """
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. Skipping upload.")
        return "mock_url_no_token"

    try:
        # Check if repo exists, if not create strictly as dataset
        # Note: In real app, might want to check existence first
        # api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
        
        future = api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"Upload {path_in_repo}"
        )
        # Construct URL (Web view)
        return f"https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}"
    except Exception as e:
        print(f"Failed to upload to HF Hub: {str(e)}")
        raise e

def download_file_from_hub(
    path_in_repo: str,
    repo_id: str = HF_DATASET_REPO,
    repo_type: str = "dataset"
) -> str:
    """
    Download file from Hub to local cache.
    Returns local path.
    """
    return hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type=repo_type,
        token=HF_TOKEN
    )
