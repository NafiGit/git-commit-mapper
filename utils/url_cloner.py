import os
from git import Repo
import multiprocessing as mp
import shutil
from git.exc import GitCommandError

def clone_github_repo(github_url, target_dir="current_repo"):
    """
    Clone a GitHub repository into the specified target directory.
    
    Args:
        github_url (str): The URL of the GitHub repository to clone
                          (e.g., 'https://github.com/username/repo.git')
        target_dir (str): The directory where the repository will be cloned.
                          Defaults to "current_repo".
    
    Returns:
        Repo: The cloned repository object if successful, None otherwise.
    
    Raises:
        ValueError: If the github_url is not valid or empty
        GitCommandError: If the clone operation fails
    """
    if not github_url or not github_url.strip():
        raise ValueError("GitHub URL cannot be empty")
    
    # Create absolute path for target directory
    target_path = os.path.abspath(target_dir)
    
    # Check if the directory already exists, remove if it does
    if os.path.exists(target_path):
        print(f"Target directory '{target_path}' already exists. Removing...")
        shutil.rmtree(target_path)
    
    # Create the directory
    os.makedirs(target_path, exist_ok=True)
    
    try:
        print(f"Cloning {github_url} into {target_path}...")
        repo = Repo.clone_from(github_url, target_path)
        print(f"Successfully cloned repository to {target_path}")
        return repo
    except GitCommandError as e:
        print(f"Error cloning repository: {e}")
        # Clean up the directory in case of error
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
        raise