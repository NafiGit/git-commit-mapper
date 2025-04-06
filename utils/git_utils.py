import os
from git import Repo
import multiprocessing as mp
from utils.code_analyzer import analyze_python_file
from utils.cache import AnalysisCache
from utils.diagram_generator import generate_ascii_diagram, generate_plantuml_diagram
from utils.metrics import calculate_metrics, format_metrics
from utils.diff_utils import diff_snapshots, format_diff

def analyze_file_worker(args):
    """Worker function for parallel file analysis."""
    try:
        file_path, commit_id, cache = args
        if file_path.endswith('.py'):
            return analyze_python_file(file_path, commit_id, cache)
        return None, None
    except Exception as e:
        print(f"Error analyzing file {args[0]}: {str(e)}")
        return None, None

def analyze_commit(repo, commit, output_dir, ascii_only=False, graphviz_format='png', 
                  show_modules=False, use_color=True, calculate_code_metrics=False, 
                  cache=None, parallel=True, max_processes=0, max_files_per_process=100, 
                  exclude_dirs=None, inheritance_only=False, relationship_only=False, 
                  detailed=False):
    """Analyze a single commit."""
    original_head = repo.head.reference  # Store the original HEAD
    original_working_tree = repo.working_tree_dir
    temp_dir = None
    
    try:
        # Instead of checking out directly, use git worktree or create a temp clone
        import tempfile
        temp_dir = tempfile.mkdtemp()
        from git import Git
        
        # Create a worktree for the specific commit
        git = Git(repo.working_tree_dir)
        git.execute(['git', 'worktree', 'add', '--detach', temp_dir, commit.hexsha])
        
        # Now analyze files from the temp directory
        temp_repo = Repo(temp_dir)
        
        snapshot = {}
        all_modules = {}
        python_files = []
        
        for root, _, files in os.walk(temp_repo.working_dir):
            if '.git' in root or '/env/' in root or '/venv/' in root or 'site-packages' in root:
                continue
            if exclude_dirs and any(exclude_dir in root for exclude_dir in exclude_dirs):
                continue
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        if parallel and len(python_files) > 1 and mp.cpu_count() > 1:
            print(f"Analyzing {len(python_files)} files in parallel...")
            num_processes = min(max_processes if max_processes > 0 else mp.cpu_count(), len(python_files))
            chunk_size = min(max_files_per_process, max(1, len(python_files) // num_processes))
            file_chunks = [python_files[i:i + chunk_size] for i in range(0, len(python_files), chunk_size)]
            
            with mp.Pool(processes=num_processes) as pool:
                for chunk in file_chunks:
                    worker_args = [(file_path, commit.hexsha, cache) for file_path in chunk]
                    results = pool.map(analyze_file_worker, worker_args)
                    for result in results:
                        if result:
                            classes, modules = result
                            snapshot.update(classes)
                            all_modules.update(modules)
        else:
            print(f"Processing {len(python_files)} files sequentially...")
            for file_path in python_files:
                classes, modules = analyze_python_file(file_path, commit.hexsha, cache)
                snapshot.update(classes)
                all_modules.update(modules)
        
        diagram = generate_ascii_diagram(
            snapshot, 
            all_modules if show_modules else None, 
            use_color,
            inheritance_only=inheritance_only,
            relationship_only=relationship_only,
            detailed=detailed
        )
        diagram_file = os.path.join(output_dir, f"diagram_{commit.hexsha[:7]}.txt")
        with open(diagram_file, 'w', encoding='utf-8') as f:
            f.write(f"Commit: {commit.hexsha}\nDate: {commit.committed_datetime}\nAuthor: {commit.author.name} <{commit.author.email}>\nMessage: {commit.message.strip()}\n\n{diagram}")
        
        if calculate_code_metrics:
            metrics = calculate_metrics(snapshot)
            metrics_formatted = format_metrics(snapshot, metrics)
            metrics_file = os.path.join(output_dir, f"metrics_{commit.hexsha[:7]}.txt")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                f.write(f"Code Metrics for Commit: {commit.hexsha}\nDate: {commit.committed_datetime}\nAuthor: {commit.author.name} <{commit.author.email}>\nMessage: {commit.message.strip()}\n\n{metrics_formatted}")
        
        if not ascii_only:
            graphviz_file = os.path.join(output_dir, f"diagram_{commit.hexsha[:7]}")
            generate_plantuml_diagram(snapshot, graphviz_file, all_modules if show_modules else None, graphviz_format)
        
        return snapshot
    
    finally:
        # Cleanup: remove the temporary worktree
        if temp_dir:
            git = Git(repo.working_tree_dir)
            git.execute(['git', 'worktree', 'remove', '--force', temp_dir])

def analyze_commits(repo_path, output_dir, max_commits, ascii_only, graphviz_format, show_modules, 
                   use_color, calculate_code_metrics, cache_dir, parallel, max_processes, 
                   max_files_per_process, exclude_dirs, inheritance_only=False, 
                   relationship_only=False, detailed=False):
    """Analyze multiple commits and generate diffs."""
    repo = Repo(repo_path)
    if repo.bare:
        raise ValueError("Cannot analyze a bare repository")
    
    commits = list(repo.iter_commits(max_count=max_commits))
    commits.reverse()
    
    cache = AnalysisCache(cache_dir) if cache_dir else None
    original_branch = repo.active_branch.name
    
    snapshots = {}
    try:
        for commit in commits:
            snapshot = analyze_commit(
                repo, commit, output_dir, ascii_only, graphviz_format, show_modules,
                use_color, calculate_code_metrics, cache, parallel, max_processes,
                max_files_per_process, exclude_dirs, inheritance_only,
                relationship_only, detailed
            )
            snapshots[commit.hexsha] = snapshot
        
        for i in range(len(commits) - 1):
            old_commit = commits[i]
            new_commit = commits[i + 1]
            diff = diff_snapshots(snapshots[old_commit.hexsha], snapshots[new_commit.hexsha])
            diff_text = format_diff(diff, old_commit.hexsha, new_commit.hexsha)
            diff_file = os.path.join(output_dir, f"diff_{old_commit.hexsha[:7]}_{new_commit.hexsha[:7]}.txt")
            with open(diff_file, 'w', encoding='utf-8') as f:
                f.write(diff_text)
        
        if cache:
            cache.save_cache()
            print(f"Analysis cache saved to {cache_dir}")
        
        repo.git.checkout(original_branch)
        return snapshots, commits
    
    except Exception as e:
        repo.git.checkout(original_branch)
        raise e