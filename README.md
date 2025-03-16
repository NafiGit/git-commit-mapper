# Git Commit Class Diagram and Communication Mapping Tool

This tool analyzes the first 10 commits of a Git repository and creates ASCII diagrams showing classes and their communications (including states, props, serializers, function calls, and API calls). It also generates diffs between consecutive commits to track how the code structure evolves over time.

## Features

- Analyzes Python code in Git repositories
- Identifies classes, methods, and their relationships
- Detects states, props, serializers, and API calls
- Generates ASCII class diagrams for each commit
- Creates diffs showing changes between consecutive commits
- Supports the first 10 commits by default (configurable)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/git_commit_mapping_tool.git
   cd git_commit_mapping_tool
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the tool on a Git repository:

```
python git_commit_mapper.py /path/to/repository [--output-dir OUTPUT_DIR] [--max-commits MAX_COMMITS]
```

### Arguments

- `repo_path`: Path to the Git repository to analyze
- `--output-dir`, `-o`: Directory where output files will be saved (default: "diagrams")
- `--max-commits`, `-n`: Maximum number of commits to analyze (default: 10)

### Output

The tool generates the following files in the output directory:

- `diagram_<commit-hash>.txt`: ASCII diagram for each commit
- `diff_<old-hash>_<new-hash>.txt`: Diff between consecutive commits

## Example Output

### Class Diagram

```
CLASS DIAGRAM:
=============

+--------------------+
| UserController     |
+--------------------+
| get_user()         |
| update_user()      |
+--------------------+

+--------------------+
| UserService        |
+--------------------+
| find_by_id()       |
| save()             |
| validate()         |
+--------------------+

CONNECTIONS:
============

UserController calls:
  └─→ UserService.find_by_id
  └─→ UserService.save

UserController API calls:
  └─→ API: requests.get
```

### Diff Output

```
DIFF BETWEEN abcd123 AND efgh456
==================================================

ADDED CLASSES:
  + Logger

MODIFIED CLASSES:
  * UserController:
      Added methods:
        + delete_user()
      Added calls:
        + Logger.log
```

## Limitations

- Currently only analyzes Python files
- May not detect all dynamic code patterns
- Simple ASCII visualization (no complex relationships)

## License

MIT 