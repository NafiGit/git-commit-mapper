#!/usr/bin/env python3
"""
Git Commit Class Diagram and Communication Mapping Tool

This script analyzes the first 10 commits of a Git repository and creates ASCII diagrams
of classes and their communications (states, props, serializers, function calls, API calls).
It also maps changes between consecutive commits.
"""

import os
import ast
import sys
import argparse
from git import Repo, GitCommandError
from collections import defaultdict
import re

class CodeAnalyzer(ast.NodeVisitor):
    """Analyzes Python code to extract classes and their communications."""
    
    def __init__(self):
        self.classes = {}
        self.current_class = None
        self.current_method = None
        self.imports = {}
        
    def visit_Import(self, node):
        """Process import statements."""
        for name in node.names:
            self.imports[name.asname or name.name] = name.name
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        """Process from-import statements."""
        module = node.module
        for name in node.names:
            imported_name = name.name
            alias = name.asname or imported_name
            self.imports[alias] = f"{module}.{imported_name}" if module else imported_name
        self.generic_visit(node)
    
    def _get_name_safely(self, node):
        """Safely get the name from a node, handling various node types."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            # For attributes, recursively build the name
            prefix = self._get_name_safely(node.value)
            if prefix:
                return f"{prefix}.{node.attr}"
            return node.attr
        return ""
        
    def visit_ClassDef(self, node):
        """Process class definitions."""
        self.current_class = node.name
        self.classes[node.name] = {
            'methods': [],
            'calls': set(),
            'states': set(),
            'props': set(),
            'serializers': set(),
            'api_calls': set(),
            'parent_classes': []
        }
        
        # Extract parent classes
        for base in node.bases:
            base_name = self._get_name_safely(base)
            if base_name:
                self.classes[node.name]['parent_classes'].append(base_name)
        
        # Visit class body
        self.generic_visit(node)
        self.current_class = None
        
    def visit_FunctionDef(self, node):
        """Process function/method definitions."""
        prev_method = self.current_method
        self.current_method = node.name
        
        if self.current_class:
            self.classes[self.current_class]['methods'].append(node.name)
            
            # Check for self.state or self.props in arguments or function body
            for arg in node.args.args:
                if getattr(arg, 'arg', None) == 'self':
                    # This is a method with self parameter
                    pass
            
        self.generic_visit(node)
        self.current_method = prev_method
        
    def visit_Attribute(self, node):
        """Process attribute access (e.g., self.state, props, etc.)."""
        if self.current_class and isinstance(node.value, ast.Name):
            if node.value.id == 'self':
                attr_name = node.attr
                if attr_name == 'state' or attr_name.startswith('state_'):
                    self.classes[self.current_class]['states'].add(attr_name)
                elif attr_name == 'props' or attr_name.endswith('_props'):
                    self.classes[self.current_class]['props'].add(attr_name)
                elif 'serializer' in attr_name.lower():
                    self.classes[self.current_class]['serializers'].add(attr_name)
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Process function/method calls."""
        if not self.current_class:
            self.generic_visit(node)
            return
            
        # Handle API calls (requests.get, http.client, etc.)
        if isinstance(node.func, ast.Attribute):
            # Extract the function call name safely
            call_name = self._get_name_safely(node.func)
            if call_name:
                # Check if it's an API call based on naming
                parts = call_name.split('.')
                if parts and (parts[0] in ('requests', 'urllib', 'http') or 'api' in call_name.lower()):
                    self.classes[self.current_class]['api_calls'].add(call_name)
                elif not (len(parts) > 0 and parts[0] == 'self'):  # Skip self.method() calls
                    self.classes[self.current_class]['calls'].add(call_name)
        elif isinstance(node.func, ast.Name):
            # Direct function calls
            call_name = node.func.id
            if call_name in self.imports:
                self.classes[self.current_class]['calls'].add(call_name)
            elif 'api' in call_name.lower() or 'request' in call_name.lower():
                self.classes[self.current_class]['api_calls'].add(call_name)
            else:
                self.classes[self.current_class]['calls'].add(call_name)
                
        self.generic_visit(node)


def analyze_python_file(file_path):
    """Analyze a Python file to extract class information."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)
        analyzer = CodeAnalyzer()
        analyzer.visit(tree)
        return analyzer.classes
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")
        return {}


def generate_ascii_diagram(classes):
    """Generate a comprehensive ASCII diagram showing class relationships."""
    if not classes:
        return "No classes found."
    
    diagram = []
    method_map = defaultdict(list)
    connections = []
    serializer_connections = []

    # Build method map and inheritance relationships
    for cls_name, details in classes.items():
        for method in details['methods']:
            method_map[method].append(cls_name)
    
    # Create class boxes with inheritance
    diagram.append("CLASS STRUCTURE:")
    diagram.append("================")
    for cls_name, details in classes.items():
        # Class header with inheritance
        parents = details['parent_classes']
        class_header = f"╭─ {cls_name}"
        if parents:
            class_header += f" ({' ← '.join(parents)})"
        class_header += " ────────────────────╮"
        diagram.append(class_header)
        
        # Body content
        body = [
            f"│ {'Methods:':<16} {', '.join(details['methods']) or 'None'}",
            f"│ {'States:':<16} {', '.join(details['states']) or 'None'}",
            f"│ {'Props:':<16} {', '.join(details['props']) or 'None'}",
            f"│ {'Serializers:':<16} {', '.join(details['serializers']) or 'None'}"
        ]
        
        # Find connections
        for called in details['calls']:
            # Check if call matches a method in other classes
            for target_cls in classes:
                if cls_name == target_cls:
                    continue
                
                # Check if the call directly targets another class (ClassName.method)
                if called.startswith(target_cls + '.'):
                    connections.append((
                        cls_name, 
                        target_cls, 
                        f"calls: {called}",
                        "method"
                    ))
                    continue
                
                # Check if the call matches another class's method
                if called in classes[target_cls]['methods']:
                    connections.append((
                        cls_name, 
                        target_cls, 
                        f"calls: {called}()",
                        "method"
                    ))
        
        # API connections
        for api_call in details['api_calls']:
            connections.append((
                cls_name,
                "API",
                f"→ {api_call}",
                "api"
            ))
        
        # Serializer connections
        for serializer in details['serializers']:
            # Check if serializer name matches or refers to another class
            for target_cls in classes:
                if target_cls.lower() in serializer.lower():
                    serializer_connections.append((
                        cls_name,
                        target_cls,
                        f"serializes with: {serializer}",
                        "serializer"
                    ))
                    break
            else:
                serializer_connections.append((
                    cls_name,
                    serializer,
                    "serializes",
                    "serializer"
                ))

        diagram.extend(body)
        diagram.append("╰───────────────────────────────────────────────╯")
        diagram.append("")

    # Add communication map
    diagram.append("\nCLASS COMMUNICATIONS:")
    diagram.append("=====================")
    
    # Method calls between classes
    method_connections = [c for c in connections if c[3] == "method"]
    if method_connections:
        diagram.append("\nMethod Calls:")
        for src, dest, label, _ in sorted(method_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {src.ljust(15)} ───[{label}]──→ {dest}")
    
    # API calls
    api_connections = [c for c in connections if c[3] == "api"]
    if api_connections:
        diagram.append("\nAPI Communications:")
        for src, _, label, _ in sorted(api_connections, key=lambda x: x[0]):
            diagram.append(f"  {src.ljust(15)} ──{label}──→ External API")

    # Serializer relationships
    if serializer_connections:
        diagram.append("\nSerializer Usage:")
        for src, dest, label, _ in sorted(serializer_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {src.ljust(15)} ──[{label}]──→ {dest}")

    # If there are enough classes, add a class dependency graph
    if len(classes) > 1 and method_connections:
        diagram.append("\nCLASS DEPENDENCY GRAPH:")
        diagram.append("======================")
        diagram.append("")
        
        # Build a simplified graph of class dependencies
        graph = defaultdict(set)
        for src, dest, _, _ in method_connections:
            if dest != "API":
                graph[src].add(dest)
        
        # Generate a simple graphical representation
        drawn_classes = set()
        for cls_name in sorted(classes.keys()):
            if cls_name in drawn_classes:
                continue
                
            # Draw this class and its dependencies
            draw_class_tree(diagram, cls_name, graph, drawn_classes, "", True)
    
    return "\n".join(diagram)

def draw_class_tree(diagram, cls_name, graph, drawn_classes, prefix="", is_last=True):
    """Helper function to draw a tree representation of class dependencies."""
    if cls_name in drawn_classes:
        # If we've drawn this class before, just note it's a reference
        branch = "└── " if is_last else "├── "
        diagram.append(f"{prefix}{branch}{cls_name} (reference)")
        return
    
    # Mark this class as drawn
    drawn_classes.add(cls_name)
    
    # Draw the current class
    branch = "└── " if is_last else "├── "
    diagram.append(f"{prefix}{branch}{cls_name}")
    
    # Calculate new prefix for children
    new_prefix = prefix + ("    " if is_last else "│   ")
    
    # Draw dependencies
    dependencies = sorted(graph[cls_name])
    for i, dep in enumerate(dependencies):
        is_last_dep = (i == len(dependencies) - 1)
        draw_class_tree(diagram, dep, graph, drawn_classes, new_prefix, is_last_dep)


def diff_snapshots(old_snapshot, new_snapshot):
    """Generate a diff between two snapshots."""
    diff = {
        'added_classes': [],
        'removed_classes': [],
        'modified_classes': {}
    }
    
    # Find added/removed classes
    old_classes = set(old_snapshot.keys())
    new_classes = set(new_snapshot.keys())
    
    diff['added_classes'] = list(new_classes - old_classes)
    diff['removed_classes'] = list(old_classes - new_classes)
    
    # Compare common classes for changes
    for class_name in old_classes.intersection(new_classes):
        old_class = old_snapshot[class_name]
        new_class = new_snapshot[class_name]
        
        changes = {}
        
        # Compare methods
        added_methods = set(new_class['methods']) - set(old_class['methods'])
        removed_methods = set(old_class['methods']) - set(new_class['methods'])
        if added_methods or removed_methods:
            changes['methods'] = {
                'added': list(added_methods),
                'removed': list(removed_methods)
            }
        
        # Compare calls
        added_calls = new_class['calls'] - old_class['calls']
        removed_calls = old_class['calls'] - new_class['calls']
        if added_calls or removed_calls:
            changes['calls'] = {
                'added': list(added_calls),
                'removed': list(removed_calls)
            }
        
        # Compare API calls
        added_api = new_class['api_calls'] - old_class['api_calls']
        removed_api = old_class['api_calls'] - new_class['api_calls']
        if added_api or removed_api:
            changes['api_calls'] = {
                'added': list(added_api),
                'removed': list(removed_api)
            }
        
        # Compare state and props
        added_states = new_class['states'] - old_class['states']
        removed_states = old_class['states'] - new_class['states']
        if added_states or removed_states:
            changes['states'] = {
                'added': list(added_states),
                'removed': list(removed_states)
            }
            
        added_props = new_class['props'] - old_class['props']
        removed_props = old_class['props'] - new_class['props']
        if added_props or removed_props:
            changes['props'] = {
                'added': list(added_props),
                'removed': list(removed_props)
            }
        
        # Only add to diff if there were changes
        if changes:
            diff['modified_classes'][class_name] = changes
    
    return diff


def format_diff(diff, old_commit_id, new_commit_id):
    """Format a diff as a readable ASCII text."""
    lines = []
    lines.append(f"DIFF BETWEEN {old_commit_id[:7]} AND {new_commit_id[:7]}")
    lines.append("=" * 50)
    lines.append("")
    
    # Added classes
    if diff['added_classes']:
        lines.append("ADDED CLASSES:")
        for class_name in sorted(diff['added_classes']):
            lines.append(f"  + {class_name}")
        lines.append("")
    
    # Removed classes
    if diff['removed_classes']:
        lines.append("REMOVED CLASSES:")
        for class_name in sorted(diff['removed_classes']):
            lines.append(f"  - {class_name}")
        lines.append("")
    
    # Modified classes
    if diff['modified_classes']:
        lines.append("MODIFIED CLASSES:")
        for class_name, changes in sorted(diff['modified_classes'].items()):
            lines.append(f"  * {class_name}:")
            
            # Methods changes
            if 'methods' in changes:
                if changes['methods']['added']:
                    lines.append("      Added methods:")
                    for method in sorted(changes['methods']['added']):
                        lines.append(f"        + {method}()")
                if changes['methods']['removed']:
                    lines.append("      Removed methods:")
                    for method in sorted(changes['methods']['removed']):
                        lines.append(f"        - {method}()")
            
            # Call changes
            if 'calls' in changes:
                if changes['calls']['added']:
                    lines.append("      Added calls:")
                    for call in sorted(changes['calls']['added']):
                        lines.append(f"        + {call}")
                if changes['calls']['removed']:
                    lines.append("      Removed calls:")
                    for call in sorted(changes['calls']['removed']):
                        lines.append(f"        - {call}")
            
            # API call changes
            if 'api_calls' in changes:
                if changes['api_calls']['added']:
                    lines.append("      Added API calls:")
                    for api in sorted(changes['api_calls']['added']):
                        lines.append(f"        + {api}")
                if changes['api_calls']['removed']:
                    lines.append("      Removed API calls:")
                    for api in sorted(changes['api_calls']['removed']):
                        lines.append(f"        - {api}")
            
            # State changes
            if 'states' in changes:
                if changes['states']['added']:
                    lines.append("      Added states:")
                    for state in sorted(changes['states']['added']):
                        lines.append(f"        + {state}")
                if changes['states']['removed']:
                    lines.append("      Removed states:")
                    for state in sorted(changes['states']['removed']):
                        lines.append(f"        - {state}")
            
            # Props changes
            if 'props' in changes:
                if changes['props']['added']:
                    lines.append("      Added props:")
                    for prop in sorted(changes['props']['added']):
                        lines.append(f"        + {prop}")
                if changes['props']['removed']:
                    lines.append("      Removed props:")
                    for prop in sorted(changes['props']['removed']):
                        lines.append(f"        - {prop}")
            
            lines.append("")
    
    return "\n".join(lines)


def analyze_commit(repo, commit, output_dir):
    """Analyze a single commit and return a snapshot of the codebase."""
    print(f"Analyzing commit {commit.hexsha[:7]}: {commit.message.strip()}")
    
    # Checkout the commit
    repo.git.checkout(commit.hexsha, force=True)
    
    snapshot = {}
    
    # Analyze Python files
    for root, _, files in os.walk(repo.working_dir):
        for file in files:
            # Skip files in .git directory and virtual environment directories
            if '.git' in root or '/env/' in root or '/venv/' in root or 'site-packages' in root:
                continue
                
            file_path = os.path.join(root, file)
            
            # Only analyze Python files
            if file.endswith('.py'):
                classes = analyze_python_file(file_path)
                snapshot.update(classes)
    
    # Generate and save ASCII diagram
    diagram = generate_ascii_diagram(snapshot)
    diagram_file = os.path.join(output_dir, f"diagram_{commit.hexsha[:7]}.txt")
    
    with open(diagram_file, 'w', encoding='utf-8') as f:
        f.write(f"Commit: {commit.hexsha}\n")
        f.write(f"Date: {commit.committed_datetime}\n")
        f.write(f"Author: {commit.author.name} <{commit.author.email}>\n")
        f.write(f"Message: {commit.message.strip()}\n\n")
        f.write(diagram)
    
    return snapshot


def main():
    """Main function to process the repository."""
    parser = argparse.ArgumentParser(description='Generate class diagrams and communication maps from Git commits')
    parser.add_argument('repo_path', help='Path to the Git repository')
    parser.add_argument('--output-dir', '-o', default='diagrams', help='Output directory for diagrams and diffs')
    parser.add_argument('--max-commits', '-n', type=int, default=10, help='Maximum number of commits to analyze')
    parser.add_argument('--skip-env', '-s', action='store_true', help='Skip virtual environment directories')
    args = parser.parse_args()
    
    repo_path = args.repo_path
    output_dir = args.output_dir
    max_commits = args.max_commits
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        repo = Repo(repo_path)
        if repo.bare:
            print("Cannot analyze a bare repository")
            return 1
    except Exception as e:
        print(f"Error opening repository: {e}")
        return 1
    
    # Get the list of commits (oldest first)
    try:
        commits = list(repo.iter_commits(max_count=max_commits))
        commits.reverse()  # Make list go from oldest to newest
    except Exception as e:
        print(f"Error retrieving commits: {e}")
        return 1
    
    # Save the original branch/commit to return to
    original_branch = repo.active_branch.name
    
    try:
        snapshots = {}
        
        # Process each commit
        for commit in commits:
            snapshot = analyze_commit(repo, commit, output_dir)
            snapshots[commit.hexsha] = snapshot
        
        # Generate diffs between consecutive commits
        for i in range(len(commits) - 1):
            old_commit = commits[i]
            new_commit = commits[i + 1]
            
            diff = diff_snapshots(snapshots[old_commit.hexsha], snapshots[new_commit.hexsha])
            diff_text = format_diff(diff, old_commit.hexsha, new_commit.hexsha)
            
            diff_file = os.path.join(output_dir, f"diff_{old_commit.hexsha[:7]}_{new_commit.hexsha[:7]}.txt")
            with open(diff_file, 'w', encoding='utf-8') as f:
                f.write(diff_text)
        
        # Return to the original branch
        repo.git.checkout(original_branch)
        
        print(f"\nAnalysis complete. Results saved to {os.path.abspath(output_dir)}")
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        # Try to return to the original branch
        try:
            repo.git.checkout(original_branch)
        except:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main()) 