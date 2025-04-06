#!/usr/bin/env python3
"""
Generate a single snapshot analysis of the entire repository's current state.
This tool analyzes all Python files in the given directory without git history.
"""

import argparse
import os
from collections import defaultdict
from utils.commits.code_analyzer import analyze_python_file
from utils.commits.diagram_generator import generate_ascii_diagram, generate_graphviz_diagram
from utils.commits.design_patterns import DesignPatternDetector
from utils.commits.metrics import calculate_metrics, format_metrics
from utils.commits.colors import Colors
from utils.commits.html_report import generate_html_report
from utils.commits.cache import AnalysisCache
from multiprocessing import Pool, cpu_count
from utils.files.class_diagram_single_file import generate_single_file_diagram
from utils.files.associations_between_files import analyze_file_associations

def _create_module_dict():
    """Default factory for module dictionary."""
    return {
        'classes': set(),
        'imported_modules': set(),
        'exporting_to': set()
    }

def analyze_file_worker(args):
    """Worker function for parallel file analysis."""
    try:
        file_path, cache = args
        if file_path.endswith('.py'):
            # Create diagrams directory in the same directory as the file
            diagrams_dir = os.path.join(os.path.dirname(file_path), 'diagrams')
            os.makedirs(diagrams_dir, exist_ok=True)
            
            try:
                # Generate individual file diagram
                ascii_file, graphviz_file = generate_single_file_diagram(
                    file_path,
                    diagrams_dir
                )
                
                if ascii_file and graphviz_file:
                    print(f"Generated diagrams for {file_path}:")
                    print(f"  ASCII: {ascii_file}")
                    print(f"  GraphViz: {graphviz_file}")
                
                # Return the analysis results for the full repository diagram
                classes, modules = analyze_python_file(file_path, 'current', cache)
                if not classes:
                    print(f"Warning: No classes found in {file_path}")
                return classes, modules
                
            except Exception as e:
                print(f"Warning: Error analyzing file {file_path}: {str(e)}")
                return {}, {}
                
        return None, None
    except Exception as e:
        print(f"Error in worker for {args[0] if args else 'unknown file'}: {str(e)}")
        return {}, {}

def analyze_directory(directory_path, exclude_dirs=None, parallel=True, max_processes=0, cache=None):
    """
    Analyze all Python files in the given directory.
    
    Args:
        directory_path (str): Path to the directory to analyze
        exclude_dirs (list): List of directory patterns to exclude
        parallel (bool): Whether to use parallel processing
        max_processes (int): Maximum number of parallel processes (0 = auto)
        cache (AnalysisCache): Cache instance for storing analysis results
    
    Returns:
        tuple: (classes_dict, modules_dict)
    """
    exclude_dirs = exclude_dirs or []
    exclude_dirs.extend(['.git', 'env', 'venv', 'site-packages', '__pycache__', 'diagrams'])
    
    # Normalize and resolve the directory path
    directory_path = os.path.abspath(os.path.expanduser(directory_path))
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    print(f"\nScanning directory: {directory_path}")
    python_files = []
    
    for root, dirs, files in os.walk(directory_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if not any(exclude in d for exclude in exclude_dirs)]
        
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                python_files.append(full_path)
                print(f"Found Python file: {os.path.relpath(full_path, directory_path)}")
    
    if not python_files:
        print(f"No Python files found in {directory_path}")
        return {}, {}
    
    all_classes = {}
    all_modules = defaultdict(_create_module_dict)
    
    if parallel and len(python_files) > 1 and cpu_count() > 1:
        num_processes = min(max_processes if max_processes > 0 else cpu_count(), len(python_files))
        print(f"\nAnalyzing {len(python_files)} files using {num_processes} processes...")
        
        with Pool(processes=num_processes) as pool:
            worker_args = [(file_path, cache) for file_path in python_files]
            results = pool.map(analyze_file_worker, worker_args)
            
            for classes, modules in results:
                if classes and modules:
                    all_classes.update(classes)
                    for module_name, module_info in modules.items():
                        all_modules[module_name]['classes'].update(module_info['classes'])
                        all_modules[module_name]['imported_modules'].update(module_info['imported_modules'])
                        all_modules[module_name]['exporting_to'].update(module_info['exporting_to'])
    else:
        print(f"\nAnalyzing {len(python_files)} files sequentially...")
        for file_path in python_files:
            classes, modules = analyze_file_worker((file_path, cache))
            if classes and modules:
                all_classes.update(classes)
                for module_name, module_info in modules.items():
                    all_modules[module_name]['classes'].update(module_info['classes'])
                    all_modules[module_name]['imported_modules'].update(module_info['imported_modules'])
                    all_modules[module_name]['exporting_to'].update(module_info['exporting_to'])
    
    # Then analyze associations between files
    print("\nAnalyzing associations between files...")
    file_associations = analyze_file_associations(python_files)
    
    # Add file association information to the output
    for file_path, assoc_info in file_associations.items():
        module_name = os.path.basename(file_path).replace('.py', '')
        all_modules[module_name]['file_associations'] = assoc_info['associations']
        
        # Add association information to classes
        for class_name in assoc_info['classes']:
            if class_name in all_classes:
                all_classes[class_name]['file_associations'] = {
                    'file': file_path,
                    'associated_files': assoc_info['associations']
                }
    
    return all_classes, all_modules

def main():
    """Parse CLI arguments and run the analysis."""
    parser = argparse.ArgumentParser(description='Generate a single snapshot analysis of the entire repository')
    parser.add_argument('path', help='Path to the repository/directory to analyze')
    parser.add_argument('--output-dir', '-o', default='snapshot_analysis', help='Output directory for diagrams')
    parser.add_argument('--ascii-only', '-a', action='store_true', help='Generate only ASCII diagrams (no GraphViz)')
    parser.add_argument('--format', '-f', choices=['png', 'svg', 'pdf'], default='png', help='Output format for GraphViz diagrams')
    parser.add_argument('--show-modules', '-m', action='store_true', help='Show module dependencies in diagrams')
    parser.add_argument('--no-color', action='store_true', help='Disable colored ASCII output')
    parser.add_argument('--metrics', action='store_true', help='Calculate and output code quality metrics')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing of files')
    parser.add_argument('--max-processes', type=int, default=0, help='Maximum number of parallel processes (0 = auto)')
    parser.add_argument('--generate-html', action='store_true', help='Generate HTML report')
    parser.add_argument('--exclude-dirs', nargs='+', default=[], help='Additional directories to exclude from analysis')
    parser.add_argument('--inheritance-only', action='store_true', help='Show only inheritance relationships')
    parser.add_argument('--relationship-only', action='store_true', help='Show only class relationships')
    parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed class information')
    parser.add_argument('--cache-dir', default='.analysis_cache', help='Directory for storing analysis cache')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of file analysis results')
    
    args = parser.parse_args()
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a diagrams directory in the repository root
    repo_diagrams_dir = os.path.join(args.path, 'diagrams')
    os.makedirs(repo_diagrams_dir, exist_ok=True)
    
    # Initialize cache if enabled
    cache = None if args.no_cache else AnalysisCache(args.cache_dir)
    
    try:
        # Analyze the repository
        print(f"Starting analysis of: {args.path}")
        classes, modules = analyze_directory(
            args.path,
            exclude_dirs=args.exclude_dirs,
            parallel=not args.no_parallel,
            max_processes=args.max_processes,
            cache=cache
        )
        
        if not classes:
            print("No classes found in the repository.")
            return 1
        
        # Generate ASCII diagram with file associations
        print("Generating ASCII diagram...")
        ascii_diagram = generate_ascii_diagram(
            classes,
            modules if args.show_modules else None,
            use_color=not args.no_color,
            inheritance_only=args.inheritance_only,
            relationship_only=args.relationship_only,
            detailed=args.detailed,
            show_file_associations=True
        )
        
        # Save ASCII diagram
        ascii_file = os.path.join(args.output_dir, 'diagram.txt')
        with open(ascii_file, 'w', encoding='utf-8') as f:
            f.write(ascii_diagram)
        print(f"ASCII diagram saved to: {ascii_file}")
        
        # Generate GraphViz diagram if requested
        if not args.ascii_only:
            print("Generating GraphViz diagram...")
            graphviz_file = os.path.join(args.output_dir, 'diagram')
            generate_graphviz_diagram(
                classes,
                graphviz_file,
                modules if args.show_modules else None,
                args.format
            )
            print(f"GraphViz diagram saved to: {graphviz_file}.{args.format}")
        
        # Calculate and save metrics if requested
        if args.metrics:
            print("Calculating code metrics...")
            metrics = calculate_metrics(classes)
            metrics_formatted = format_metrics(classes, metrics)
            metrics_file = os.path.join(args.output_dir, 'metrics.txt')
            with open(metrics_file, 'w', encoding='utf-8') as f:
                f.write(metrics_formatted)
            print(f"Metrics saved to: {metrics_file}")
        
        # Generate HTML report if requested
        if args.generate_html:
            print("Generating HTML report...")
            snapshot_data = {'latest': classes}
            commits_data = [{'hexsha': 'latest', 'message': 'Current repository state', 
                           'author': {'name': 'Current State'}, 
                           'committed_datetime': 'N/A'}]
            report_file = generate_html_report(snapshot_data, commits_data, args.output_dir)
            print(f"HTML report saved to: {report_file}")
        
        # Save cache if enabled
        if cache:
            cache.save_cache()
            print(f"Analysis cache saved to: {args.cache_dir}")
        
        print(f"\nAnalysis complete. Results saved to: {os.path.abspath(args.output_dir)}")
        return 0
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())