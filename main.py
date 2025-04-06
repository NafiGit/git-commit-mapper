#!/usr/bin/env python3
"""
Main entry point for the Git Commit Class Diagram and Communication Mapping Tool.
"""

import sys
import argparse
import os
from utils.git_utils import analyze_commits
from utils.html_report import generate_html_report

def main():
    """Parse CLI arguments and orchestrate the analysis."""
    parser = argparse.ArgumentParser(description='Generate class diagrams and communication maps from Git commits')
    parser.add_argument('repo_path', help='Path to the Git repository')
    parser.add_argument('--output-dir', '-o', default='diagrams', help='Output directory for diagrams and diffs')
    parser.add_argument('--max-commits', '-n', type=int, default=10, help='Maximum number of commits to analyze')
    parser.add_argument('--skip-env', '-s', action='store_true', help='Skip virtual environment directories')
    parser.add_argument('--ascii-only', '-a', action='store_true', help='Generate only ASCII diagrams (no GraphViz)')
    parser.add_argument('--format', '-f', choices=['png', 'svg', 'pdf'], default='png', help='Output format for GraphViz diagrams')
    parser.add_argument('--show-modules', '-m', action='store_true', help='Show module dependencies in diagrams')
    parser.add_argument('--no-color', action='store_true', help='Disable colored ASCII output')
    parser.add_argument('--metrics', action='store_true', help='Calculate and output code quality metrics')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching of file analysis results')
    parser.add_argument('--cache-dir', default='.analysis_cache', help='Directory for storing analysis cache')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing of files')
    parser.add_argument('--max-processes', type=int, default=0, help='Maximum number of parallel processes (0 = auto)')
    parser.add_argument('--max-files-per-process', type=int, default=100, help='Maximum number of files per process')
    parser.add_argument('--generate-html', action='store_true', help='Generate HTML report with interactive diagrams')
    parser.add_argument('--exclude-dirs', nargs='+', default=[], help='Additional directories to exclude from analysis')
    parser.add_argument('--inheritance-only', action='store_true', help='Show only inheritance relationships')
    parser.add_argument('--relationship-only', action='store_true', help='Show only class relationships')
    parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed class information')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Analyze commits and get snapshots
    snapshots, commits = analyze_commits(
        repo_path=args.repo_path,
        output_dir=args.output_dir,
        max_commits=args.max_commits,
        ascii_only=args.ascii_only,
        graphviz_format=args.format,
        show_modules=args.show_modules,
        use_color=not args.no_color,
        calculate_code_metrics=args.metrics,
        cache_dir=args.cache_dir if not args.no_cache else None,
        parallel=not args.no_parallel,
        max_processes=args.max_processes,
        max_files_per_process=args.max_files_per_process,
        exclude_dirs=args.exclude_dirs,
        inheritance_only=args.inheritance_only,
        relationship_only=args.relationship_only,
        detailed=args.detailed
    )

    # Generate HTML report if requested
    if args.generate_html:
        print("Generating HTML report...")
        generate_html_report(snapshots, commits, args.output_dir)

    print(f"\nAnalysis complete. Results saved to {os.path.abspath(args.output_dir)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())