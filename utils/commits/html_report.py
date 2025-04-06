from datetime import datetime
from utils.commits.diagram_generator import generate_ascii_diagram
from utils.commits.metrics import calculate_metrics, format_metrics
from utils.commits.diff_utils import diff_snapshots, format_diff

def generate_html_report(snapshots, commits, output_dir):
    """Generate an interactive HTML report."""
    report_file = f"{output_dir}/class_analysis_report.html"
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Code Architecture Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        .commit-section {{ display: none; }}
        .commit-section.active {{ display: block; }}
        .commit-btn {{ margin: 5px; padding: 10px; cursor: pointer; }}
        .commit-btn.active {{ background-color: #ddd; }}
        pre {{ background-color: #f5f5f5; padding: 10px; }}
    </style>
</head>
<body>
    <h1>Code Architecture Analysis Report</h1>
    <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <div class="commit-nav">
"""
    for i, commit in enumerate(commits):
        active = " active" if i == 0 else ""
        html += f'        <button class="commit-btn{active}" onclick="showCommit({i})">{commit.hexsha[:7]}</button>\n'
    
    html += "    </div>\n    <div id=\"commit-sections\">\n"
    for i, commit in enumerate(commits):
        snapshot = snapshots[commit.hexsha]
        active = " active" if i == 0 else ""
        html += f"""
        <div id="commit-{i}" class="commit-section{active}">
            <h2>Commit: {commit.hexsha[:7]}</h2>
            <p>Author: {commit.author.name}<br>Date: {commit.committed_datetime}<br>Message: {commit.message.strip()}</p>
            <pre>{generate_ascii_diagram(snapshot, None, False)}</pre>
            <pre>{format_metrics(snapshot, calculate_metrics(snapshot))}</pre>
"""
        if i > 0:
            diff = diff_snapshots(snapshots[commits[i-1].hexsha], snapshot)
            html += f"            <h3>Diff from {commits[i-1].hexsha[:7]}</h3>\n            <pre>{format_diff(diff, commits[i-1].hexsha, commit.hexsha)}</pre>\n"
        html += "        </div>\n"
    
    html += """    </div>
    <script>
        function showCommit(index) {
            document.querySelectorAll('.commit-section').forEach(s => s.classList.remove('active'));
            document.getElementById(`commit-${index}`).classList.add('active');
            document.querySelectorAll('.commit-btn').forEach((b, i) => b.classList.toggle('active', i === index));
        }
    </script>
</body>
</html>"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML report generated: {report_file}")
    return report_file