from collections import defaultdict
from typing import Dict, Set, List, Tuple
import ast
import os
from utils.files.class_diagram_single_file import SingleFileAnalyzer, analyze_single_file

class FileAssociationAnalyzer:
    """Analyzes associations between Python files in a codebase."""
    
    def __init__(self):
        self.file_classes: Dict[str, Dict] = {}  # Maps files to their classes
        self.class_to_file: Dict[str, str] = {}  # Maps class names to their files
        self.file_imports: Dict[str, Set[str]] = defaultdict(set)  # Maps files to their imports
        self.file_associations: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
    def analyze_files(self, files: List[str]) -> Dict[str, Dict]:
        """Analyze multiple files and their relationships."""
        # First pass: collect all classes and their locations
        for file_path in files:
            try:
                classes, _ = analyze_single_file(file_path)
                if classes:
                    self.file_classes[file_path] = classes
                    for class_name in classes:
                        self.class_to_file[class_name] = file_path
            except Exception as e:
                print(f"Error analyzing file {file_path}: {str(e)}")
        
        # Second pass: analyze relationships between files
        for file_path, classes in self.file_classes.items():
            self._analyze_file_associations(file_path, classes)
        
        return self._build_association_report()
    
    def _analyze_file_associations(self, file_path: str, classes: Dict):
        """Analyze associations for a single file."""
        for class_name, class_info in classes.items():
            # Check parent classes
            for parent in class_info.get('parent_classes', []):
                if parent in self.class_to_file:
                    parent_file = self.class_to_file[parent]
                    if parent_file != file_path:
                        self.file_associations[file_path]['inherits_from'].add(parent_file)
            
            # Check dependencies and compositions
            for dep in class_info.get('dependencies', set()):
                if dep in self.class_to_file:
                    dep_file = self.class_to_file[dep]
                    if dep_file != file_path:
                        self.file_associations[file_path]['depends_on'].add(dep_file)
            
            for comp in class_info.get('compositions', set()):
                if comp in self.class_to_file:
                    comp_file = self.class_to_file[comp]
                    if comp_file != file_path:
                        self.file_associations[file_path]['composes'].add(comp_file)
            
            # Check method calls
            for call in class_info.get('calls', set()):
                called_class = call.split('.')[0] if '.' in call else call
                if called_class in self.class_to_file:
                    called_file = self.class_to_file[called_class]
                    if called_file != file_path:
                        self.file_associations[file_path]['calls'].add(called_file)
    
    def _build_association_report(self) -> Dict[str, Dict]:
        """Build a comprehensive report of file associations."""
        report = {}
        for file_path, associations in self.file_associations.items():
            report[file_path] = {
                'classes': list(self.file_classes[file_path].keys()),
                'associations': {
                    'inherits_from': list(associations['inherits_from']),
                    'depends_on': list(associations['depends_on']),
                    'composes': list(associations['composes']),
                    'calls': list(associations['calls'])
                }
            }
        return report

def analyze_file_associations(files: List[str]) -> Dict[str, Dict]:
    """
    Analyze associations between multiple Python files.
    
    Args:
        files: List of Python file paths to analyze
        
    Returns:
        Dictionary containing file associations and relationships
    """
    analyzer = FileAssociationAnalyzer()
    return analyzer.analyze_files(files)
