#!/usr/bin/env python3
"""
Single file class diagram generator that analyzes Python files using AST
and creates detailed class associations.
"""

import ast
from collections import defaultdict
from typing import Dict, Set, Tuple, Any, Optional
import os
from utils.commits.code_analyzer import CodeAnalyzer, _create_module_dict
from utils.commits.design_patterns import DesignPatternDetector

class SingleFileAnalyzer(ast.NodeVisitor):
    """Analyzes a single Python file to extract class relationships and associations."""
    
    def __init__(self):
        self.classes = {}
        self.current_class = None
        self.current_method = None
        self.associations = defaultdict(set)  # Track class associations
        self.dependencies = defaultdict(set)  # Track class dependencies
        self.aggregations = defaultdict(set)  # Track aggregation relationships
        self.compositions = defaultdict(set)  # Track composition relationships
        self.method_calls = defaultdict(set)  # Track method calls between classes
        
    def analyze_file(self, file_path: str) -> Tuple[Dict[str, Any], Dict[str, Set[str]]]:
        """
        Analyze a single Python file and extract class relationships.
        
        Args:
            file_path: Path to the Python file to analyze
            
        Returns:
            Tuple containing class information and relationships
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            tree = ast.parse(code)
            self.visit(tree)
            
            # Create the relationships dictionary
            relationships = {
                'associations': dict(self.associations),
                'dependencies': dict(self.dependencies),
                'aggregations': dict(self.aggregations),
                'compositions': dict(self.compositions),
                'method_calls': dict(self.method_calls)
            }
            
            return self.classes, relationships
            
        except Exception as e:
            print(f"Error analyzing file {file_path}: {str(e)}")
            return {}, {}
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Process class definitions and track relationships."""
        prev_class = self.current_class
        self.current_class = node.name
        
        # Initialize class data structure with all required fields
        self.classes[node.name] = {
            'methods': [],
            'attributes': set(),
            'instance_variables': set(),
            'class_variables': set(),
            'parent_classes': [],
            'inner_classes': set(),
            'method_calls': set(),
            'dependencies': set(),
            'associations': set(),
            'compositions': set(),
            'aggregations': set(),
            'states': set(),  # Added missing field
            'props': set(),   # Added missing field
            'serializers': set(), # Added missing field
            'calls': set(),   # Added missing field
            'api_calls': set(), # Added missing field
            'raises_exceptions': set(), # Added missing field
            'catches_exceptions': set(), # Added missing field
            'lambda_count': 0, # Added missing field
            'generator_count': 0, # Added missing field
            'context_managers': set(), # Added missing field
            'decorator_patterns': set(), # Added missing field
            'decorators': set() # Added missing field
        }
        
        # Handle inheritance
        for base in node.bases:
            base_name = self._get_name(base)
            if base_name:
                self.classes[node.name]['parent_classes'].append(base_name)
        
        # Visit class body
        for item in node.body:
            if isinstance(item, ast.ClassDef):
                # Track inner classes
                self.classes[node.name]['inner_classes'].add(item.name)
            self.visit(item)
        
        self.current_class = prev_class
    
    def visit_Assign(self, node: ast.Assign):
        """Process assignments to detect relationships."""
        if not self.current_class:
            return
        
        # Check for instance variable assignments
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                if target.value.id == 'self':
                    var_name = target.attr
                    self.classes[self.current_class]['instance_variables'].add(var_name)
                    
                    # Check for composition (strong relationship)
                    if isinstance(node.value, ast.Call):
                        value_type = self._get_name(node.value.func)
                        if value_type in self.classes:
                            self.compositions[self.current_class].add(value_type)
                            self.classes[self.current_class]['compositions'].add(value_type)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Process method definitions."""
        if not self.current_class:
            return
            
        prev_method = self.current_method
        self.current_method = node.name
        self.classes[self.current_class]['methods'].append(node.name)
        
        # Analyze method parameters for dependencies
        for arg in node.args.args:
            if hasattr(arg, 'annotation') and arg.annotation:
                arg_type = self._get_name(arg.annotation)
                if arg_type and arg_type in self.classes:
                    self.dependencies[self.current_class].add(arg_type)
                    self.classes[self.current_class]['dependencies'].add(arg_type)
        
        self.generic_visit(node)
        self.current_method = prev_method
    
    def visit_Call(self, node: ast.Call):
        """Process method calls to track inter-class communication."""
        if not self.current_class or not self.current_method:
            return
            
        if isinstance(node.func, ast.Attribute):
            # Track method calls between classes
            if isinstance(node.func.value, ast.Name):
                called_obj = node.func.value.id
                if called_obj in self.classes:
                    self.method_calls[self.current_class].add((called_obj, node.func.attr))
                    self.classes[self.current_class]['method_calls'].add(f"{called_obj}.{node.func.attr}")
        
        self.generic_visit(node)
    
    def _get_name(self, node: ast.AST) -> Optional[str]:
        """Safely extract names from AST nodes."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return None

def analyze_single_file(file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Analyze a single Python file and generate class diagram information.
    
    Args:
        file_path: Path to the Python file to analyze
        
    Returns:
        Tuple containing class information and relationships
    """
    analyzer = SingleFileAnalyzer()
    classes, relationships = analyzer.analyze_file(file_path)
    
    # Detect design patterns
    pattern_detector = DesignPatternDetector()
    patterns = pattern_detector.detect_patterns(classes)
    
    # Add design patterns to class information
    for pattern_name, pattern_classes in patterns.items():
        for class_name in pattern_classes:
            if class_name in classes:
                classes[class_name].setdefault('design_patterns', set()).add(pattern_name)
    
    return classes, relationships

def generate_single_file_diagram(file_path: str, output_dir: str) -> Tuple[str, str]:
    """
    Generate class diagram for a single Python file.
    
    Args:
        file_path: Path to the Python file to analyze
        output_dir: Directory to save the output diagrams
        
    Returns:
        Tuple containing paths to the ASCII and GraphViz diagrams
    """
    classes, relationships = analyze_single_file(file_path)
    
    # Generate ASCII diagram
    from utils.commits.diagram_generator import generate_ascii_diagram
    ascii_diagram = generate_ascii_diagram(
        classes,
        None,  # No module information for single file
        use_color=True,
        detailed=True
    )
    
    # Save ASCII diagram
    file_name = os.path.basename(file_path)
    ascii_file = os.path.join(output_dir, f"{file_name}_diagram.txt")
    with open(ascii_file, 'w', encoding='utf-8') as f:
        f.write(f"Class Diagram for {file_name}\n")
        f.write("=" * (16 + len(file_name)) + "\n\n")
        f.write(ascii_diagram)
    
    # Generate GraphViz diagram
    from utils.commits.diagram_generator import generate_graphviz_diagram
    graphviz_file = os.path.join(output_dir, f"{file_name}_diagram")
    generate_graphviz_diagram(
        classes,
        graphviz_file,
        None,  # No module information for single file
        'png'
    )
    
    return ascii_file, f"{graphviz_file}.png"
