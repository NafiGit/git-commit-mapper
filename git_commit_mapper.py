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
        self.modules = defaultdict(lambda: {
            'classes': set(),
            'imported_modules': set(),
            'exporting_to': set()
        })
        self.current_module = None
        
    def visit_Module(self, node):
        """Process a module."""
        # Store the current module name based on file path
        self.current_module = getattr(self, 'file_path', 'unknown').split('/')[-1].replace('.py', '')
        self.generic_visit(node)
        
    def visit_Import(self, node):
        """Process import statements."""
        for name in node.names:
            module_name = name.name.split('.')[0]  # Get the top-level module
            self.imports[name.asname or name.name] = name.name
            if self.current_module:
                self.modules[self.current_module]['imported_modules'].add(module_name)
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
            'parent_classes': [],
            'attributes': set(),               # Store class attributes
            'composed_classes': set(),         # Classes used in composition
            'param_classes': set(),            # Classes used as parameters
            'return_classes': set(),           # Classes returned from methods
            'instantiated_classes': set(),     # Classes directly instantiated
            'implements_interfaces': set(),    # Track interfaces/abstract classes implemented
            'abstract_methods': set(),         # Track abstract methods defined in this class
            'implemented_methods': set(),      # Track methods that implement abstract methods
        }
        
        # Extract parent classes
        for base in node.bases:
            base_name = self._get_name_safely(base)
            if base_name:
                self.classes[node.name]['parent_classes'].append(base_name)
        
        # Visit class body to extract methods and find abstract methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                # Check for @abstractmethod decorator
                for decorator in item.decorator_list:
                    if self._get_name_safely(decorator) == 'abstractmethod':
                        self.classes[node.name]['abstract_methods'].add(item.name)
        
        # Visit class body
        self.generic_visit(node)
        
        # After visiting, check if this class implements methods from abstract parents
        self._check_interface_implementation()
        
        self.current_class = None

    def _check_interface_implementation(self):
        """Check if the current class implements abstract methods from parent classes."""
        if not self.current_class:
            return
        
        # For each parent class
        for parent in self.classes[self.current_class]['parent_classes']:
            if parent in self.classes and self.classes[parent]['abstract_methods']:
                # Check if any of the parent's abstract methods are implemented in this class
                for abstract_method in self.classes[parent]['abstract_methods']:
                    if abstract_method in self.classes[self.current_class]['methods']:
                        self.classes[self.current_class]['implemented_methods'].add(abstract_method)
                        self.classes[self.current_class]['implements_interfaces'].add(parent)
        
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
            
        # Check for factory method pattern
        if self.current_class and node.returns:
            return_type = self._get_name_safely(node.returns)
            if return_type and return_type in self.classes:
                # Method returns another class type - potential factory
                self.classes[self.current_class].setdefault('factory_methods', {})
                self.classes[self.current_class]['factory_methods'][node.name] = return_type
                
                # Also track the created class's factory relationship
                self.classes[return_type].setdefault('created_by_factories', set())
                self.classes[return_type]['created_by_factories'].add(f"{self.current_class}.{node.name}")
        
        # Check for constructor-based dependency injection
        if self.current_class and node.name == '__init__':
            for arg in node.args.args:
                if arg.arg != 'self' and hasattr(arg, 'annotation'):
                    injected_type = self._get_name_safely(arg.annotation)
                    if injected_type in self.classes:
                        self.classes[self.current_class].setdefault('injected_dependencies', set())
                        self.classes[self.current_class]['injected_dependencies'].add(injected_type)
                        
                        # Track reverse relationship
                        self.classes[injected_type].setdefault('injected_into', set())
                        self.classes[injected_type]['injected_into'].add(self.current_class)
        
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
        """Process function/method calls with event pattern detection."""
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
                
        # Check for common event pattern methods
        event_pattern_methods = {
            'on_': 'subscriber',
            'add_listener': 'subscriber',
            'addEventListener': 'subscriber',
            'emit': 'publisher',
            'dispatch': 'publisher',
            'trigger': 'publisher',
            'notify': 'publisher'
        }
        
        if self.current_class and isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            for pattern, role in event_pattern_methods.items():
                if method_name.startswith(pattern) or method_name == pattern:
                    if role == 'publisher':
                        self.classes[self.current_class].setdefault('publishes_events', set())
                        self.classes[self.current_class]['publishes_events'].add(method_name)
                    elif role == 'subscriber':
                        self.classes[self.current_class].setdefault('subscribes_to_events', set())
                        self.classes[self.current_class]['subscribes_to_events'].add(method_name)
        
        self.generic_visit(node)

    def visit_Assign(self, node):
        """Process attribute assignments (e.g., self.db = Database())."""
        if not self.current_class:
            self.generic_visit(node)
            return
        
        # Track class attributes
        if isinstance(node.targets[0], ast.Attribute) and isinstance(node.targets[0].value, ast.Name):
            if node.targets[0].value.id == 'self':
                attr_name = node.targets[0].attr
                self.classes[self.current_class]['attributes'].add(attr_name)
                
                # Check for class instantiation in the assignment
                if isinstance(node.value, ast.Call):
                    class_name = self._get_name_safely(node.value.func)
                    if class_name and not class_name.startswith(('self.', 'cls.')):
                        self.classes[self.current_class]['instantiated_classes'].add(class_name)
                        self.classes[self.current_class]['composed_classes'].add(class_name)
            
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Process function/method definitions with type annotations."""
        prev_method = self.current_method
        self.current_method = node.name
        
        if self.current_class:
            self.classes[self.current_class]['methods'].append(node.name)
            
            # Check for parameter type annotations
            for arg in node.args.args:
                if hasattr(arg, 'annotation') and arg.annotation:
                    param_type = self._get_name_safely(arg.annotation)
                    if param_type and param_type not in ('str', 'int', 'float', 'bool', 'list', 'dict', 'tuple'):
                        self.classes[self.current_class]['param_classes'].add(param_type)
            
            # Check for return type annotations
            if hasattr(node, 'returns') and node.returns:
                return_type = self._get_name_safely(node.returns)
                if return_type and return_type not in ('str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'None'):
                    self.classes[self.current_class]['return_classes'].add(return_type)
        
        self.generic_visit(node)
        self.current_method = prev_method

    def visit_Return(self, node):
        """Process return statements to detect returned classes."""
        if not self.current_class or not self.current_method:
            self.generic_visit(node)
            return
        
        # Check for class instantiation in return statements
        if isinstance(node.value, ast.Call):
            class_name = self._get_name_safely(node.value.func)
            if class_name and not class_name.startswith(('self.', 'cls.')):
                self.classes[self.current_class]['return_classes'].add(class_name)
        
        self.generic_visit(node)


def analyze_python_file(file_path):
    """Analyze a Python file to extract class information."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)
        analyzer = CodeAnalyzer()
        analyzer.file_path = file_path  # Set file path for module name extraction
        analyzer.visit(tree)
        return analyzer.classes, analyzer.modules
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")
        return {}, {}


def detect_design_patterns(classes):
    """Detect common design patterns in the class structure."""
    patterns = defaultdict(list)
    
    # Singleton pattern
    for cls_name, details in classes.items():
        # Check for private instance variable and getInstance method
        has_instance_var = any('_instance' in attr for attr in details.get('attributes', set()))
        has_get_instance = any('get_instance' in method or 'getInstance' in method 
                              for method in details.get('methods', []))
        if has_instance_var and has_get_instance:
            patterns['Singleton'].append(cls_name)
    
    # Factory pattern
    for cls_name, details in classes.items():
        if 'factory_methods' in details and details['factory_methods']:
            patterns['Factory'].append(cls_name)
    
    # Observer pattern
    for cls_name, details in classes.items():
        if ('publishes_events' in details and details['publishes_events'] and
            'subscribes_to_events' in details and details['subscribes_to_events']):
            patterns['Observer'].append(cls_name)
    
    # Builder pattern
    for cls_name, details in classes.items():
        # Check for methods that return self (chaining)
        builder_methods = [m for m in details.get('methods', []) 
                          if m.startswith('set') or m.startswith('with') or m.startswith('add')]
        if len(builder_methods) >= 3 and 'build' in details.get('methods', []):
            patterns['Builder'].append(cls_name)
    
    return patterns


def generate_ascii_diagram(classes, modules=None):
    """Generate a comprehensive ASCII diagram showing class relationships."""
    if not classes:
        return "No classes found."
    
    diagram = []
    method_map = defaultdict(list)
    connections = []
    serializer_connections = []
    composition_connections = []
    parameter_connections = []
    return_connections = []
    instantiation_connections = []
    
    # Build method map and relationship maps
    for cls_name, details in classes.items():
        for method in details['methods']:
            method_map[method].append(cls_name)
        
        # Collect composition relationships
        for composed in details.get('composed_classes', set()):
            if composed in classes:  # Only include classes that exist in our snapshot
                composition_connections.append((cls_name, composed, "composes", "composition"))
        
        # Collect parameter type relationships
        for param_cls in details.get('param_classes', set()):
            if param_cls in classes:
                parameter_connections.append((cls_name, param_cls, "uses as parameter", "parameter"))
        
        # Collect return type relationships
        for return_cls in details.get('return_classes', set()):
            if return_cls in classes:
                return_connections.append((cls_name, return_cls, "returns", "return"))
        
        # Collect instantiation relationships
        for inst_cls in details.get('instantiated_classes', set()):
            if inst_cls in classes:
                instantiation_connections.append((cls_name, inst_cls, "instantiates", "instantiation"))
    
    # Collect implementation relationships
    implementation_connections = []
    for cls_name, details in classes.items():
        for interface in details.get('implements_interfaces', set()):
            implementation_connections.append((
                cls_name, 
                interface, 
                f"implements {', '.join(details.get('implemented_methods', set()))}", 
                "implementation"
            ))
    
    # Collect factory relationships
    factory_connections = []
    for cls_name, details in classes.items():
        for factory_method, target_cls in details.get('factory_methods', {}).items():
            factory_connections.append((
                cls_name, 
                target_cls, 
                f"creates via {factory_method}()", 
                "factory"
            ))
    
    # Collect dependency injection relationships
    di_connections = []
    for cls_name, details in classes.items():
        for dependency in details.get('injected_dependencies', set()):
            di_connections.append((
                cls_name, 
                dependency, 
                "depends on", 
                "dependency_injection"
            ))
    
    # Collect event relationships
    event_connections = []
    publishers = {}
    subscribers = {}
    
    for cls_name, details in classes.items():
        if 'publishes_events' in details:
            for event in details['publishes_events']:
                publishers[event] = cls_name
        if 'subscribes_to_events' in details:
            for event in details['subscribes_to_events']:
                subscribers.setdefault(event, []).append(cls_name)
    
    # Match publishers with subscribers
    for event, pub_cls in publishers.items():
        for sub_cls in subscribers.get(event, []):
            event_connections.append((
                pub_cls, 
                sub_cls, 
                f"notifies via {event}", 
                "event"
            ))
    
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

    # Add composition relationships
    if composition_connections:
        diagram.append("\nComposition Relationships:")
        for src, dest, label, _ in sorted(composition_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {src.ljust(15)} ◆──[{label}]──→ {dest}")
    
    # Add parameter type relationships
    if parameter_connections:
        diagram.append("\nParameter Type Relationships:")
        for src, dest, label, _ in sorted(parameter_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {src.ljust(15)} ○──[{label}]──→ {dest}")
    
    # Add return type relationships
    if return_connections:
        diagram.append("\nReturn Type Relationships:")
        for src, dest, label, _ in sorted(return_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {src.ljust(15)} ●──[{label}]──→ {dest}")
    
    # Add class instantiation relationships
    if instantiation_connections:
        diagram.append("\nClass Instantiation Relationships:")
        for src, dest, label, _ in sorted(instantiation_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {src.ljust(15)} ⬢──[{label}]──→ {dest}")
    
    # Add implementation relationships
    if implementation_connections:
        diagram.append("\nInterface Implementation Relationships:")
        for src, dest, label, _ in sorted(implementation_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {src.ljust(15)} ⊳──[{label}]──→ {dest}")
    
    # Add factory relationships
    if factory_connections:
        diagram.append("\nFactory Method Relationships:")
        for src, dest, label, _ in sorted(factory_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {src.ljust(15)} ⊕──[{label}]──→ {dest}")
    
    # Add dependency injection relationships
    if di_connections:
        diagram.append("\nDependency Injection Relationships:")
        for src, dest, label, _ in sorted(di_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {src.ljust(15)} ⊖──[{label}]──→ {dest}")
    
    # Add event relationships
    if event_connections:
        diagram.append("\nEvent/Observer Relationships:")
        for src, dest, label, _ in sorted(event_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {src.ljust(15)} ⚡──[{label}]──→ {dest}")
    
    # Enhanced inheritance hierarchy visualization
    diagram.append("\nINHERITANCE HIERARCHY:")
    diagram.append("=====================")
    
    # Build inheritance tree
    root_classes = []
    inheritance_map = defaultdict(list)
    
    for cls_name, details in classes.items():
        parents = details.get('parent_classes', [])
        if not parents:
            root_classes.append(cls_name)
        for parent in parents:
            if parent in classes:  # Only include classes that exist in our snapshot
                inheritance_map[parent].append(cls_name)
    
    # Draw inheritance tree
    def draw_inheritance(class_name, prefix="", is_last=True):
        connector = "└── " if is_last else "├── "
        diagram.append(f"{prefix}{connector}{class_name}")
        
        children = sorted(inheritance_map.get(class_name, []))
        prefix_extension = "    " if is_last else "│   "
        for i, child in enumerate(children):
            draw_inheritance(child, prefix + prefix_extension, i == len(children) - 1)
    
    # Draw trees starting from root classes
    for i, cls in enumerate(sorted(root_classes)):
        is_last = i == len(root_classes) - 1
        draw_inheritance(cls, "", is_last)
    
    # If there are enough classes, add the full relationship graph
    if len(classes) > 1:
        all_connections = (
            [(s, d, "calls methods in") for s, d, _, _ in method_connections] +
            [(s, d, "composed of") for s, d, _, _ in composition_connections] +
            [(s, d, "uses as param") for s, d, _, _ in parameter_connections] +
            [(s, d, "returns") for s, d, _, _ in return_connections]
        )
        
        if all_connections:
            diagram.append("\nFULL CLASS RELATIONSHIP GRAPH:")
            diagram.append("============================")
            diagram.append("")
            
            # Build detailed relationship graph
            relationship_graph = defaultdict(lambda: defaultdict(set))
            for src, dest, rel_type in all_connections:
                relationship_graph[src][dest].add(rel_type)
            
            # Generate visual graph
            drawn_classes = set()
            for cls_name in sorted(classes.keys()):
                if cls_name in drawn_classes:
                    continue
                
                # Create a relationship-oriented graph
                draw_relationship_graph(diagram, cls_name, relationship_graph, drawn_classes, "", True)
    
    # Add this after the class relationships
    if modules:
        diagram.append("\nMODULE DEPENDENCIES:")
        diagram.append("===================")
        
        for module_name, details in sorted(modules.items()):
            if details['imported_modules']:
                diagram.append(f"\n{module_name} imports:")
                for imported in sorted(details['imported_modules']):
                    diagram.append(f"  └─→ {imported}")
            
            if 'classes' in details and details['classes']:
                diagram.append(f"\n{module_name} defines classes:")
                for cls in sorted(details['classes']):
                    diagram.append(f"  └─→ {cls}")
    
    # Add design pattern detection
    patterns = detect_design_patterns(classes)
    if any(patterns.values()):
        diagram.append("\nDETECTED DESIGN PATTERNS:")
        diagram.append("=======================")
        
        for pattern, class_list in patterns.items():
            if class_list:
                diagram.append(f"\n{pattern} Pattern:")
                for cls in sorted(class_list):
                    diagram.append(f"  ⚙ {cls}")
    
    return "\n".join(diagram)

def draw_relationship_graph(diagram, cls_name, graph, drawn_classes, prefix="", is_last=True):
    """Helper function to draw a comprehensive relationship graph."""
    if cls_name in drawn_classes:
        branch = "└── " if is_last else "├── "
        diagram.append(f"{prefix}{branch}{cls_name} (reference)")
        return
    
    # Mark this class as drawn
    drawn_classes.add(cls_name)
    
    # Draw the current class
    branch = "└── " if is_last else "├── "
    diagram.append(f"{prefix}{branch}{cls_name}")
    
    # Calculate new prefix for relationships
    new_prefix = prefix + ("    " if is_last else "│   ")
    
    # Find all relationships for this class
    relationships = []
    for target, rel_types in graph.get(cls_name, {}).items():
        relationships.append((target, rel_types))
    
    # Draw relationships
    for i, (target, rel_types) in enumerate(sorted(relationships)):
        is_last_rel = (i == len(relationships) - 1)
        rel_branch = "└── " if is_last_rel else "├── "
        rel_types_str = ", ".join(sorted(rel_types))
        diagram.append(f"{new_prefix}{rel_branch}→ {target} [{rel_types_str}]")
        
        # Recursively draw the target's relationships if not already drawn
        if target not in drawn_classes:
            target_prefix = new_prefix + ("    " if is_last_rel else "│   ")
            draw_relationship_graph(diagram, target, graph, drawn_classes, target_prefix, True)


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
        
        # Compare composition relationships
        old_composed = set(old_class.get('composed_classes', set()))
        new_composed = set(new_class.get('composed_classes', set()))
        if old_composed != new_composed:
            changes['composed_classes'] = {
                'added': list(new_composed - old_composed),
                'removed': list(old_composed - new_composed)
            }
        
        # Compare parameter type relationships
        old_params = set(old_class.get('param_classes', set()))
        new_params = set(new_class.get('param_classes', set()))
        if old_params != new_params:
            changes['param_classes'] = {
                'added': list(new_params - old_params),
                'removed': list(old_params - new_params)
            }
        
        # Compare return type relationships
        old_returns = set(old_class.get('return_classes', set()))
        new_returns = set(new_class.get('return_classes', set()))
        if old_returns != new_returns:
            changes['return_classes'] = {
                'added': list(new_returns - old_returns),
                'removed': list(old_returns - new_returns)
            }
        
        # Compare instantiation relationships
        old_inst = set(old_class.get('instantiated_classes', set()))
        new_inst = set(new_class.get('instantiated_classes', set()))
        if old_inst != new_inst:
            changes['instantiated_classes'] = {
                'added': list(new_inst - old_inst),
                'removed': list(old_inst - new_inst)
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
            
            # Composition changes
            if 'composed_classes' in changes:
                if changes['composed_classes']['added']:
                    lines.append("      Added composed classes:")
                    for class_name in sorted(changes['composed_classes']['added']):
                        lines.append(f"        + {class_name}")
                if changes['composed_classes']['removed']:
                    lines.append("      Removed composed classes:")
                    for class_name in sorted(changes['composed_classes']['removed']):
                        lines.append(f"        - {class_name}")
            
            # Parameter type changes
            if 'param_classes' in changes:
                if changes['param_classes']['added']:
                    lines.append("      Added parameter classes:")
                    for class_name in sorted(changes['param_classes']['added']):
                        lines.append(f"        + {class_name}")
                if changes['param_classes']['removed']:
                    lines.append("      Removed parameter classes:")
                    for class_name in sorted(changes['param_classes']['removed']):
                        lines.append(f"        - {class_name}")
            
            # Return type changes
            if 'return_classes' in changes:
                if changes['return_classes']['added']:
                    lines.append("      Added return classes:")
                    for class_name in sorted(changes['return_classes']['added']):
                        lines.append(f"        + {class_name}")
                if changes['return_classes']['removed']:
                    lines.append("      Removed return classes:")
                    for class_name in sorted(changes['return_classes']['removed']):
                        lines.append(f"        - {class_name}")
            
            # Instantiation changes
            if 'instantiated_classes' in changes:
                if changes['instantiated_classes']['added']:
                    lines.append("      Added instantiated classes:")
                    for class_name in sorted(changes['instantiated_classes']['added']):
                        lines.append(f"        + {class_name}")
                if changes['instantiated_classes']['removed']:
                    lines.append("      Removed instantiated classes:")
                    for class_name in sorted(changes['instantiated_classes']['removed']):
                        lines.append(f"        - {class_name}")
            
            lines.append("")
    
    return "\n".join(lines)


def analyze_commit(repo, commit, output_dir):
    """Analyze a single commit and return a snapshot of the codebase."""
    print(f"Analyzing commit {commit.hexsha[:7]}: {commit.message.strip()}")
    
    # Checkout the commit
    repo.git.checkout(commit.hexsha, force=True)
    
    snapshot = {}
    all_modules = {}
    
    # Analyze Python files
    for root, _, files in os.walk(repo.working_dir):
        for file in files:
            # Skip files in .git directory and virtual environment directories
            if '.git' in root or '/env/' in root or '/venv/' in root or 'site-packages' in root:
                continue
                
            file_path = os.path.join(root, file)
            
            # Only analyze Python files
            if file.endswith('.py'):
                classes, modules = analyze_python_file(file_path)
                snapshot.update(classes)
                all_modules.update(modules)
    
    # Generate and save ASCII diagram
    diagram = generate_ascii_diagram(snapshot, all_modules)
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