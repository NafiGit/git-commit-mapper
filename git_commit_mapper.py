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
import hashlib
import pickle
from functools import lru_cache
from git import Repo, GitCommandError
from collections import defaultdict
import re
import graphviz
import multiprocessing as mp
from datetime import datetime

# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

# Define a module-level function for defaultdict to fix pickling issues
def _create_module_dict():
    return {
        'classes': set(),
        'imported_modules': set(),
        'exporting_to': set()
    }

class CodeAnalyzer(ast.NodeVisitor):
    """Analyzes Python code to extract classes and their communications."""
    
    def __init__(self):
        self.classes = {}
        self.current_class = None
        self.current_method = None
        self.imports = {}
        self.modules = defaultdict(_create_module_dict)
        self.current_module = None
        
        # Common decorator patterns
        self.pattern_decorators = {
            'singleton': ['singleton', 'Singleton'],
            'factory': ['factory', 'Factory', 'factory_method'],
            'observer': ['observer', 'Observable', 'event_listener'],
            'command': ['command', 'Command'],
            'strategy': ['strategy', 'Strategy'],
            'adapter': ['adapter', 'Adapter'],
            'decorator': ['decorator', 'Decorator'],
            'facade': ['facade', 'Facade'],
            'proxy': ['proxy', 'Proxy'],
            'template': ['template_method', 'Template'],
            'state': ['state', 'State'],
            'builder': ['builder', 'Builder'],
        }
        
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
            'decorators': set(),               # Track decorators used by the class
            'decorator_patterns': set(),       # Track design patterns identified from decorators
            'raises_exceptions': set(),        # Track exceptions raised by methods
            'catches_exceptions': set(),       # Track exceptions caught in try-except blocks
            'lambda_count': 0,                 # Count of lambda expressions
            'generator_count': 0,              # Count of generator functions
            'context_managers': set(),         # Track context manager methods (__enter__, __exit__)
        }
        
        # Check if class has decorators and identify patterns
        for decorator in node.decorator_list:
            decorator_name = self._get_name_safely(decorator)
            if decorator_name:
                self.classes[node.name]['decorators'].add(decorator_name)
                
                # Check if decorator indicates a design pattern
                for pattern, pattern_decorators in self.pattern_decorators.items():
                    for pattern_decorator in pattern_decorators:
                        if pattern_decorator.lower() in decorator_name.lower():
                            self.classes[node.name]['decorator_patterns'].add(pattern)
        
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

    def visit_Lambda(self, node):
        """Count lambda expressions in classes."""
        if self.current_class:
            self.classes[self.current_class]['lambda_count'] += 1
        self.generic_visit(node)
    
    def visit_With(self, node):
        """Process 'with' statements to identify context manager usage."""
        if not self.current_class:
            self.generic_visit(node)
            return
            
        # Extract context manager class/object
        for item in node.items:
            context_expr = item.context_expr
            context_name = self._get_name_safely(context_expr)
            if context_name:
                self.classes[self.current_class].setdefault('uses_context_managers', set())
                self.classes[self.current_class]['uses_context_managers'].add(context_name)
                
                # Check if context relates to a known class
                for cls_name in self.classes:
                    if context_name.startswith(cls_name + '.') or context_name == cls_name:
                        self.classes[cls_name].setdefault('used_as_context_manager', set())
                        self.classes[cls_name]['used_as_context_manager'].add(self.current_class)
        
        self.generic_visit(node)
    
    def visit_Raise(self, node):
        """Process raise statements to track exception flows."""
        if not self.current_class or not self.current_method:
            self.generic_visit(node)
            return
            
        # Extract exception type
        if hasattr(node, 'exc') and node.exc:
            exc_type = self._get_name_safely(node.exc)
            if exc_type:
                self.classes[self.current_class]['raises_exceptions'].add(exc_type)
        
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node):
        """Process except handlers to track caught exceptions."""
        if not self.current_class:
            self.generic_visit(node)
            return
            
        # Extract exception type
        if node.type:
            exc_type = self._get_name_safely(node.type)
            if exc_type:
                self.classes[self.current_class]['catches_exceptions'].add(exc_type)
        
        self.generic_visit(node)


# Cache for file contents and analysis results
class AnalysisCache:
    """Cache for storing analysis results to avoid reprocessing unchanged files."""
    
    def __init__(self, cache_dir=".analysis_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = {}
        self.load_cache()
    
    def load_cache(self):
        """Load existing cache from disk if available."""
        cache_file = os.path.join(self.cache_dir, "analysis_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded {len(self.cache)} cached file analyses")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = {}
    
    def save_cache(self):
        """Save cache to disk."""
        cache_file = os.path.join(self.cache_dir, "analysis_cache.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def get_file_hash(self, file_path):
        """Calculate hash of file contents."""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception:
            return None
    
    def get_analysis(self, file_path, commit_id):
        """Get cached analysis if available."""
        file_hash = self.get_file_hash(file_path)
        if not file_hash:
            return None
        
        cache_key = f"{commit_id}:{file_path}"
        if cache_key in self.cache and self.cache[cache_key]['hash'] == file_hash:
            return self.cache[cache_key]['result']
        return None
    
    def store_analysis(self, file_path, commit_id, result):
        """Store analysis result in cache."""
        file_hash = self.get_file_hash(file_path)
        if file_hash:
            cache_key = f"{commit_id}:{file_path}"
            self.cache[cache_key] = {
                'hash': file_hash,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }


def analyze_python_file(file_path, commit_id=None, cache=None):
    """Analyze a Python file to extract class information."""
    # Check cache first if available
    if cache and commit_id:
        cached_result = cache.get_analysis(file_path, commit_id)
        if cached_result:
            return cached_result
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)
        analyzer = CodeAnalyzer()
        analyzer.file_path = file_path  # Set file path for module name extraction
        analyzer.visit(tree)
        result = (analyzer.classes, analyzer.modules)
        
        # Store in cache if available
        if cache and commit_id:
            cache.store_analysis(file_path, commit_id, result)
            
        return result
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
        
        # Check if flagged by a decorator
        if 'singleton' in details.get('decorator_patterns', set()):
            if cls_name not in patterns['Singleton']:
                patterns['Singleton'].append(cls_name)
    
    # Factory pattern
    for cls_name, details in classes.items():
        if 'factory_methods' in details and details['factory_methods']:
            patterns['Factory'].append(cls_name)
        
        # Check if flagged by a decorator
        if 'factory' in details.get('decorator_patterns', set()):
            if cls_name not in patterns['Factory']:
                patterns['Factory'].append(cls_name)
            
        # Check for methods that create and return objects
        create_methods = [m for m in details.get('methods', []) 
                        if m.startswith('create') or m.startswith('make') or m.startswith('build')]
        if create_methods and details.get('return_classes'):
            if cls_name not in patterns['Factory']:
                patterns['Factory'].append(cls_name)
    
    # Observer pattern
    for cls_name, details in classes.items():
        if ('publishes_events' in details and details['publishes_events'] and
            'subscribes_to_events' in details and details['subscribes_to_events']):
            patterns['Observer'].append(cls_name)
        
        # Check if class has observer methods like notify, addObserver, etc.
        observer_methods = ('notify', 'addObserver', 'removeObserver', 'update')
        has_observer_methods = any(method in observer_methods for method in details.get('methods', []))
        if has_observer_methods:
            if cls_name not in patterns['Observer']:
                patterns['Observer'].append(cls_name)
                
        # Check if flagged by a decorator
        if 'observer' in details.get('decorator_patterns', set()):
            if cls_name not in patterns['Observer']:
                patterns['Observer'].append(cls_name)
    
    # Builder pattern
    for cls_name, details in classes.items():
        # Check for methods that return self (chaining)
        builder_methods = [m for m in details.get('methods', []) 
                          if m.startswith('set') or m.startswith('with') or m.startswith('add')]
        if len(builder_methods) >= 3 and 'build' in details.get('methods', []):
            patterns['Builder'].append(cls_name)
            
        # Check if flagged by a decorator
        if 'builder' in details.get('decorator_patterns', set()):
            if cls_name not in patterns['Builder']:
                patterns['Builder'].append(cls_name)
    
    # Adapter pattern
    for cls_name, details in classes.items():
        # Check for classes that wrap other classes and provide a different interface
        # Often has 'Adapter' in the name or is used to adapt between interfaces
        if ('Adapter' in cls_name or 'Wrapper' in cls_name or 
            any('adapt' in method.lower() for method in details.get('methods', []))):
            patterns['Adapter'].append(cls_name)
            
        # Check if flagged by a decorator
        if 'adapter' in details.get('decorator_patterns', set()):
            if cls_name not in patterns['Adapter']:
                patterns['Adapter'].append(cls_name)
    
    # Decorator pattern (not to be confused with Python decorators)
    for cls_name, details in classes.items():
        # Decorator pattern typically has a component interface and concrete decorator
        # that wraps a component and adds functionality
        if 'Decorator' in cls_name or 'decorator' in details.get('decorator_patterns', set()):
            patterns['Decorator'].append(cls_name)
    
    # Proxy pattern
    for cls_name, details in classes.items():
        # Proxy often has 'Proxy' in name and delegates to another object
        if 'Proxy' in cls_name or 'proxy' in details.get('decorator_patterns', set()):
            patterns['Proxy'].append(cls_name)
    
    # Strategy pattern
    for cls_name, details in classes.items():
        # Strategy pattern often involves interchangeable algorithms
        if ('Strategy' in cls_name or 
            'strategy' in details.get('decorator_patterns', set()) or
            any('strategy' in attr.lower() for attr in details.get('attributes', set()))):
            patterns['Strategy'].append(cls_name)
    
    # Command pattern
    for cls_name, details in classes.items():
        # Command pattern often has 'execute' method and 'Command' in name
        if (('Command' in cls_name or 'command' in details.get('decorator_patterns', set())) and
           ('execute' in details.get('methods', []) or 'run' in details.get('methods', []))):
            patterns['Command'].append(cls_name)
    
    # Context Manager pattern (Python-specific)
    for cls_name, details in classes.items():
        # Check for __enter__ and __exit__ methods
        if '__enter__' in details.get('methods', []) and '__exit__' in details.get('methods', []):
            patterns['ContextManager'].append(cls_name)
    
    return patterns


def generate_ascii_diagram(classes, modules=None, use_color=True):
    """Generate a comprehensive ASCII diagram showing class relationships."""
    if not classes:
        return "No classes found."
    
    # Initialize color settings
    if use_color:
        C = Colors
    else:
        # Create a dummy Colors class with empty strings
        class DummyColors:
            pass
        C = DummyColors()
        for attr in dir(Colors):
            if not attr.startswith('__'):
                setattr(C, attr, '')
    
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
    diagram.append(f"{C.BOLD}{C.CYAN}CLASS STRUCTURE:{C.RESET}")
    diagram.append(f"{C.CYAN}================={C.RESET}")
    for cls_name, details in classes.items():
        # Class header with inheritance
        parents = details['parent_classes']
        class_header = f"{C.BLUE}╭─ {C.BOLD}{cls_name}{C.RESET}"
        if parents:
            class_header += f"{C.BLUE} ({C.MAGENTA}" + f"{C.RESET}{C.MAGENTA} ← {C.RESET}{C.MAGENTA}".join(parents) + f"{C.RESET}{C.BLUE}){C.RESET}"
        class_header += f"{C.BLUE} ────────────────────╮{C.RESET}"
        diagram.append(class_header)
        
        # Body content
        body = [
            f"{C.BLUE}│{C.RESET} {C.GREEN}{'Methods:':<16}{C.RESET} {', '.join(details['methods']) or 'None'}",
            f"{C.BLUE}│{C.RESET} {C.YELLOW}{'States:':<16}{C.RESET} {', '.join(details['states']) or 'None'}",
            f"{C.BLUE}│{C.RESET} {C.YELLOW}{'Props:':<16}{C.RESET} {', '.join(details['props']) or 'None'}",
            f"{C.BLUE}│{C.RESET} {C.CYAN}{'Serializers:':<16}{C.RESET} {', '.join(details['serializers']) or 'None'}"
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
        diagram.append(f"{C.BLUE}╰───────────────────────────────────────────────╯{C.RESET}")
        diagram.append("")

    # Add communication map
    diagram.append(f"\n{C.BOLD}{C.CYAN}CLASS COMMUNICATIONS:{C.RESET}")
    diagram.append(f"{C.CYAN}====================={C.RESET}")
    
    # Method calls between classes
    method_connections = [c for c in connections if c[3] == "method"]
    if method_connections:
        diagram.append(f"\n{C.YELLOW}Method Calls:{C.RESET}")
        for src, dest, label, _ in sorted(method_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {C.GREEN}{src.ljust(15)}{C.RESET} {C.BLUE}───[{C.RESET}{label}{C.BLUE}]──→{C.RESET} {C.MAGENTA}{dest}{C.RESET}")
    
    # API calls
    api_connections = [c for c in connections if c[3] == "api"]
    if api_connections:
        diagram.append(f"\n{C.YELLOW}API Communications:{C.RESET}")
        for src, _, label, _ in sorted(api_connections, key=lambda x: x[0]):
            diagram.append(f"  {C.GREEN}{src.ljust(15)}{C.RESET} {C.RED}──{label}──→{C.RESET} {C.BOLD}External API{C.RESET}")

    # Serializer relationships
    if serializer_connections:
        diagram.append(f"\n{C.YELLOW}Serializer Usage:{C.RESET}")
        for src, dest, label, _ in sorted(serializer_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {C.GREEN}{src.ljust(15)}{C.RESET} {C.CYAN}──[{label}]──→{C.RESET} {C.MAGENTA}{dest}{C.RESET}")

    # Add composition relationships
    if composition_connections:
        diagram.append(f"\n{C.YELLOW}Composition Relationships:{C.RESET}")
        for src, dest, label, _ in sorted(composition_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {C.GREEN}{src.ljust(15)}{C.RESET} {C.BLUE}◆──[{C.RESET}{label}{C.BLUE}]──→{C.RESET} {C.MAGENTA}{dest}{C.RESET}")
    
    # Add parameter type relationships
    if parameter_connections:
        diagram.append(f"\n{C.YELLOW}Parameter Type Relationships:{C.RESET}")
        for src, dest, label, _ in sorted(parameter_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {C.GREEN}{src.ljust(15)}{C.RESET} {C.BLUE}○──[{C.RESET}{label}{C.BLUE}]──→{C.RESET} {C.MAGENTA}{dest}{C.RESET}")
    
    # Add return type relationships
    if return_connections:
        diagram.append(f"\n{C.YELLOW}Return Type Relationships:{C.RESET}")
        for src, dest, label, _ in sorted(return_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {C.GREEN}{src.ljust(15)}{C.RESET} {C.BLUE}●──[{C.RESET}{label}{C.BLUE}]──→{C.RESET} {C.MAGENTA}{dest}{C.RESET}")
    
    # Add class instantiation relationships
    if instantiation_connections:
        diagram.append(f"\n{C.YELLOW}Class Instantiation Relationships:{C.RESET}")
        for src, dest, label, _ in sorted(instantiation_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {C.GREEN}{src.ljust(15)}{C.RESET} {C.BLUE}⬢──[{C.RESET}{label}{C.BLUE}]──→{C.RESET} {C.MAGENTA}{dest}{C.RESET}")
    
    # Add implementation relationships
    if implementation_connections:
        diagram.append(f"\n{C.YELLOW}Interface Implementation Relationships:{C.RESET}")
        for src, dest, label, _ in sorted(implementation_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {C.GREEN}{src.ljust(15)}{C.RESET} {C.BLUE}⊳──[{C.RESET}{label}{C.BLUE}]──→{C.RESET} {C.MAGENTA}{dest}{C.RESET}")
    
    # Add factory relationships
    if factory_connections:
        diagram.append(f"\n{C.YELLOW}Factory Method Relationships:{C.RESET}")
        for src, dest, label, _ in sorted(factory_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {C.GREEN}{src.ljust(15)}{C.RESET} {C.BLUE}⊕──[{C.RESET}{label}{C.BLUE}]──→{C.RESET} {C.MAGENTA}{dest}{C.RESET}")
    
    # Add dependency injection relationships
    if di_connections:
        diagram.append(f"\n{C.YELLOW}Dependency Injection Relationships:{C.RESET}")
        for src, dest, label, _ in sorted(di_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {C.GREEN}{src.ljust(15)}{C.RESET} {C.BLUE}⊖──[{C.RESET}{label}{C.BLUE}]──→{C.RESET} {C.MAGENTA}{dest}{C.RESET}")
    
    # Add event relationships
    if event_connections:
        diagram.append(f"\n{C.YELLOW}Event/Observer Relationships:{C.RESET}")
        for src, dest, label, _ in sorted(event_connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {C.GREEN}{src.ljust(15)}{C.RESET} {C.BLUE}⚡──[{C.RESET}{label}{C.BLUE}]──→{C.RESET} {C.MAGENTA}{dest}{C.RESET}")
    
    # Enhanced inheritance hierarchy visualization
    diagram.append(f"\n{C.BOLD}{C.CYAN}INHERITANCE HIERARCHY:{C.RESET}")
    diagram.append(f"{C.CYAN}====================={C.RESET}")
    
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
        diagram.append(f"{C.BLUE}{prefix}{connector}{C.RESET}{C.MAGENTA}{class_name}{C.RESET}")
        
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
            diagram.append(f"\n{C.BOLD}{C.CYAN}FULL CLASS RELATIONSHIP GRAPH:{C.RESET}")
            diagram.append(f"{C.CYAN}============================{C.RESET}")
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
                draw_relationship_graph(diagram, cls_name, relationship_graph, drawn_classes, "", True, C)
    
    # Add module relationships if modules are provided
    if modules:
        diagram.append(f"\n{C.BOLD}{C.CYAN}MODULE DEPENDENCIES:{C.RESET}")
        diagram.append(f"{C.CYAN}==================={C.RESET}")
        
        for module_name, details in sorted(modules.items()):
            if details['imported_modules']:
                diagram.append(f"\n{C.GREEN}{module_name}{C.RESET} {C.YELLOW}imports:{C.RESET}")
                for imported in sorted(details['imported_modules']):
                    diagram.append(f"  {C.BLUE}└─→{C.RESET} {C.MAGENTA}{imported}{C.RESET}")
            
            if 'classes' in details and details['classes']:
                diagram.append(f"\n{C.GREEN}{module_name}{C.RESET} {C.YELLOW}defines classes:{C.RESET}")
                for cls in sorted(details['classes']):
                    diagram.append(f"  {C.BLUE}└─→{C.RESET} {C.MAGENTA}{cls}{C.RESET}")
    
    # Add design pattern detection
    patterns = detect_design_patterns(classes)
    if any(patterns.values()):
        diagram.append(f"\n{C.BOLD}{C.CYAN}DETECTED DESIGN PATTERNS:{C.RESET}")
        diagram.append(f"{C.CYAN}======================={C.RESET}")
        
        for pattern, class_list in patterns.items():
            if class_list:
                diagram.append(f"\n{C.YELLOW}{pattern} Pattern:{C.RESET}")
                for cls in sorted(class_list):
                    diagram.append(f"  {C.BLUE}⚙{C.RESET} {C.GREEN}{cls}{C.RESET}")
    
    # Add decorator-based patterns if any were detected
    decorator_patterns = {}
    for cls_name, details in classes.items():
        for pattern in details.get('decorator_patterns', set()):
            decorator_patterns.setdefault(pattern, []).append(cls_name)
    
    if decorator_patterns:
        diagram.append(f"\n{C.BOLD}{C.CYAN}DECORATOR-BASED PATTERNS:{C.RESET}")
        diagram.append(f"{C.CYAN}======================={C.RESET}")
        
        for pattern, class_list in sorted(decorator_patterns.items()):
            diagram.append(f"\n{C.YELLOW}{pattern.capitalize()} Pattern (via decorators):{C.RESET}")
            for cls in sorted(class_list):
                diagram.append(f"  {C.BLUE}⚙{C.RESET} {C.GREEN}{cls}{C.RESET}")
    
    # Add functional programming metrics
    has_lambdas = any(details.get('lambda_count', 0) > 0 for details in classes.values())
    has_generators = any(details.get('generator_count', 0) > 0 for details in classes.values())
    
    if has_lambdas or has_generators:
        diagram.append(f"\n{C.BOLD}{C.CYAN}FUNCTIONAL PROGRAMMING METRICS:{C.RESET}")
        diagram.append(f"{C.CYAN}=============================={C.RESET}")
        
        if has_lambdas:
            diagram.append(f"\n{C.YELLOW}Lambda Expression Usage:{C.RESET}")
            for cls_name, details in sorted(classes.items()):
                lambda_count = details.get('lambda_count', 0)
                if lambda_count > 0:
                    diagram.append(f"  {C.GREEN}{cls_name}{C.RESET}: {C.MAGENTA}{lambda_count}{C.RESET} lambda expression(s)")
        
        if has_generators:
            diagram.append(f"\n{C.YELLOW}Generator Function Usage:{C.RESET}")
            for cls_name, details in sorted(classes.items()):
                generator_count = details.get('generator_count', 0)
                if generator_count > 0:
                    diagram.append(f"  {C.GREEN}{cls_name}{C.RESET}: {C.MAGENTA}{generator_count}{C.RESET} generator function(s)")
    
    # Add exception flow analysis
    has_exceptions = any(details.get('raises_exceptions') or details.get('catches_exceptions') 
                         for details in classes.values())
    
    if has_exceptions:
        diagram.append(f"\n{C.BOLD}{C.CYAN}EXCEPTION FLOW ANALYSIS:{C.RESET}")
        diagram.append(f"{C.CYAN}======================={C.RESET}")
        
        # Exception raised by classes
        raises = {}
        for cls_name, details in classes.items():
            for exc in details.get('raises_exceptions', set()):
                raises.setdefault(exc, []).append(cls_name)
        
        if raises:
            diagram.append(f"\n{C.YELLOW}Exception Propagation:{C.RESET}")
            for exc, cls_list in sorted(raises.items()):
                diagram.append(f"  {C.RED}{exc}{C.RESET} raised by: {C.GREEN}{', '.join(sorted(cls_list))}{C.RESET}")
        
        # Exception handling 
        catches = {}
        for cls_name, details in classes.items():
            for exc in details.get('catches_exceptions', set()):
                catches.setdefault(exc, []).append(cls_name)
        
        if catches:
            diagram.append(f"\n{C.YELLOW}Exception Handling:{C.RESET}")
            for exc, cls_list in sorted(catches.items()):
                diagram.append(f"  {C.RED}{exc}{C.RESET} caught by: {C.GREEN}{', '.join(sorted(cls_list))}{C.RESET}")
    
    return "\n".join(diagram)

def draw_relationship_graph(diagram, cls_name, graph, drawn_classes, prefix="", is_last=True, C=None):
    """Helper function to draw a comprehensive relationship graph."""
    if not C:
        class DummyColors:
            pass
        C = DummyColors()
        for attr in dir(Colors):
            if not attr.startswith('__'):
                setattr(C, attr, '')
    
    if cls_name in drawn_classes:
        branch = "└── " if is_last else "├── "
        diagram.append(f"{C.BLUE}{prefix}{branch}{C.RESET}{C.MAGENTA}{cls_name}{C.RESET} {C.YELLOW}(reference){C.RESET}")
        return
    
    # Mark this class as drawn
    drawn_classes.add(cls_name)
    
    # Draw the current class
    branch = "└── " if is_last else "├── "
    diagram.append(f"{C.BLUE}{prefix}{branch}{C.RESET}{C.BOLD}{C.GREEN}{cls_name}{C.RESET}")
    
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
        diagram.append(f"{C.BLUE}{new_prefix}{rel_branch}→{C.RESET} {C.MAGENTA}{target}{C.RESET} {C.YELLOW}[{rel_types_str}]{C.RESET}")
        
        # Recursively draw the target's relationships if not already drawn
        if target not in drawn_classes:
            target_prefix = new_prefix + ("    " if is_last_rel else "│   ")
            draw_relationship_graph(diagram, target, graph, drawn_classes, target_prefix, True, C)


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


def generate_graphviz_diagram(classes, output_file, modules=None, format='png'):
    """Generate a GraphViz diagram showing class relationships."""
    # Create a new directed graph
    dot = graphviz.Digraph(
        comment='Class Diagram', 
        format=format,
        node_attr={'shape': 'box', 'style': 'rounded,filled', 'fillcolor': 'lightblue'}
    )
    
    # Add classes as nodes
    for cls_name, details in classes.items():
        # Create label with class details
        label = f"{cls_name}\\n--------------------\\n"
        
        # Add methods
        if details['methods']:
            label += "\\nMethods:\\n" + "\\n".join(details['methods'][:5])
            if len(details['methods']) > 5:
                label += f"\\n... ({len(details['methods']) - 5} more)"
        
        # Add states if available
        if details['states']:
            label += "\\n\\nStates:\\n" + "\\n".join(list(details['states'])[:3])
            if len(details['states']) > 3:
                label += f"\\n... ({len(details['states']) - 3} more)"
        
        dot.node(cls_name, label=label)
    
    # Add inheritance relationships
    for cls_name, details in classes.items():
        for parent in details.get('parent_classes', []):
            if parent in classes:  # Only add edges to classes that exist in our snapshot
                dot.edge(parent, cls_name, arrowhead='empty', style='solid', color='blue')
    
    # Add composition relationships
    for cls_name, details in classes.items():
        for composed in details.get('composed_classes', set()):
            if composed in classes:
                dot.edge(cls_name, composed, arrowhead='diamond', style='solid', color='darkgreen')
    
    # Add method call relationships
    for cls_name, details in classes.items():
        for called in details['calls']:
            for target_cls in classes:
                if cls_name == target_cls:
                    continue
                
                # Check if the call directly targets another class (ClassName.method)
                if called.startswith(target_cls + '.'):
                    dot.edge(cls_name, target_cls, style='dashed', color='red')
                    break
                
                # Check if the call matches another class's method
                if called in classes[target_cls]['methods']:
                    dot.edge(cls_name, target_cls, style='dashed', color='red')
                    break
    
    # Add parameter type relationships
    for cls_name, details in classes.items():
        for param_cls in details.get('param_classes', set()):
            if param_cls in classes:
                dot.edge(cls_name, param_cls, style='dotted', color='purple')
    
    # Add factory relationships
    for cls_name, details in classes.items():
        for factory_method, target_cls in details.get('factory_methods', {}).items():
            if target_cls in classes:
                dot.edge(cls_name, target_cls, label=f"creates via {factory_method}", style='dashed', color='orange')
    
    # Add dependency injection relationships
    for cls_name, details in classes.items():
        for dependency in details.get('injected_dependencies', set()):
            if dependency in classes:
                dot.edge(cls_name, dependency, style='dotted', color='blue')
    
    # Create a subgraph for design patterns if any
    patterns = detect_design_patterns(classes)
    if any(patterns.values()):
        with dot.subgraph(name='cluster_patterns') as c:
            c.attr(style='filled', color='lightgrey', label='Design Patterns')
            
            for pattern, cls_list in patterns.items():
                for cls in cls_list:
                    if cls in classes:
                        pattern_node = f"{cls}_pattern"
                        c.node(pattern_node, label=f"{pattern} Pattern", shape='hexagon', style='filled', fillcolor='yellow')
                        c.edge(pattern_node, cls, style='dotted', dir='none')
    
    # Add module relationships if modules are provided
    if modules:
        with dot.subgraph(name='cluster_modules') as c:
            c.attr(style='filled', color='lightgrey', label='Module Dependencies')
            
            for module_name, details in modules.items():
                c.node(f"module_{module_name}", label=module_name, shape='folder', style='filled', fillcolor='lightyellow')
                
                # Add edges for module imports
                for imported in details.get('imported_modules', []):
                    imported_node = f"module_{imported}"
                    if imported_node in [f"module_{m}" for m in modules]:
                        c.edge(f"module_{module_name}", imported_node, style='dotted')
                
                # Connect modules to classes they define
                for cls in details.get('classes', []):
                    if cls in classes:
                        # Use invisible edges to keep related classes close together
                        dot.edge(f"module_{module_name}", cls, style='dotted', constraint='false', color='gray')
    
    # Save the diagram
    dot.render(output_file, cleanup=True)
    return f"{output_file}.{format}"


def calculate_metrics(classes):
    """Calculate code quality metrics for each class."""
    metrics = {}
    
    # Calculate class metrics for each class
    for cls_name, details in classes.items():
        cls_metrics = {}
        
        # Calculate class size metrics
        cls_metrics['num_methods'] = len(details['methods'])
        cls_metrics['num_attributes'] = len(details['attributes'])
        cls_metrics['num_states'] = len(details['states'])
        cls_metrics['num_props'] = len(details['props'])
        
        # Calculate coupling metrics
        # Efferent coupling (CE): number of classes this class depends on
        efferent_coupling = len(set(details.get('composed_classes', set())) | 
                               set(details.get('param_classes', set())) | 
                               set(details.get('injected_dependencies', set())))
        cls_metrics['efferent_coupling'] = efferent_coupling
        
        # Afferent coupling (CA): number of classes that depend on this class
        afferent_coupling = 0
        for other_cls, other_details in classes.items():
            if other_cls == cls_name:
                continue
            if (cls_name in other_details.get('composed_classes', set()) or
                cls_name in other_details.get('param_classes', set()) or
                cls_name in other_details.get('injected_dependencies', set())):
                afferent_coupling += 1
        cls_metrics['afferent_coupling'] = afferent_coupling
        
        # Calculate instability: I = CE / (CE + CA)
        # Ranges from 0 to 1. 0 = maximally stable, 1 = maximally unstable
        total_coupling = efferent_coupling + afferent_coupling
        instability = efferent_coupling / total_coupling if total_coupling > 0 else 0
        cls_metrics['instability'] = round(instability, 2)
        
        # Calculate abstraction level: number of abstract methods / total methods
        abstract_methods = len(details.get('abstract_methods', set()))
        abstraction = abstract_methods / cls_metrics['num_methods'] if cls_metrics['num_methods'] > 0 else 0
        cls_metrics['abstraction'] = round(abstraction, 2)
        
        # Calculate inheritance depth
        inheritance_depth = 0
        current_parents = details.get('parent_classes', [])
        while current_parents:
            inheritance_depth += 1
            next_parents = []
            for parent in current_parents:
                if parent in classes:
                    next_parents.extend(classes[parent].get('parent_classes', []))
            current_parents = next_parents
        cls_metrics['inheritance_depth'] = inheritance_depth
        
        # Add metrics to the result dictionary
        metrics[cls_name] = cls_metrics
    
    return metrics


def format_metrics(classes, metrics):
    """Format code metrics into a readable string."""
    if not metrics:
        return "No metrics calculated."
    
    lines = []
    lines.append("CODE QUALITY METRICS:")
    lines.append("=====================")
    lines.append("")
    
    for cls_name, cls_metrics in sorted(metrics.items()):
        lines.append(f"Class: {cls_name}")
        lines.append("-" * (len(cls_name) + 7))
        
        # Size metrics
        lines.append(f"Size: {cls_metrics['num_methods']} methods, {cls_metrics['num_attributes']} attributes")
        
        # Coupling metrics
        lines.append(f"Efferent Coupling (CE): {cls_metrics['efferent_coupling']} (outgoing dependencies)")
        lines.append(f"Afferent Coupling (CA): {cls_metrics['afferent_coupling']} (incoming dependencies)")
        lines.append(f"Instability (I = CE/(CE+CA)): {cls_metrics['instability']:.2f} (0=stable, 1=unstable)")
        
        # Inheritance metrics
        lines.append(f"Inheritance Depth: {cls_metrics['inheritance_depth']}")
        lines.append(f"Abstraction: {cls_metrics['abstraction']:.2f}")
        
        # Warnings
        warnings = []
        if cls_metrics['num_methods'] > 20:
            warnings.append("Class has too many methods (> 20)")
        if cls_metrics['efferent_coupling'] > 5:
            warnings.append("High efferent coupling (> 5)")
        if cls_metrics['inheritance_depth'] > 3:
            warnings.append("Deep inheritance hierarchy (> 3)")
        
        if warnings:
            lines.append("\nWarnings:")
            for warning in warnings:
                lines.append(f"  - {warning}")
        
        lines.append("")
    
    return "\n".join(lines)


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


def analyze_commit(repo, commit, output_dir, ascii_only=False, graphviz_format='png', show_modules=False, 
                use_color=True, calculate_code_metrics=False, cache=None, parallel=True, 
                max_processes=0, max_files_per_process=100, exclude_dirs=None):
    """Analyze a single commit and return a snapshot of the codebase."""
    print(f"Analyzing commit {commit.hexsha[:7]}: {commit.message.strip()}")
    
    # Checkout the commit
    repo.git.checkout(commit.hexsha, force=True)
    
    snapshot = {}
    all_modules = {}
    
    # Get list of Python files to analyze
    python_files = []
    for root, _, files in os.walk(repo.working_dir):
        # Skip files in .git directory and virtual environment directories
        if '.git' in root or '/env/' in root or '/venv/' in root or 'site-packages' in root:
            continue
        
        # Skip user-specified exclude directories
        if exclude_dirs and any(exclude_dir in root for exclude_dir in exclude_dirs):
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
    
    # Analyze Python files - either in parallel or sequentially
    if parallel and len(python_files) > 1 and mp.cpu_count() > 1:
        print(f"Analyzing {len(python_files)} files in parallel...")
        
        # Determine number of processes to use
        if max_processes <= 0:
            num_processes = min(mp.cpu_count(), len(python_files))
        else:
            num_processes = min(max_processes, mp.cpu_count(), len(python_files))
        
        # Split files into chunks to avoid memory issues with large repositories
        chunk_size = min(max_files_per_process, max(1, len(python_files) // num_processes))
        file_chunks = [python_files[i:i + chunk_size] for i in range(0, len(python_files), chunk_size)]
        
        print(f"Using {num_processes} processes with max {chunk_size} files per process")
        print(f"Processing {len(file_chunks)} chunks of files")
        
        try:
            # Use a process pool to analyze file chunks in parallel
            with mp.Pool(processes=num_processes) as pool:
                for chunk in file_chunks:
                    # Prepare arguments for worker function
                    worker_args = [(file_path, commit.hexsha, cache) for file_path in chunk]
                    
                    # Process this chunk of files
                    results = pool.map(analyze_file_worker, worker_args)
                    
                    # Combine results from this chunk
                    for result in results:
                        if result:
                            classes, modules = result
                            snapshot.update(classes)
                            all_modules.update(modules)
        except Exception as e:
            print(f"Error in parallel processing: {str(e)}")
            print("Falling back to sequential processing...")
            # Fall back to sequential processing
            for file_path in python_files:
                try:
                    classes, modules = analyze_python_file(file_path, commit.hexsha, cache)
                    snapshot.update(classes)
                    all_modules.update(modules)
                except Exception as file_e:
                    print(f"Error analyzing {file_path}: {str(file_e)}")
    else:
        # Sequential processing
        print(f"Processing {len(python_files)} files sequentially...")
        for file_path in python_files:
            try:
                classes, modules = analyze_python_file(file_path, commit.hexsha, cache)
                snapshot.update(classes)
                all_modules.update(modules)
            except Exception as e:
                print(f"Error analyzing {file_path}: {str(e)}")
    
    # Generate and save ASCII diagram
    diagram = generate_ascii_diagram(snapshot, all_modules if show_modules else None, use_color)
    diagram_file = os.path.join(output_dir, f"diagram_{commit.hexsha[:7]}.txt")
    
    with open(diagram_file, 'w', encoding='utf-8') as f:
        f.write(f"Commit: {commit.hexsha}\n")
        f.write(f"Date: {commit.committed_datetime}\n")
        f.write(f"Author: {commit.author.name} <{commit.author.email}>\n")
        f.write(f"Message: {commit.message.strip()}\n\n")
        f.write(diagram)
    
    # Calculate and save code metrics if requested
    if calculate_code_metrics:
        metrics = calculate_metrics(snapshot)
        metrics_formatted = format_metrics(snapshot, metrics)
        metrics_file = os.path.join(output_dir, f"metrics_{commit.hexsha[:7]}.txt")
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(f"Code Metrics for Commit: {commit.hexsha}\n")
            f.write(f"Date: {commit.committed_datetime}\n")
            f.write(f"Author: {commit.author.name} <{commit.author.email}>\n")
            f.write(f"Message: {commit.message.strip()}\n\n")
            f.write(metrics_formatted)
    
    # Generate and save GraphViz diagram
    if not ascii_only:
        graphviz_file = os.path.join(output_dir, f"diagram_{commit.hexsha[:7]}")
        try:
            generate_graphviz_diagram(snapshot, graphviz_file, all_modules if show_modules else None, graphviz_format)
            print(f"GraphViz diagram saved to {graphviz_file}.{graphviz_format}")
        except Exception as e:
            print(f"Error generating GraphViz diagram: {str(e)}")
    
    return snapshot


def generate_html_report(snapshots, commits, output_dir):
    """Generate an interactive HTML report with class diagrams and metrics."""
    # Create HTML report file
    report_file = os.path.join(output_dir, "class_analysis_report.html")
    
    # Basic HTML structure with CSS and JavaScript
    html_header = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Architecture Analysis Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; line-height: 1.6; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background-color: #2c3e50; color: white; padding: 1em; text-align: center; }
        .commit-nav { display: flex; overflow-x: auto; margin-bottom: 20px; background-color: #f5f5f5; padding: 10px; }
        .commit-btn { min-width: 120px; margin-right: 10px; padding: 8px 15px; background-color: #3498db; 
                     color: white; border: none; border-radius: 4px; cursor: pointer; transition: background-color 0.3s; }
        .commit-btn:hover { background-color: #2980b9; }
        .commit-btn.active { background-color: #2c3e50; }
        .commit-section { display: none; margin-top: 20px; }
        .commit-section.active { display: block; }
        .diagram { background-color: #f8f9fa; padding: 20px; margin-bottom: 20px; border-radius: 5px; overflow-x: auto; }
        .metrics { background-color: #f8f9fa; padding: 20px; border-radius: 5px; }
        pre { white-space: pre-wrap; font-family: 'Courier New', Courier, monospace; }
        .class-node { fill: #3498db; stroke: #2980b9; }
        .tab-container { display: flex; margin-bottom: 10px; }
        .tab { padding: 10px 20px; background-color: #f0f0f0; border: none; cursor: pointer; border-radius: 5px 5px 0 0; }
        .tab.active { background-color: #3498db; color: white; }
        .tab-content { display: none; padding: 20px; background-color: #f8f9fa; border-radius: 0 5px 5px 5px; }
        .tab-content.active { display: block; }
        .summary { margin-bottom: 20px; }
        .relationship-graph svg { max-width: 100%; height: auto; }
        .diff-section { margin-top: 30px; border-top: 1px solid #ddd; padding-top: 20px; }
        .added { color: green; font-weight: bold; }
        .removed { color: red; text-decoration: line-through; }
        .modified { color: orange; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Code Architecture Analysis Report</h1>
        <p>Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>
    <div class="container">
        <div class="summary">
            <h2>Repository Summary</h2>
            <p>Total commits analyzed: """ + str(len(commits)) + """</p>
            <p>Date range: """ + commits[0].committed_datetime.strftime("%Y-%m-%d") + """ to """ + commits[-1].committed_datetime.strftime("%Y-%m-%d") + """</p>
        </div>
        
        <h2>Commit Navigation</h2>
        <div class="commit-nav">
"""
    
    # Add commit navigation buttons
    for i, commit in enumerate(commits):
        active = " active" if i == 0 else ""
        short_msg = commit.message.strip().split('\n')[0][:50]
        html_header += f'            <button class="commit-btn{active}" onclick="showCommit({i})">{commit.hexsha[:7]} - {short_msg}</button>\n'
    
    html_header += """        </div>
        
        <div id="commit-sections">
"""
    
    # Add each commit section
    commit_sections = ""
    for i, commit in enumerate(commits):
        snapshot = snapshots[commit.hexsha]
        active = " active" if i == 0 else ""
        
        # Create commit section with tabs for different views
        section = f"""
        <div id="commit-{i}" class="commit-section{active}">
            <h2>Commit: {commit.hexsha[:7]}</h2>
            <p><strong>Author:</strong> {commit.author.name} ({commit.author.email})<br>
            <strong>Date:</strong> {commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S")}<br>
            <strong>Message:</strong> {commit.message.strip()}</p>
            
            <div class="tab-container">
                <button class="tab active" onclick="showTab('diagram-{i}', this)">Class Diagram</button>
                <button class="tab" onclick="showTab('structure-{i}', this)">Class Structure</button>
                <button class="tab" onclick="showTab('metrics-{i}', this)">Metrics</button>
            </div>
            
            <div id="diagram-{i}" class="tab-content active">
                <div class="diagram">
                    <h3>Class Diagram</h3>
                    <p>This diagram shows the relationships between classes.</p>
                    <div class="ascii-diagram">
                        <pre>{generate_ascii_diagram(snapshot, None, False)}</pre>
                    </div>
                </div>
            </div>
            
            <div id="structure-{i}" class="tab-content">
                <div class="class-structure">
                    <h3>Class Structure</h3>
                    <p>Detailed structure of each class in this commit.</p>
"""
        
        # Add details for each class
        for cls_name, details in sorted(snapshot.items()):
            section += f"""
                    <div class="class-box">
                        <h4>{cls_name}</h4>
                        <p><strong>Methods:</strong> {', '.join(details['methods']) or 'None'}</p>
                        <p><strong>States:</strong> {', '.join(details['states']) or 'None'}</p>
                        <p><strong>Props:</strong> {', '.join(details['props']) or 'None'}</p>
                        <p><strong>Serializers:</strong> {', '.join(details['serializers']) or 'None'}</p>
                        <p><strong>Parent Classes:</strong> {', '.join(details.get('parent_classes', [])) or 'None'}</p>
                    </div>
"""
        
        section += """
                </div>
            </div>
            
            <div id="metrics-{i}" class="tab-content">
                <div class="metrics">
                    <h3>Code Metrics</h3>
                    <p>Metrics for evaluating code quality and architecture.</p>
                    <pre>""" + format_metrics(snapshot, calculate_metrics(snapshot)) + """</pre>
                </div>
            </div>
"""
        
        # Add diff section if not the first commit
        if i > 0:
            prev_commit = commits[i-1]
            diff = diff_snapshots(snapshots[prev_commit.hexsha], snapshot)
            
            section += f"""
            <div class="diff-section">
                <h3>Changes from Previous Commit</h3>
                <p>This shows what changed since commit {prev_commit.hexsha[:7]}</p>
                <pre>{format_diff(diff, prev_commit.hexsha, commit.hexsha)}</pre>
            </div>
"""
        
        section += """        </div>
"""
        commit_sections += section
    
    # JavaScript for interactivity
    javascript = """
        </div>
    </div>
    
    <script>
        function showCommit(index) {
            // Hide all commit sections
            document.querySelectorAll('.commit-section').forEach(section => {
                section.classList.remove('active');
            });
            
            // Show selected commit section
            document.getElementById(`commit-${index}`).classList.add('active');
            
            // Update button active state
            document.querySelectorAll('.commit-btn').forEach((btn, i) => {
                if (i === index) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
        }
        
        function showTab(tabId, button) {
            // Get the parent commit section
            const commitSection = button.closest('.commit-section');
            
            // Hide all tab contents in this commit section
            commitSection.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Show the selected tab content
            document.getElementById(tabId).classList.add('active');
            
            // Update tab button active state
            commitSection.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            button.classList.add('active');
        }
    </script>
</body>
</html>
"""
    
    # Write the HTML report
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_header + commit_sections + javascript)
    
    print(f"HTML report generated: {report_file}")
    return report_file


def main():
    """Main function to process the repository."""
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
    parser.add_argument('--max-files-per-process', type=int, default=100, help='Maximum number of files per process to avoid memory issues')
    parser.add_argument('--generate-html', action='store_true', help='Generate HTML report with interactive diagrams')
    parser.add_argument('--exclude-dirs', nargs='+', default=[], help='Additional directories to exclude from analysis')
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
    
    # Initialize analysis cache if enabled
    cache = None
    if not args.no_cache:
        print(f"Initializing analysis cache in {args.cache_dir}")
        cache = AnalysisCache(args.cache_dir)
    
    # Save the original branch/commit to return to
    original_branch = repo.active_branch.name
    
    try:
        snapshots = {}
        
        # Process each commit
        for commit in commits:
            snapshot = analyze_commit(
                repo, 
                commit, 
                output_dir, 
                args.ascii_only, 
                args.format, 
                args.show_modules, 
                not args.no_color, 
                args.metrics,
                cache,
                not args.no_parallel,
                args.max_processes,
                args.max_files_per_process,
                args.exclude_dirs
            )
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
        
        # Generate HTML report if requested
        if args.generate_html:
            print("Generating HTML report...")
            generate_html_report(snapshots, commits, output_dir)
        
        # Save cache if used
        if cache:
            cache.save_cache()
            print(f"Analysis cache saved to {args.cache_dir}")
        
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