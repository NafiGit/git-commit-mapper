import ast
from collections import defaultdict

def _create_module_dict():
    """Default factory for module dictionary."""
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
        self.current_module = getattr(self, 'file_path', 'unknown').split('/')[-1].replace('.py', '')
        self.generic_visit(node)
        
    def visit_Import(self, node):
        """Process import statements."""
        for name in node.names:
            module_name = name.name.split('.')[0]
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
        """Safely get the name from a node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
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
            'attributes': set(),
            'composed_classes': set(),
            'param_classes': set(),
            'return_classes': set(),
            'instantiated_classes': set(),
            'implements_interfaces': set(),
            'abstract_methods': set(),
            'implemented_methods': set(),
            'decorators': set(),
            'decorator_patterns': set(),
            'raises_exceptions': set(),
            'catches_exceptions': set(),
            'lambda_count': 0,
            'generator_count': 0,
            'context_managers': set(),
        }
        
        for decorator in node.decorator_list:
            decorator_name = self._get_name_safely(decorator)
            if decorator_name:
                self.classes[node.name]['decorators'].add(decorator_name)
                for pattern, pattern_decorators in self.pattern_decorators.items():
                    for pattern_decorator in pattern_decorators:
                        if pattern_decorator.lower() in decorator_name.lower():
                            self.classes[node.name]['decorator_patterns'].add(pattern)
        
        for base in node.bases:
            base_name = self._get_name_safely(base)
            if base_name:
                self.classes[node.name]['parent_classes'].append(base_name)
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                for decorator in item.decorator_list:
                    if self._get_name_safely(decorator) == 'abstractmethod':
                        self.classes[node.name]['abstract_methods'].add(item.name)
        
        self.generic_visit(node)
        self._check_interface_implementation()
        self.current_class = None

    def _check_interface_implementation(self):
        """Check if the current class implements abstract methods."""
        if not self.current_class:
            return
        
        for parent in self.classes[self.current_class]['parent_classes']:
            if parent in self.classes and self.classes[parent]['abstract_methods']:
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
            
            for arg in node.args.args:
                if getattr(arg, 'arg', None) == 'self':
                    pass
            
            if node.returns:
                return_type = self._get_name_safely(node.returns)
                if return_type and return_type in self.classes:
                    self.classes[self.current_class].setdefault('factory_methods', {})
                    self.classes[self.current_class]['factory_methods'][node.name] = return_type
                    self.classes[return_type].setdefault('created_by_factories', set())
                    self.classes[return_type]['created_by_factories'].add(f"{self.current_class}.{node.name}")
        
            if node.name == '__init__':
                for arg in node.args.args:
                    if arg.arg != 'self' and hasattr(arg, 'annotation'):
                        injected_type = self._get_name_safely(arg.annotation)
                        if injected_type in self.classes:
                            self.classes[self.current_class].setdefault('injected_dependencies', set())
                            self.classes[self.current_class]['injected_dependencies'].add(injected_type)
                            self.classes[injected_type].setdefault('injected_into', set())
                            self.classes[injected_type]['injected_into'].add(self.current_class)
        
        self.generic_visit(node)
        self.current_method = prev_method
        
    def visit_Attribute(self, node):
        """Process attribute access."""
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
            
        if isinstance(node.func, ast.Attribute):
            call_name = self._get_name_safely(node.func)
            if call_name:
                parts = call_name.split('.')
                if parts and (parts[0] in ('requests', 'urllib', 'http') or 'api' in call_name.lower()):
                    self.classes[self.current_class]['api_calls'].add(call_name)
                elif not (len(parts) > 0 and parts[0] == 'self'):
                    self.classes[self.current_class]['calls'].add(call_name)
        elif isinstance(node.func, ast.Name):
            call_name = node.func.id
            if call_name in self.imports:
                self.classes[self.current_class]['calls'].add(call_name)
            elif 'api' in call_name.lower() or 'request' in call_name.lower():
                self.classes[self.current_class]['api_calls'].add(call_name)
            else:
                self.classes[self.current_class]['calls'].add(call_name)
                
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
        """Process attribute assignments."""
        if not self.current_class:
            self.generic_visit(node)
            return
        
        if isinstance(node.targets[0], ast.Attribute) and isinstance(node.targets[0].value, ast.Name):
            if node.targets[0].value.id == 'self':
                attr_name = node.targets[0].attr
                self.classes[self.current_class]['attributes'].add(attr_name)
                if isinstance(node.value, ast.Call):
                    class_name = self._get_name_safely(node.value.func)
                    if class_name and not class_name.startswith(('self.', 'cls.')):
                        self.classes[self.current_class]['instantiated_classes'].add(class_name)
                        self.classes[self.current_class]['composed_classes'].add(class_name)
        
        self.generic_visit(node)

    def visit_Return(self, node):
        """Process return statements."""
        if not self.current_class or not self.current_method:
            self.generic_visit(node)
            return
        
        if isinstance(node.value, ast.Call):
            class_name = self._get_name_safely(node.value.func)
            if class_name and not class_name.startswith(('self.', 'cls.')):
                self.classes[self.current_class]['return_classes'].add(class_name)
        
        self.generic_visit(node)

    def visit_Lambda(self, node):
        """Count lambda expressions."""
        if self.current_class:
            self.classes[self.current_class]['lambda_count'] += 1
        self.generic_visit(node)
    
    def visit_With(self, node):
        """Process 'with' statements."""
        if not self.current_class:
            self.generic_visit(node)
            return
            
        for item in node.items:
            context_expr = item.context_expr
            context_name = self._get_name_safely(context_expr)
            if context_name:
                self.classes[self.current_class].setdefault('uses_context_managers', set())
                self.classes[self.current_class]['uses_context_managers'].add(context_name)
                for cls_name in self.classes:
                    if context_name.startswith(cls_name + '.') or context_name == cls_name:
                        self.classes[cls_name].setdefault('used_as_context_manager', set())
                        self.classes[cls_name]['used_as_context_manager'].add(self.current_class)
        
        self.generic_visit(node)
    
    def visit_Raise(self, node):
        """Process raise statements."""
        if not self.current_class or not self.current_method:
            self.generic_visit(node)
            return
            
        if hasattr(node, 'exc') and node.exc:
            exc_type = self._get_name_safely(node.exc)
            if exc_type:
                self.classes[self.current_class]['raises_exceptions'].add(exc_type)
        
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node):
        """Process except handlers."""
        if not self.current_class:
            self.generic_visit(node)
            return
            
        if node.type:
            exc_type = self._get_name_safely(node.type)
            if exc_type:
                self.classes[self.current_class]['catches_exceptions'].add(exc_type)
        
        self.generic_visit(node)

def analyze_python_file(file_path, commit_id=None, cache=None):
    """Analyze a Python file to extract class information."""
    from utils.cache import AnalysisCache
    if cache and commit_id:
        cached_result = cache.get_analysis(file_path, commit_id)
        if cached_result:
            return cached_result
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)
        analyzer = CodeAnalyzer()
        analyzer.file_path = file_path
        analyzer.visit(tree)
        result = (analyzer.classes, analyzer.modules)
        
        if cache and commit_id:
            cache.store_analysis(file_path, commit_id, result)
            
        return result
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")
        return {}, {}