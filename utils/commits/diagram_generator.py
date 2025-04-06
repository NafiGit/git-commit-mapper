from collections import defaultdict
import graphviz
from utils.commits.colors import Colors
from utils.commits.design_patterns import DesignPatternDetector

def generate_ascii_diagram(classes, modules=None, use_color=True, inheritance_only=False, relationship_only=False, detailed=False, show_file_associations=False):
    """Generate a comprehensive ASCII diagram with filtering options."""
    if not classes:
        return "No classes found."
    
    # Create Colors class dynamically if color is disabled
    C = Colors if use_color else type('DummyColors', (), {attr: '' for attr in dir(Colors) if not attr.startswith('__')})()
    
    diagram = []
    connections = []
    
    # Handle mutual exclusivity of filters
    if inheritance_only and relationship_only:
        raise ValueError("Cannot specify both --inheritance-only and --relationship-only")
    
    # Detailed class information section (only if detailed mode or no filters)
    if detailed or not (inheritance_only or relationship_only):
        diagram.append(f"{C.BOLD}{C.CYAN}CLASS DETAILS:{C.RESET}")
        diagram.append(f"{C.CYAN}=============={C.RESET}")
        
        for cls_name, details in classes.items():
            parents = details['parent_classes']
            class_header = f"{C.BLUE}╭─ {C.BOLD}{cls_name}{C.RESET}"
            if parents:
                class_header += f"{C.BLUE} ({C.MAGENTA}" + f"{C.RESET}{C.MAGENTA} ← {C.RESET}{C.MAGENTA}".join(parents) + f"{C.RESET}{C.BLUE}){C.RESET}"
            class_header += f"{C.BLUE} ────────────────────╮{C.RESET}"
            diagram.append(class_header)
            
            body = [
                f"{C.BLUE}│{C.RESET} {C.GREEN}{'Methods:':<16}{C.RESET} {', '.join(details['methods']) or 'None'}",
                f"{C.BLUE}│{C.RESET} {C.YELLOW}{'States:':<16}{C.RESET} {', '.join(details['states']) or 'None'}",
                f"{C.BLUE}│{C.RESET} {C.YELLOW}{'Props:':<16}{C.RESET} {', '.join(details['props']) or 'None'}",
                f"{C.BLUE}│{C.RESET} {C.CYAN}{'Serializers:':<16}{C.RESET} {', '.join(details['serializers']) or 'None'}"
            ]
            diagram.extend(body)
            diagram.append(f"{C.BLUE}╰───────────────────────────────────────────────╯{C.RESET}")
            diagram.append("")
    
    # Build inheritance tree
    inheritance_tree = {}
    for cls_name, details in classes.items():
        for parent in details.get('parent_classes', []):
            if parent not in inheritance_tree:
                inheritance_tree[parent] = []
            inheritance_tree[parent].append(cls_name)
    
    # Build relationship graph
    relationship_graph = {}
    for cls_name, details in classes.items():
        if cls_name not in relationship_graph:
            relationship_graph[cls_name] = []
        for composed_class in details.get('composed_classes', set()):
            relationship_graph[cls_name].append((composed_class, 'composes'))
        for called_class in details.get('calls', set()):
            relationship_graph[cls_name].append((called_class, 'calls'))
    
    # Class hierarchy section
    diagram.append(f"\n{C.BOLD}{C.CYAN}CLASS HIERARCHY:{C.RESET}")
    diagram.append(f"{C.CYAN}================{C.RESET}")
    
    drawn_classes = set()
    for cls_name in classes:
        if cls_name not in drawn_classes and not any(cls_name in children for children in inheritance_tree.values()):
            drawn_classes.update(draw_inheritance(cls_name))
    
    # Class relationships section with more detail
    if not inheritance_only:
        diagram.append(f"\n{C.BOLD}{C.CYAN}CLASS RELATIONSHIPS:{C.RESET}")
        diagram.append(f"{C.CYAN}==================={C.RESET}")
        
        drawn_classes = set()
        relationship_graph = {}
        
        # Build more detailed relationships
        for cls_name, details in classes.items():
            if cls_name not in relationship_graph:
                relationship_graph[cls_name] = []
            for composed_class in details.get('composed_classes', set()):
                relationship_graph[cls_name].append((composed_class, 'composes'))
            for called_class in details.get('calls', set()):
                relationship_graph[cls_name].append((called_class, 'calls'))
            for injected in details.get('injected_dependencies', set()):
                relationship_graph[cls_name].append((injected, 'depends on'))
        
        # Draw relationships
        for cls_name in classes:
            if cls_name not in drawn_classes:
                draw_relationship_graph(diagram, cls_name, relationship_graph, drawn_classes, C=C)
    
    # Class communications section
    diagram.append(f"\n{C.BOLD}{C.CYAN}CLASS COMMUNICATIONS:{C.RESET}")
    diagram.append(f"{C.CYAN}====================={C.RESET}")
    if connections:
        diagram.append(f"\n{C.YELLOW}Method Calls:{C.RESET}")
        for src, dest, label, _ in sorted(connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {C.GREEN}{src.ljust(15)}{C.RESET} {C.BLUE}───[{C.RESET}{label}{C.BLUE}]──→{C.RESET} {C.MAGENTA}{dest}{C.RESET}")
    
    # Add design patterns section
    pattern_detector = DesignPatternDetector()
    patterns = pattern_detector.detect_patterns(classes)
    if patterns:
        diagram.append(f"\n{C.BOLD}{C.CYAN}DESIGN PATTERNS:{C.RESET}")
        diagram.append(f"{C.CYAN}==============={C.RESET}")
        for pattern, cls_list in patterns.items():
            if cls_list:
                diagram.append(f"{C.YELLOW}{pattern}:{C.RESET}")
                for cls in cls_list:
                    diagram.append(f"  {C.GREEN}• {cls}{C.RESET}")
    
    # Add Functional Programming Metrics section
    diagram.append(f"\n{C.BOLD}{C.CYAN}FUNCTIONAL PROGRAMMING METRICS:{C.RESET}")
    diagram.append(f"{C.CYAN}=============================={C.RESET}")
    diagram.append(f"\n{C.YELLOW}Lambda Expression Usage:{C.RESET}")
    for cls_name, details in classes.items():
        if details.get('lambda_count', 0) > 0:
            diagram.append(f"  {C.GREEN}{cls_name}{C.RESET}: {C.MAGENTA}{details['lambda_count']}{C.RESET} lambda expression(s)")

    # Add Exception Flow Analysis section
    diagram.append(f"\n{C.BOLD}{C.CYAN}EXCEPTION FLOW ANALYSIS:{C.RESET}")
    diagram.append(f"{C.CYAN}======================={C.RESET}")
    
    # Exception Propagation
    diagram.append(f"\n{C.YELLOW}Exception Propagation:{C.RESET}")
    for cls_name, details in classes.items():
        if details.get('raises_exceptions'):
            for exc in details['raises_exceptions']:
                diagram.append(f"  {C.RED}e{C.RESET} raised by: {C.GREEN}{cls_name}{C.RESET}")
    
    # Exception Handling
    diagram.append(f"\n{C.YELLOW}Exception Handling:{C.RESET}")
    exception_handlers = defaultdict(list)
    for cls_name, details in classes.items():
        if details.get('catches_exceptions'):
            for exc in details['catches_exceptions']:
                exception_handlers[exc].append(cls_name)
    
    for exc, handlers in sorted(exception_handlers.items()):
        diagram.append(f"  {C.RED}{exc}{C.RESET} caught by: {C.GREEN}{', '.join(handlers)}{C.RESET}")
    
    # Add Decorator-based Patterns section
    diagram.append(f"\n{C.BOLD}{C.CYAN}DECORATOR-BASED PATTERNS:{C.RESET}")
    diagram.append(f"{C.CYAN}======================={C.RESET}")
    
    decorator_patterns = defaultdict(list)
    for cls_name, details in classes.items():
        for pattern in details.get('decorator_patterns', set()):
            decorator_patterns[pattern].append(cls_name)
    
    for pattern, cls_list in decorator_patterns.items():
        diagram.append(f"\n{C.YELLOW}{pattern} Pattern (via decorators):{C.RESET}")
        for cls in cls_list:
            diagram.append(f"  {C.BLUE}⚙{C.RESET} {C.GREEN}{cls}{C.RESET}")
    
    if show_file_associations and detailed:
        diagram.append(f"\n{C.BOLD}{C.CYAN}FILE ASSOCIATIONS:{C.RESET}")
        diagram.append(f"{C.CYAN}================={C.RESET}")
        
        file_associations_seen = set()
        for cls_name, details in classes.items():
            if 'file_associations' in details:
                file_info = details['file_associations']
                file_path = file_info['file']
                
                if file_path not in file_associations_seen:
                    file_associations_seen.add(file_path)
                    diagram.append(f"\n{C.BLUE}File: {file_path}{C.RESET}")
                    
                    assocs = file_info['associated_files']
                    if assocs['inherits_from']:
                        diagram.append(f"{C.GREEN}  Inherits from:{C.RESET}")
                        for f in assocs['inherits_from']:
                            diagram.append(f"    → {f}")
                    
                    if assocs['depends_on']:
                        diagram.append(f"{C.YELLOW}  Depends on:{C.RESET}")
                        for f in assocs['depends_on']:
                            diagram.append(f"    → {f}")
                    
                    if assocs['composes']:
                        diagram.append(f"{C.MAGENTA}  Composes:{C.RESET}")
                        for f in assocs['composes']:
                            diagram.append(f"    → {f}")
                    
                    if assocs['calls']:
                        diagram.append(f"{C.CYAN}  Calls:{C.RESET}")
                        for f in assocs['calls']:
                            diagram.append(f"    → {f}")
    
    return "\n".join(diagram)

def generate_graphviz_diagram(classes, output_file, modules=None, format='png'):
    """Generate a GraphViz diagram."""
    dot = graphviz.Digraph(
        comment='Class Diagram', 
        format=format,
        node_attr={
            'shape': 'box', 
            'style': 'rounded,filled', 
            'fillcolor': 'lightblue',
            'fontname': 'Helvetica',
            'fontsize': '10'
        }
    )
    
    # Add subgraphs for modules if available
    if modules:
        for module_name, module_info in modules.items():
            with dot.subgraph(name=f'cluster_{module_name}') as c:
                c.attr(label=module_name, style='rounded', color='blue')
                for cls_name in module_info['classes']:
                    if cls_name in classes:
                        # Add class name to details for proper labeling
                        class_details = classes[cls_name].copy()
                        class_details['name'] = cls_name
                        c.node(cls_name, _format_class_node(class_details))
    
    # Add remaining classes
    for cls_name, details in classes.items():
        if not modules or not any(cls_name in m['classes'] for m in modules.values()):
            # Add class name to details for proper labeling
            class_details = details.copy()
            class_details['name'] = cls_name
            dot.node(cls_name, _format_class_node(class_details))
    
    # Add relationships with different colors and styles
    for cls_name, details in classes.items():
        # Inheritance (blue, solid line, empty arrow)
        for parent in details.get('parent_classes', []):
            if parent in classes:
                dot.edge(parent, cls_name, arrowhead='empty', style='solid', color='blue')
        
        # Composition (dark green, solid line, diamond arrow)
        for composed in details.get('composed_classes', set()):
            if composed in classes:
                dot.edge(cls_name, composed, arrowhead='diamond', style='solid', color='darkgreen')
        
        # Method calls (red, dashed line, vee arrow)
        for call in details.get('calls', set()):
            if call in classes:
                dot.edge(cls_name, call, arrowhead='vee', style='dashed', color='red')
        
        # Dependencies (gray, dotted line, vee arrow)
        for dep in details.get('injected_dependencies', set()):
            if dep in classes:
                dot.edge(cls_name, dep, arrowhead='vee', style='dotted', color='gray')
    
    dot.render(output_file, cleanup=True)
    return f"{output_file}.{format}"

def _format_class_node(details):
    """Format class node label with detailed information."""
    label_parts = []
    
    # Add class name
    label_parts.append(f"{details.get('name', '')}") if 'name' in details else label_parts.append(details.get('class_name', ''))
    
    # Add methods section
    if details.get('methods'):
        label_parts.append('Methods:')
        for method in sorted(details['methods'])[:5]:
            label_parts.append(f"  {method}()")
        if len(details['methods']) > 5:
            label_parts.append('  ...')
    
    # Add attributes section
    if details.get('attributes'):
        label_parts.append('Attributes:')
        for attr in sorted(details['attributes'])[:3]:
            label_parts.append(f"  {attr}")
        if len(details['attributes']) > 3:
            label_parts.append('  ...')
    
    # Add states if present
    if details.get('states'):
        label_parts.append('States:')
        for state in sorted(details['states'])[:2]:
            label_parts.append(f"  {state}")
        if len(details['states']) > 2:
            label_parts.append('  ...')
    
    # Add props if present
    if details.get('props'):
        label_parts.append('Props:')
        for prop in sorted(details['props'])[:2]:
            label_parts.append(f"  {prop}")
        if len(details['props']) > 2:
            label_parts.append('  ...')
    
    # Add design patterns if present
    if details.get('decorator_patterns'):
        label_parts.append('Patterns:')
        for pattern in sorted(details['decorator_patterns'])[:2]:
            label_parts.append(f"  {pattern}")
    
    return '\\n'.join(label_parts)

def draw_inheritance(class_name, prefix="", is_last=True):
    """Draw inheritance tree in ASCII format."""
    lines = []
    branch = "└── " if is_last else "├── "
    lines.append(prefix + branch + class_name)
    child_prefix = prefix + ("    " if is_last else "│   ")
    return lines

def draw_relationship_graph(diagram, cls_name, graph, drawn_classes, prefix="", is_last=True, C=None):
    """Draw relationship graph in ASCII format."""
    if C is None:
        class DummyColors:
            RESET = ''
            BLUE = ''
            GREEN = ''
            YELLOW = ''
            MAGENTA = ''
            CYAN = ''
            BOLD = ''
        C = DummyColors()

    if cls_name in drawn_classes:
        return

    drawn_classes.add(cls_name)
    branch = "└── " if is_last else "├── "
    diagram.append(prefix + branch + f"{C.BOLD}{cls_name}{C.RESET}")

    if cls_name in graph:
        relationships = sorted(graph[cls_name])
        for idx, (rel_cls, rel_type) in enumerate(relationships):
            is_last_child = idx == len(relationships) - 1
            child_prefix = prefix + ("    " if is_last else "│   ")
            rel_branch = "└── " if is_last_child else "├── "
            diagram.append(child_prefix + rel_branch + f"{C.BLUE}[{rel_type}]{C.RESET} {C.MAGENTA}{rel_cls}{C.RESET}")
            draw_relationship_graph(diagram, rel_cls, graph, drawn_classes, child_prefix, is_last_child, C)
