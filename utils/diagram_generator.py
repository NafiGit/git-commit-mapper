from collections import defaultdict
import graphviz
from utils.colors import Colors

def detect_design_patterns(classes):
    """Detect common design patterns."""
    patterns = defaultdict(list)
    
    for cls_name, details in classes.items():
        has_instance_var = any('_instance' in attr for attr in details.get('attributes', set()))
        has_get_instance = any('get_instance' in method or 'getInstance' in method 
                              for method in details.get('methods', []))
        if has_instance_var and has_get_instance or 'singleton' in details.get('decorator_patterns', set()):
            patterns['Singleton'].append(cls_name)
        
        if 'factory_methods' in details and details['factory_methods'] or 'factory' in details.get('decorator_patterns', set()):
            patterns['Factory'].append(cls_name)
        
        if ('publishes_events' in details and details['publishes_events'] and
            'subscribes_to_events' in details and details['subscribes_to_events']) or 'observer' in details.get('decorator_patterns', set()):
            patterns['Observer'].append(cls_name)
    
    return patterns

def generate_ascii_diagram(classes, modules=None, use_color=True, inheritance_only=False, relationship_only=False, detailed=False):
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
    patterns = detect_design_patterns(classes)
    if patterns:
        diagram.append(f"\n{C.BOLD}{C.CYAN}DESIGN PATTERNS:{C.RESET}")
        diagram.append(f"{C.CYAN}==============={C.RESET}")
        for pattern, cls_list in patterns.items():
            if cls_list:
                diagram.append(f"{C.YELLOW}{pattern}:{C.RESET}")
                for cls in cls_list:
                    diagram.append(f"  {C.GREEN}• {cls}{C.RESET}")
    
    return "\n".join(diagram)

def generate_graphviz_diagram(classes, output_file, modules=None, format='png'):
    """Generate a GraphViz diagram."""
    dot = graphviz.Digraph(
        comment='Class Diagram', 
        format=format,
        node_attr={'shape': 'box', 'style': 'rounded,filled', 'fillcolor': 'lightblue'}
    )
    
    for cls_name, details in classes.items():
        label = f"{cls_name}\\n--------------------\\n"
        if details['methods']:
            label += "\\nMethods:\\n" + "\\n".join(details['methods'][:5])
        dot.node(cls_name, label=label)
    
    for cls_name, details in classes.items():
        for parent in details.get('parent_classes', []):
            if parent in classes:
                dot.edge(parent, cls_name, arrowhead='empty', style='solid', color='blue')
    
    dot.render(output_file, cleanup=True)
    return f"{output_file}.{format}"

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
