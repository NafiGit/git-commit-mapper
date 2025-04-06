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

def generate_ascii_diagram(classes, modules=None, use_color=True):
    """Generate a comprehensive ASCII diagram."""
    if not classes:
        return "No classes found."
    
    C = Colors if use_color else type('DummyColors', (), {attr: '' for attr in dir(Colors) if not attr.startswith('__')})()
    diagram = []
    connections = []
    
    diagram.append(f"{C.BOLD}{C.CYAN}CLASS STRUCTURE:{C.RESET}")
    diagram.append(f"{C.CYAN}================={C.RESET}")
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
    
    diagram.append(f"\n{C.BOLD}{C.CYAN}CLASS COMMUNICATIONS:{C.RESET}")
    diagram.append(f"{C.CYAN}====================={C.RESET}")
    if connections:
        diagram.append(f"\n{C.YELLOW}Method Calls:{C.RESET}")
        for src, dest, label, _ in sorted(connections, key=lambda x: (x[0], x[1])):
            diagram.append(f"  {C.GREEN}{src.ljust(15)}{C.RESET} {C.BLUE}───[{C.RESET}{label}{C.BLUE}]──→{C.RESET} {C.MAGENTA}{dest}{C.RESET}")
    
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