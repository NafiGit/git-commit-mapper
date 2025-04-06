from collections import defaultdict
from utils.colors import Colors
from utils.design_patterns import DesignPatternDetector

import os

# Try importing optional dependencies with fallbacks
try:
    import graphviz
except ImportError:
    graphviz = None

try:
    import plantuml
except ImportError:
    plantuml = None
    print("Warning: PlantUML module not found. To enable PlantUML diagrams, run: pip install plantuml six")

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
    
    return "\n".join(diagram)

def generate_graphviz_diagram(classes, output_file, modules=None, format='png'):
    """Generate a GraphViz diagram."""
    if graphviz is None:
        print("Warning: Graphviz Python package not found. To enable Graphviz diagrams, run: pip install graphviz")
        return None
    
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
    
    try:
        dot.render(output_file, cleanup=True)
        return f"{output_file}.{format}"
    except Exception as e:
        print(f"Error rendering GraphViz diagram: {e}")
        return None

def generate_plantuml_diagram(classes, output_file, modules=None, format='png'):
    """Generate a PlantUML diagram."""
    if plantuml is None:
        print("Warning: PlantUML Python package not found. To enable PlantUML diagrams, run: pip install plantuml six")
        return None
    
    # Start PlantUML content
    plantuml_str = ["@startuml"]
    
    # Add skinparam to make it look better
    plantuml_str.append("skinparam classAttributeIconSize 0")
    plantuml_str.append("skinparam classFontStyle bold")
    plantuml_str.append("skinparam classFontSize 14")
    plantuml_str.append("skinparam classBackgroundColor LightBlue")
    plantuml_str.append("skinparam classBorderColor DarkBlue")
    plantuml_str.append("skinparam arrowColor #33658A")
    plantuml_str.append("skinparam packageBackgroundColor WhiteSmoke")
    
    # If modules exist, organize classes by modules
    if modules:
        for module_name, module_info in modules.items():
            # Sanitize module name for PlantUML
            safe_module_name = module_name.replace(".", "_dot_")
            
            # Create a package for this module
            plantuml_str.append(f'package "{module_name}" as {safe_module_name} {{')
            
            # Add classes that belong to this module
            for cls_name in module_info.get('classes', []):
                if cls_name in classes:
                    # Define the class with its methods
                    details = classes[cls_name]
                    plantuml_str.append(f'  class {cls_name} {{')
                    
                    # Add methods
                    for method in details.get('methods', [])[:7]:  # Limit to first 7 for readability
                        plantuml_str.append(f'    +{method}()')
                    
                    # Add attributes/states
                    for state in details.get('states', [])[:5]:  # Limit to first 5 for readability
                        plantuml_str.append(f'    -{state}')
                    
                    # Add props
                    for prop in details.get('props', [])[:5]:  # Limit to first 5 for readability
                        plantuml_str.append(f'    +{prop}')
                    
                    plantuml_str.append('  }')
            
            plantuml_str.append('}')
    else:
        # Add all classes without module organization
        for cls_name, details in classes.items():
            plantuml_str.append(f'class {cls_name} {{')
            
            # Add methods
            for method in details.get('methods', [])[:7]:
                plantuml_str.append(f'  +{method}()')
            
            # Add attributes/states
            for state in details.get('states', [])[:5]:
                plantuml_str.append(f'  -{state}')
            
            # Add props
            for prop in details.get('props', [])[:5]:
                plantuml_str.append(f'  +{prop}')
            
            plantuml_str.append('}')
    
    # Add inheritance relationships
    for cls_name, details in classes.items():
        for parent in details.get('parent_classes', []):
            if parent in classes:
                plantuml_str.append(f'{parent} <|-- {cls_name}')
    
    # Add composition relationships (these were missing in the original implementation)
    for cls_name, details in classes.items():
        for composed in details.get('composed_classes', set()):
            if composed in classes:
                plantuml_str.append(f'{cls_name} *-- {composed} : contains')
    
    # Add method calls (these were missing in the original implementation)
    for cls_name, details in classes.items():
        for call in details.get('calls', set()):
            if call in classes:
                plantuml_str.append(f'{cls_name} ..> {call} : calls')
    
    # Add dependencies (these were missing in the original implementation)
    for cls_name, details in classes.items():
        for dep in details.get('injected_dependencies', set()):
            if dep in classes:
                plantuml_str.append(f'{cls_name} --> {dep} : depends on')
    
    plantuml_str.append("@enduml")
    
    # Write to file
    puml_file = f"{output_file}.puml"
    with open(puml_file, 'w') as f:
        f.write("\n".join(plantuml_str))
    
    # Try generating the diagram using PlantUML
    try:
        plantuml_instance = plantuml.PlantUML()
        plantuml_instance.processes_file(puml_file, outfile=f"{output_file}.{format}")
        return f"{output_file}.{format}"
    except Exception as e:
        print(f"Error generating PlantUML diagram: {e}")
        print(f"PlantUML source file saved to: {puml_file}")
        return puml_file  # Return the puml file as a fallback