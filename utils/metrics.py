def calculate_metrics(classes):
    """Calculate code quality metrics."""
    metrics = {}
    for cls_name, details in classes.items():
        cls_metrics = {
            'num_methods': len(details['methods']),
            'num_attributes': len(details['attributes']),
            'num_states': len(details['states']),
            'num_props': len(details['props']),
            'efferent_coupling': len(set(details.get('composed_classes', set()))),
            'afferent_coupling': sum(1 for other_cls, other_details in classes.items() 
                                   if cls_name != other_cls and cls_name in other_details.get('composed_classes', set())),
            'instability': 0,
            'abstraction': 0,
            'inheritance_depth': 0
        }
        total_coupling = cls_metrics['efferent_coupling'] + cls_metrics['afferent_coupling']
        cls_metrics['instability'] = round(cls_metrics['efferent_coupling'] / total_coupling, 2) if total_coupling > 0 else 0
        metrics[cls_name] = cls_metrics
    return metrics

def format_metrics(classes, metrics):
    """Format code metrics."""
    if not metrics:
        return "No metrics calculated."
    
    lines = ["CODE QUALITY METRICS:", "=====================", ""]
    for cls_name, cls_metrics in sorted(metrics.items()):
        lines.extend([
            f"Class: {cls_name}",
            "-" * (len(cls_name) + 7),
            f"Size: {cls_metrics['num_methods']} methods, {cls_metrics['num_attributes']} attributes",
            f"Efferent Coupling (CE): {cls_metrics['efferent_coupling']}",
            f"Afferent Coupling (CA): {cls_metrics['afferent_coupling']}",
            f"Instability (I = CE/(CE+CA)): {cls_metrics['instability']:.2f}",
            ""
        ])
    return "\n".join(lines)