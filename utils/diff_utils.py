def diff_snapshots(old_snapshot, new_snapshot):
    """Generate a diff between two snapshots."""
    diff = {
        'added_classes': list(set(new_snapshot.keys()) - set(old_snapshot.keys())),
        'removed_classes': list(set(old_snapshot.keys()) - set(new_snapshot.keys())),
        'modified_classes': {}
    }
    
    for class_name in set(old_snapshot.keys()) & set(new_snapshot.keys()):
        old_class = old_snapshot[class_name]
        new_class = new_snapshot[class_name]
        changes = {}
        
        if set(new_class['methods']) != set(old_class['methods']):
            changes['methods'] = {
                'added': list(set(new_class['methods']) - set(old_class['methods'])),
                'removed': list(set(old_class['methods']) - set(new_class['methods']))
            }
        
        if changes:
            diff['modified_classes'][class_name] = changes
    
    return diff

def format_diff(diff, old_commit_id, new_commit_id):
    """Format a diff as readable ASCII text."""
    lines = [
        f"DIFF BETWEEN {old_commit_id[:7]} AND {new_commit_id[:7]}",
        "=" * 50,
        ""
    ]
    
    if diff['added_classes']:
        lines.append("ADDED CLASSES:")
        for class_name in sorted(diff['added_classes']):
            lines.append(f"  + {class_name}")
        lines.append("")
    
    if diff['removed_classes']:
        lines.append("REMOVED CLASSES:")
        for class_name in sorted(diff['removed_classes']):
            lines.append(f"  - {class_name}")
        lines.append("")
    
    if diff['modified_classes']:
        lines.append("MODIFIED CLASSES:")
        for class_name, changes in sorted(diff['modified_classes'].items()):
            lines.append(f"  * {class_name}:")
            if 'methods' in changes:
                if changes['methods']['added']:
                    lines.append("      Added methods:")
                    for method in sorted(changes['methods']['added']):
                        lines.append(f"        + {method}()")
                if changes['methods']['removed']:
                    lines.append("      Removed methods:")
                    for method in sorted(changes['methods']['removed']):
                        lines.append(f"        - {method}()")
            lines.append("")
    
    return "\n".join(lines)