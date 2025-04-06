from collections import defaultdict

class DesignPatternDetector:
    """Class responsible for detecting design patterns in code."""
    
    def __init__(self):
        # Common decorator patterns that might indicate design pattern usage
        self.pattern_decorators = {
            'singleton': ['singleton', 'Singleton'],
            'factory': ['factory', 'Factory', 'factory_method'],
            'observer': ['observer', 'Observable', 'event_listener'],
            'command': ['command', 'Command'],
            'strategy': ['strategy', 'Strategy'],
            'adapter': ['adapter', 'Adapter'],
        }

    def detect_patterns(self, classes):
        """
        Detect common design patterns in the analyzed classes.
        
        Args:
            classes (dict): Dictionary containing class analysis data
            
        Returns:
            defaultdict: Dictionary mapping pattern names to lists of class names
        """
        patterns = defaultdict(list)
        
        for cls_name, details in classes.items():
            self._check_singleton_pattern(cls_name, details, patterns)
            self._check_factory_pattern(cls_name, details, patterns)
            self._check_observer_pattern(cls_name, details, patterns)
            self._check_command_pattern(cls_name, details, patterns)
            self._check_strategy_pattern(cls_name, details, patterns)
            self._check_adapter_pattern(cls_name, details, patterns)
        
        return patterns

    def _check_singleton_pattern(self, cls_name, details, patterns):
        """Check if class implements Singleton pattern."""
        has_instance_var = any('_instance' in attr for attr in details.get('attributes', set()))
        has_get_instance = any('get_instance' in method or 'getInstance' in method 
                            for method in details.get('methods', []))
        if has_instance_var and has_get_instance or 'singleton' in details.get('decorator_patterns', set()):
            patterns['Singleton'].append(cls_name)

    def _check_factory_pattern(self, cls_name, details, patterns):
        """Check if class implements Factory pattern."""
        if 'factory_methods' in details and details['factory_methods'] or 'factory' in details.get('decorator_patterns', set()):
            patterns['Factory'].append(cls_name)

    def _check_observer_pattern(self, cls_name, details, patterns):
        """Check if class implements Observer pattern."""
        if ('publishes_events' in details and details['publishes_events'] and
            'subscribes_to_events' in details and details['subscribes_to_events']) or 'observer' in details.get('decorator_patterns', set()):
            patterns['Observer'].append(cls_name)

    def _check_command_pattern(self, cls_name, details, patterns):
        """Check if class implements Command pattern."""
        if any('execute' in method or 'Command' in method for method in details.get('methods', [])):
            patterns['Command'].append(cls_name)

    def _check_strategy_pattern(self, cls_name, details, patterns):
        """Check if class implements Strategy pattern."""
        if any('strategy' in method.lower() for method in details.get('methods', [])):
            patterns['Strategy'].append(cls_name)

    def _check_adapter_pattern(self, cls_name, details, patterns):
        """Check if class implements Adapter pattern."""
        if any('adapt' in method.lower() for method in details.get('methods', [])):
            patterns['Adapter'].append(cls_name)
