# Minimal stub for custom Transformers that expects:
#   from visualizer import get_local
#   @get_local('xxx') decorator on functions
#
# This implementation is a NO-OP decorator: it does nothing, but keeps compatibility.

from functools import wraps

def get_local(*names, **kwargs):
    """
    Usage in patched transformers:
        @get_local('attn_weights')
        def forward(...):
            ...

    We return a decorator that returns the original function unchanged.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kw):
            return fn(*args, **kw)
        return wrapper
    return decorator