from functools import wraps

def todo(reason="Tato funkce potřebuje úpravy."):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"TODO ve funkci '{func.__name__}': {reason}")
            return func(*args, **kwargs)
        return wrapper
    return decorator