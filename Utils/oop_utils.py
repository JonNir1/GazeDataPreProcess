import inspect


def is_property(obj, attr: str) -> bool:
    """
    Returns True if `attr` is a public property of `obj`.
    see https://stackoverflow.com/a/9058322/8543025
    """
    attributes = inspect.getmembers(obj, lambda a: not (inspect.isroutine(a)))
    public_attributes = [a for a in attributes if not a[0].startswith('_')]
    return attr in [a[0] for a in public_attributes]

