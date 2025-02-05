from functools import wraps
from flask import flash, redirect, url_for

def error_handler(error_message="Errore generico"):
    """
    Decorator per gestire le eccezioni nelle route di Flask.
    Se la funzione decorata solleva un'eccezione, viene eseguito il flash
    del messaggio di errore (personalizzato) e viene eseguito un redirect alla route 'index'.
    
    :param error_message: Messaggio di errore personalizzato da mostrare in caso di eccezione.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                flash(f"{error_message}: {str(e)}", "danger")
                return redirect(url_for("index"))
        return wrapper
    return decorator