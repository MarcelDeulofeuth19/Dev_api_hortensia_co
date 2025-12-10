# auth_token.py
import json
import logging

class AuthToken:
    """
    Clase para manejar tokens de autenticación.
    """

    def __init__(self, filename):
        self.filename = filename
        self.tokens = self.load_tokens()

    def load_tokens(self):
        try:
            with open(self.filename, 'r') as file:
                data = json.load(file)
                return data.get("tokens", [])
        except FileNotFoundError:
            logging.error("Archivo de tokens no encontrado: %s", self.filename)
            return []
        except json.JSONDecodeError:
            logging.error("Error al decodificar el archivo JSON: %s", self.filename)
            return []

    def verify_token(self, token):
        """
        Verifica si un token está en la lista de tokens cargados.
        """
        if not isinstance(token, str):
            logging.error("El token proporcionado no es una cadena de texto: %s", token)
            return False
        return token in self.tokens
