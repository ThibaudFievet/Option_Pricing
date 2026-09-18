"""
Ce fichier définit les structures de données fondamentales pour représenter les options vanilles.
Il utilise des Enums pour restreindre les valeurs possibles à un ensemble fermé et strict, ce qui élimine les fautes de frappe et offre l'autocomplétion dans l'éditeur
et une Dataclass pour encapsuler les caractéristiques du contrat sans code répétitif, en générant automatiquement le constructeur, un affichage lisible et la comparaison d'objets
"""

from enum import Enum
from dataclasses import dataclass


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"

class ExerciseStyle(str, Enum):
    """
    EUROPEAN : L'option ne peut être exercée qu'à la date d'échéance exacte.
    AMERICAN : L'option peut être exercée à tout moment jusqu'à l'échéance.
    """
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass
class VanillaOption:
    """
    strike (float) : Prix d'exercice convenu (K). Doit être strictement positif (> 0).
    expiry (float) : Temps restant jusqu'à l'échéance (T) exprimé en années.
                     Exemple : 6 mois = 0.5, 3 mois = 0.25.
    option_type (OptionType) : Type d'option, soit OptionType.CALL (par défaut) soit OptionType.PUT.
    exercise_style (ExerciseStyle) : Style d'exercice (par défaut EUROPEAN).
    """
    strike: float
    expiry: float
    option_type: OptionType = OptionType.CALL
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN

    def __post_init__(self):
        """
        Validation élémentaire des paramètres à l'instanciation.
        Note : comme la Dataclass génère automatiquement le __init__, on utilise __post_init__ pour injecter nos vérifications de validité, 
        qui auraient sinon été écrites directement dans un __init__ classique
        """
        if self.strike <= 0:
            raise ValueError(f"Le strike K doit être strictement positif. Reçu : {self.strike}")
        if self.expiry < 0:
            raise ValueError(f"L'échéance T ne peut pas être négative. Reçu : {self.expiry}")

    def payoff(self, spot_price: float):
        """
        Payoff Call = max(S - K, 0)
        Payoff Put  = max(K - S, 0)
        """
        if spot_price < 0:
            raise ValueError(f"Le cours du sous-jacent S ne peut pas être négatif. Reçu : {spot_price}")

        if self.option_type == OptionType.CALL:
            return max(spot_price - self.strike, 0.0)
        else:
            return max(self.strike - spot_price, 0.0)
