"""
Ce module implémente les formules analytiques du modèle de Black-Scholes-Merton pour valoriser des options vanilles européennes.
    
Il prend en compte le taux de dividende continu (q) :
    - Pour une action sans dividende : q = 0.0
    - Pour une action ou un indice avec dividende : q = taux annuel continu
"""

import math
from scipy.stats import norm
from core.instruments import OptionType, VanillaOption


def calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """
    Calcule les variables intermédiaires d1 et d2.
    d1 mesure le nombre d'écarts-types dont le sous-jacent est au-dessus du strike
    (ajusté du drift risque-neutre et de la convexité de l'actif). N(d1) est lié au Delta.
    d2 = d1 - sigma * sqrt(T). N(d2) représente la probabilité risque-neutre que l'option finisse dans la monnaie à l'échéance T.

    d1 = (ln(S / K) + (r - q + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T) ou (ln(S / K) + (r - q - 0.5 * sigma^2) * T) / (sigma * sqrt(T))

    S : Cours actuel du sous-jacent (Spot price), S > 0.
    K : Prix d'exercice de l'option (Strike), K > 0.
    T : Durée résiduelle jusqu'à échéance en années, T > 0.
    r : Taux d'intérêt sans risque annuel (ex: 0.03 pour 3%).
    sigma : Volatilité annuelle du sous-jacent (ex: 0.20 pour 20%), sigma > 0.
    q : Taux de dividende continu annuel (par défaut 0.0).
    """
    if S <= 0:
        raise ValueError(f"Le spot S doit être strictement positif (> 0). Reçu : {S}")
    if K <= 0:
        raise ValueError(f"Le strike K doit être strictement positif (> 0). Reçu : {K}")
    if T <= 0:
        raise ValueError(f"L'échéance T doit être strictement positive (> 0) pour d1/d2. Reçu : {T}")
    if sigma <= 0:
        raise ValueError(f"La volatilité sigma doit être strictement positive (> 0). Reçu : {sigma}")

    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    return d1, d2


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option_type: OptionType = OptionType.CALL):
    """
    Calcule le prix théorique d'une option européenne vanille selon Black-Scholes-Merton.

    Pour un Call :
        Prix = S * exp(-q * T) * N(d1) - K * exp(-r * T) * N(d2)
        Le premier terme représente la valeur attendue du sous-jacent reçu, le second représente le coût actualisé du paiement du strike K.
    Pour un Put :
        Prix = K * exp(-r * T) * N(-d2) - S * exp(-q * T) * N(-d1)
        Inversement, c'est la valeur du cash reçu diminuée de la remise du titre.

    N(x) est la fonction de répartition de la loi normale centrée réduite (scipy.stats.norm.cdf).

    Si T == 0 (maturité atteinte) : retourne le payoff intrinsèque immédiat.
    Si sigma == 0 : l'incertitude disparaît, le prix tend vers la valeur intrinsèque actualisée.
    Note : À maturité (T = 0) ou à volatilité nulle (sigma = 0) : si S > K, d1 et d2 tendent vers +infini (Call dans la monnaie, N = 1) ; 
    si S < K, d1 et d2 tendent vers -infini (Put dans la monnaie car -d tend vers +infini, N(-d) = 1) ; si S == K, d1 et d2 valent 0 (N = 0.5).

    S : Cours du sous-jacent.
    K : Prix d'exercice (Strike).
    T : Temps restant avant échéance (en années).
    r : Taux d'intérêt sans risque annuel.
    sigma : Volatilité annuelle.
    q : Taux de dividende continu (0.0 si aucun dividende).
    option_type : OptionType.CALL ou OptionType.PUT.
    """
    if T <= 0.0:
        if option_type == OptionType.CALL:
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    if sigma <= 0.0:
        forward_price = S * math.exp((r - q) * T)
        discount = math.exp(-r * T)
        if option_type == OptionType.CALL:
            return max(discount * (forward_price - K), 0.0)
        else:
            return max(discount * (K - forward_price), 0.0)

    d1, d2 = calculate_d1_d2(S=S, K=K, T=T, r=r, sigma=sigma, q=q)

    df_q = math.exp(-q * T)  # Facteur d'actualisation du dividende
    df_r = math.exp(-r * T)  # Facteur d'actualisation sans risque

    if option_type == OptionType.CALL:
        price = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    elif option_type == OptionType.PUT:
        price = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    else:
        raise ValueError(f"Type d'option inconnu : {option_type}")

    # Le prix d'une option vanille ne peut pas être inférieur à 0
    return max(price, 0.0)