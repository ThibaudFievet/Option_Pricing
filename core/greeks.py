"""
Ce module calcule les sensibilités (les Grecques) d'une option vanille.
    
Il propose deux méthodes de calcul :
    1. Méthode analytique (formules fermées exactes dérivées de Black-Scholes-Merton).
    2. Méthode numérique par différences finies centrales (bump-and-revalue).
    
Toutes les formules respectent l'extension de Merton avec dividende continu (q).
"""

import math
from dataclasses import dataclass
from scipy.stats import norm
from core.instruments import OptionType
from core.black_and_scholes import calculate_d1_d2, black_scholes_price


@dataclass
class GreeksResult:
    """
    Structure regroupant les grecques d'une option.

    delta : Sensibilité au prix du sous-jacent (dV / dS). C'est aussi la quantité de sous-jacent nécessaire pour couvrir la position.
    gamma : Convexité du prix par rapport au sous-jacent (d²V / dS²). Mesure la vitesse à laquelle le Delta évolue.
    vega :  Sensibilité brute à la volatilité (dV / d_sigma).
    vega_1pct : Sensibilité pour une variation de 1% de volatilité (vega / 100).
    theta : Perte de valeur due au temps par an (dV / dt, généralement négatif).
    theta_1day : Perte de valeur par jour calendaire (theta / 365).
    rho : Sensibilité au taux d'intérêt sans risque (dV / dr).
    rho_1pct : Sensibilité pour une hausse de 1 point de pourcentage de taux (rho / 100).
    """
    delta: float
    gamma: float
    vega: float
    vega_1pct: float
    theta: float
    theta_1day: float
    rho: float
    rho_1pct: float


def calculate_greeks_analytical(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, option_type: OptionType = OptionType.CALL):
    """
    Calcule les grecques analytiques exactes selon le modèle de Black-Scholes-Merton.

    Delta :
        - Acheteur Call : exp(-q * T) * N(d1)  (compris entre 0 et 1 si q=0)
        - Acheteur Put : -exp(-q * T) * N(-d1) (compris entre -1 et 0 si q=0)
    Gamma :
        - Identique pour Call et Put : exp(-q * T) * N'(d1) / (S * sigma * sqrt(T))
        - Toujours positif pour les positions longues vanilles (convexité positive). Toujours négatif pour les positions short (gamma neg).
    Vega :
        - Identique pour Call et Put : S * exp(-q * T) * N'(d1) * sqrt(T)
        - Toujours positif pour un acheteur d'option (la hausse de vol valorise l'option). Le contraire pour un vendeur d'options.
    Theta :
        - Mesure la perte de valeur liée au temps qui s'écoule.
        - Généralement négatif pour une position longue. Généralement car peut etre positif (ex : Put deep ITM à l'échéance ou Call deep ITM avec fort taux de dividende q>r) . 
        - Call : - (S * exp(-q*T) * N'(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-r*T) * N(d2) + q * S * exp(-q*T) * N(d1)
        - Put  : - (S * exp(-q*T) * N'(d1) * sigma) / (2 * sqrt(T)) + r * K * exp(-r*T) * N(-d2) - q * S * exp(-q*T) * N(-d1)
    Rho :
        - Call : K * T * exp(-r * T) * N(d2)
        - Put  : -K * T * exp(-r * T) * N(-d2)
        Positif pour une position longue, négatif pour l'inverse.
        Hausse des taux -> le prix forward augmente et le strike actualisé diminue -> valorise les acheteurs de Call et les vendeurs de Put.

        S : Cours du sous-jacent.
        K : Prix d'exercice (Strike).
        T : Temps jusqu'à l'échéance en années.
        r : Taux sans risque annuel.
        sigma : Volatilité annuelle.
        q : Taux de dividende continu (0 si aucun).
        option_type : OptionType.CALL ou OptionType.PUT (Call par défaut).
    """
    # Option expirée : l'option ne vaut plus que son payoff immédiat.
    if T <= 0.0:
        # Le Delta est la dérivée du payoff par rapport à S : d(Payoff) / dS
        # - Call : Payoff = max(S - K, 0)
        #     * S > K (ITM) : Payoff = S - K  -> d(S - K)/dS = (d(S) / dS) - (d(K) / dS) = 1 - 0 = 1.0 (dérivée d'une constante (S) par rapport à une variable (K) est toujours nulle)
        #     * S < K (OTM) : Payoff = 0      -> d(0)/dS = 0.0
        #     * S == K (ATM): moyenne des pentes / limite N(0) = 0.5
        # - Put : Payoff = max(K - S, 0)
        #     * S < K (ITM) : Payoff = K - S  -> d(K - S)/dS = 0 - 1 = -1.0
        #     * S > K (OTM) : Payoff = 0      -> d(0)/dS = 0.0
        #     * S == K (ATM): moyenne des pentes / limite -N(0) = -0.5
        if option_type == OptionType.CALL:
            delta = 1.0 if S > K else (0.5 if S == K else 0.0)
        else:
            delta = -1.0 if S < K else (-0.5 if S == K else 0.0)
        return GreeksResult(delta=delta, gamma=0.0, vega=0.0, vega_1pct=0.0, theta=0.0, theta_1day=0.0, rho=0.0, rho_1pct=0.0)

    d1, d2 = calculate_d1_d2(S=S, K=K, T=T, r=r, sigma=sigma, q=q)

    df_q = math.exp(-q * T) # Discount factor (dividends and risk free rate)
    df_r = math.exp(-r * T)
    pdf_d1 = norm.pdf(d1) # Probability density function. norm.pdf(x) = (1 / sqrt(2 * pi)) * exp(-0.5 * x^2)
    sqrt_T = math.sqrt(T)

    # 1. Delta
    if option_type == OptionType.CALL:
        delta = df_q * norm.cdf(d1)
    else:
        delta = -df_q * norm.cdf(-d1)

    # 2. Gamma
    gamma = (df_q * pdf_d1) / (S * sigma * sqrt_T)

    # 3. Vega
    vega = S * df_q * pdf_d1 * sqrt_T

    # 4. Theta (exprimé en variation par an)
    decay_term = -(S * df_q * pdf_d1 * sigma) / (2.0 * sqrt_T) # Time decay, perte d'incertitude
    if option_type == OptionType.CALL:
        theta = decay_term - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1)
    else:
        theta = decay_term + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)

    # 5. Rho
    if option_type == OptionType.CALL:
        rho = K * T * df_r * norm.cdf(d2)
    else:
        rho = -K * T * df_r * norm.cdf(-d2)

    return GreeksResult(delta=delta, gamma=gamma, vega=vega, vega_1pct=vega / 100.0, theta=theta, theta_1day=theta / 365.0, rho=rho, rho_1pct=rho / 100.0)


def calculate_greeks_numerical(S: float, K: float,T: float, r: float, sigma: float, q: float = 0.0, option_type: OptionType = OptionType.CALL,
    dS: float = 0.001, # 0.1 centime d'euro pour créer un décalage qui capture la tangente locale sans faire exploser le bruit d'arrondi machine lors de la division par dS^2 pour le Gamma.
    dsigma: float = 0.0001, # 1 point de base (0.01% de vol) pour mesurer la pente marginale du Vega sans déformer la dynamique non linéaire de la volatilité.
    dT: float = 0.0001, # environ 52 minutes qui correspond à un horizon suffisamment court pour capturer fidèlement la vitesse d'érosion instantanée sans trop s'éloigner de la maturité actuelle.
    dr: float = 0.0001 # 1 point de base (0.01% de taux d'intérêt) pour mesurer le Rho de façon réaliste et stable.
):
    """
    Calcule les grecques par différences finies centrales (bump-and-revalue).
    Note : Cette méthode m'a été apprise lors de mon stage à Aurora

    Au lieu d'utiliser les formules différentielles théoriques, on décale légèrement chaque paramètre d'un petit montant (bump h) et on recalcule le prix de l'option :
    - Dérivée première centrale : f'(x) approx = (f(x + h) - f(x - h)) / (2 * h)
    - Dérivée seconde centrale   : f''(x) approx = (f(x + h) - 2 * f(x) + f(x - h)) / (h^2)

    Cette méthode utilise le pricer Black-Scholes-Merton.

    S, K, T, r, sigma : Paramètres obligatoires de marché et du contrat.
    q, option_type    : Dividende (0.0 par défaut) et type d'option (CALL par défaut).
    dS, dsigma, dT, dr : Amplitudes des décalages numériques (bumps).
    """
    # Prix de base
    p_base = black_scholes_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)

    # 1. Delta et Gamma (choc sur le spot S)
    p_up_S = black_scholes_price(S=S + dS, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)
    p_dn_S = black_scholes_price(S=S - dS, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)
    delta_num = (p_up_S - p_dn_S) / (2.0 * dS)
    gamma_num = (p_up_S - 2.0 * p_base + p_dn_S) / (dS ** 2)

    # 2. Vega (choc sur sigma)
    p_up_vol = black_scholes_price(S=S, K=K, T=T, r=r, sigma=sigma + dsigma, q=q, option_type=option_type)
    p_dn_vol = black_scholes_price(S=S, K=K, T=T, r=r, sigma=max(sigma - dsigma, 1e-6), q=q, option_type=option_type)
    vega_num = (p_up_vol - p_dn_vol) / (2.0 * dsigma)

    # 3. Theta (choc sur le temps T qui s'écoule)
    if T > dT: # Évite une maturité négative (T - dT < 0) qui ferait planter Black-Scholes si l'option est trop proche de l'échéance.
        p_prev_T = black_scholes_price(S=S, K=K, T=T - dT, r=r, sigma=sigma, q=q, option_type=option_type)
        p_next_T = black_scholes_price(S=S, K=K, T=T + dT, r=r, sigma=sigma, q=q, option_type=option_type)
        theta_num = -(p_next_T - p_prev_T) / (2.0 * dT)
    else:
        theta_num = 0.0

    # 4. Rho (choc sur le taux d'intérêt r)
    p_up_r = black_scholes_price(S=S, K=K, T=T, r=r + dr, sigma=sigma, q=q, option_type=option_type)
    p_dn_r = black_scholes_price(S=S, K=K, T=T, r=r - dr, sigma=sigma, q=q, option_type=option_type)
    rho_num = (p_up_r - p_dn_r) / (2.0 * dr)

    return GreeksResult(delta=delta_num, gamma=gamma_num, vega=vega_num, vega_1pct=vega_num / 100.0, theta=theta_num, theta_1day=theta_num / 365.0,rho=rho_num, rho_1pct=rho_num / 100.0)