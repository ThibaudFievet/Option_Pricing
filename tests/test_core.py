"""
Suite de tests pour le moteur de calcul core/

Note : On aurait tout aussi bien pu utiliser la bibliothèque standard 'unittest' (qui ne requiert aucune installation externe), mais pytest offre une syntaxe plus lisible et concise
"""

import math
import pytest

from core.instruments import OptionType, VanillaOption
from core.black_and_scholes import black_scholes_price
from core.greeks import calculate_greeks_analytical, calculate_greeks_numerical

def test_call_payoff():
    call = VanillaOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    assert call.payoff(spot_price=120.0) == pytest.approx(20.0) # assert vérifie qu'une condition est vraie : si c'est le cas, l'exécution continue normalement, sinon le programme s'arrête net en levant une erreur AssertionError.
    assert call.payoff(spot_price=100.0) == pytest.approx(0.0) # On utilise pytest.approx pour comparer des float en tolérant une infime marge d'erreur, ce qui évite que le test n'échoue bêtement à cause des imprécisions d'arrondi binaire des ordinateurs.
    assert call.payoff(spot_price=80.0) == pytest.approx(0.0)


def test_put_payoff():
    put = VanillaOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
    assert put.payoff(spot_price=80.0) == pytest.approx(20.0)
    assert put.payoff(spot_price=100.0) == pytest.approx(0.0)
    assert put.payoff(spot_price=120.0) == pytest.approx(0.0)


def test_invalid_parameters():
    """Vérifie que des exceptions ValueError sont bien levées pour des paramètres invalides."""
    with pytest.raises(ValueError):
        VanillaOption(strike=-10.0, expiry=1.0)
    with pytest.raises(ValueError):
        VanillaOption(strike=100.0, expiry=-0.5)

def test_benchmark_prices():
    """Paramètres classiques pris directement depuis le livre Options, Futures, and Other Derivatives de John Hull."""
    S, K, r, sigma, T, q = 49.0, 50.0, 0.05, 0.20, 0.3846, 0.0  # approx 140 jours / 365
    call_price = black_scholes_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=OptionType.CALL)
    put_price = black_scholes_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=OptionType.PUT)
    # Valeurs théoriques attendues selon le livre : Call ~ 2.4005, Put ~ 2.4481
    assert call_price == pytest.approx(2.4005, abs=0.01) # between 2.4005 - 0.01 and 2.4005 + 0.01
    assert put_price == pytest.approx(2.4481, abs=0.01)


def test_put_call_parity_no_dividend():
    """Vérifie la parité Call-Put sans dividende : C - P = S - K * exp(-r * T)."""
    call = black_scholes_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, q=0.0, option_type=OptionType.CALL)
    put = black_scholes_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, q=0.0, option_type=OptionType.PUT)
    expected_diff = 100.0 - 100.0 * math.exp(-0.05 * 1.0)
    assert (call - put) == pytest.approx(expected_diff, abs=0.000001)


def test_put_call_parity_with_dividend():
    """Vérifie la parité Call-Put avec dividende continu q : C - P = S * exp(-q*T) - K * exp(-r*T)."""
    S, K, T, r, sigma, q = 100.0, 95.0, 0.75, 0.04, 0.20, 0.02
    call = black_scholes_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=OptionType.CALL)
    put = black_scholes_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=OptionType.PUT)
    expected_diff = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert (call - put) == pytest.approx(expected_diff, abs=0.000001)


def test_at_maturity():
    """À maturité T=0, le prix de l'option doit égaler son payoff exact."""
    call_price = black_scholes_price(S=110.0, K=100.0, T=0.0, r=0.05, sigma=0.2, option_type=OptionType.CALL)
    assert call_price == 10.0

    put_price = black_scholes_price(S=90.0, K=100.0, T=0.0, r=0.05, sigma=0.2, option_type=OptionType.PUT)
    assert put_price == 10.0

@pytest.fixture # transforme une fonction en un fournisseur de données réutilisable, que pytest injecte automatiquement en argument dans les tests qui en ont besoin pour leur éviter de répéter du code.
def greeks_params():
    return {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.20, "q": 0.02}


def test_greeks_call_signs_and_bounds(greeks_params):
    """Vérifie la cohérence des signes des grecques pour un Call."""
    g = calculate_greeks_analytical(S=greeks_params["S"], K=greeks_params["K"], T=greeks_params["T"], r=greeks_params["r"], sigma=greeks_params["sigma"], q=greeks_params["q"], option_type=OptionType.CALL)
    assert 0.0 < g.delta <= 1.0
    assert g.gamma > 0.0
    assert g.vega > 0.0
    assert g.theta < 0.0
    assert g.rho > 0.0


def test_greeks_put_signs_and_bounds(greeks_params):
    """Vérifie la cohérence des signes des grecques pour un Put."""
    g = calculate_greeks_analytical(S=greeks_params["S"], K=greeks_params["K"], T=greeks_params["T"], r=greeks_params["r"], sigma=greeks_params["sigma"], q=greeks_params["q"], option_type=OptionType.PUT)
    assert -1.0 <= g.delta < 0.0
    assert g.gamma > 0.0
    assert g.theta < 0.0 # Avec les paramètres initiés dans greeks_params, le theta est négatif mais pour des options européennes il pourrait être positif (voir commentaires de calculate_greeks_analytical dans greeks.py)
    assert g.vega > 0.0
    assert g.rho < 0.0


def test_analytical_vs_numerical_greeks_call(greeks_params):
    """Vérifie que les grecques calculées par différences finies convergent vers celles calculées avec les formules analytiques."""
    ga = calculate_greeks_analytical(S=greeks_params["S"],K=greeks_params["K"], T=greeks_params["T"], r=greeks_params["r"], sigma=greeks_params["sigma"], q=greeks_params["q"], option_type=OptionType.CALL)
    gn = calculate_greeks_numerical(S=greeks_params["S"], K=greeks_params["K"], T=greeks_params["T"], r=greeks_params["r"], sigma=greeks_params["sigma"], q=greeks_params["q"], option_type=OptionType.CALL)

    assert ga.delta == pytest.approx(gn.delta, abs=0.001)
    assert ga.gamma == pytest.approx(gn.gamma, abs=0.001)
    assert ga.vega == pytest.approx(gn.vega, abs=0.001)
    assert ga.theta == pytest.approx(gn.theta, abs=0.01)
    assert ga.rho == pytest.approx(gn.rho, abs=0.001)


def test_analytical_vs_numerical_greeks_put(greeks_params):
    """Vérifie la convergence analytique vs numérique pour un Put."""
    ga = calculate_greeks_analytical(S=greeks_params["S"], K=greeks_params["K"], T=greeks_params["T"], r=greeks_params["r"], sigma=greeks_params["sigma"], q=greeks_params["q"], option_type=OptionType.PUT)
    gn = calculate_greeks_numerical(S=greeks_params["S"], K=greeks_params["K"], T=greeks_params["T"], r=greeks_params["r"], sigma=greeks_params["sigma"], q=greeks_params["q"], option_type=OptionType.PUT)

    assert ga.delta == pytest.approx(gn.delta, abs=0.001)
    assert ga.gamma == pytest.approx(gn.gamma, abs=0.001)
    assert ga.vega == pytest.approx(gn.vega, abs=0.001)
    assert ga.theta == pytest.approx(gn.theta, abs=0.01)
    assert ga.rho == pytest.approx(gn.rho, abs=0.001)