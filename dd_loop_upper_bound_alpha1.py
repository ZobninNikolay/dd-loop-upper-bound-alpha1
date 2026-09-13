#!/usr/bin/env python3
"""V2.0-alpha-1: strict upper bound for a D-D fusion feedback loop.

The module answers a deliberately narrow question: can energy released by
D-D reactions recreate the energetic-deuteron population that caused those
reactions?  The default ``absolute_upper`` mode assumes 100% capture of all
fusion-product energy and 100% conversion into the next deuteron tail.  It
therefore overestimates, never underestimates, the physically useful feedback.

Reference cross sections follow Bosch & Hale, Nuclear Fusion 32 (1992) 611,
DOI 10.1088/0029-5515/32/4/I07.  Energies passed to the fit are centre-of-mass
energies in keV; the fit returns millibarns.

The code uses only the Python standard library and runs on a laptop.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal


ELEMENTARY_CHARGE_C = 1.602_176_634e-19
DEUTERON_MASS_KG = 3.343_583_7768e-27
KEV_TO_J = 1.0e3 * ELEMENTARY_CHARGE_C
MEV_TO_J = 1.0e6 * ELEMENTARY_CHARGE_C
MILLIBARN_TO_M2 = 1.0e-31

Topology = Literal["beam_target", "head_on_equal_beams"]
ReturnMode = Literal["absolute_upper", "charged_products_only"]


@dataclass(frozen=True)
class BoschHaleBranch:
    name: str
    b_g: float
    a: tuple[float, float, float, float, float]
    b: tuple[float, float, float, float]
    q_total_mev: float
    q_charged_mev: float

    def cross_section_m2(self, e_cm_kev: float) -> float:
        """Return branch cross section in m^2 for E_cm in keV."""
        if e_cm_kev <= 0.0:
            return 0.0
        e = e_cm_kev
        a1, a2, a3, a4, a5 = self.a
        b1, b2, b3, b4 = self.b
        numerator = a1 + e * (a2 + e * (a3 + e * (a4 + e * a5)))
        denominator = 1.0 + e * (b1 + e * (b2 + e * (b3 + e * b4)))
        s_kev_millibarn = numerator / denominator
        sigma_millibarn = (s_kev_millibarn / e) * math.exp(-self.b_g / math.sqrt(e))
        return max(0.0, sigma_millibarn * MILLIBARN_TO_M2)


DD_N_HE3 = BoschHaleBranch(
    name="D(d,n)3He",
    b_g=31.3970,
    a=(5.3701e4, 3.3027e2, -1.2706e-1, 2.9327e-5, -2.5151e-9),
    b=(0.0, 0.0, 0.0, 0.0),
    q_total_mev=3.268,
    q_charged_mev=0.820,
)

DD_P_T = BoschHaleBranch(
    name="D(d,p)T",
    b_g=31.3970,
    a=(5.5576e4, 2.1054e2, -3.2638e-2, 1.4987e-6, 1.8181e-10),
    b=(0.0, 0.0, 0.0, 0.0),
    q_total_mev=4.033,
    q_charged_mev=4.033,
)

DD_BRANCHES = (DD_N_HE3, DD_P_T)


@dataclass(frozen=True)
class EnergyLedger:
    tail_inventory_j: float
    nuclear_release_j: float
    useful_feedback_j: float
    unreturned_product_energy_j: float
    numerical_residual_j: float

    @property
    def closure_fraction(self) -> float:
        scale = max(abs(self.nuclear_release_j), 1.0e-300)
        return abs(self.numerical_residual_j) / scale


@dataclass(frozen=True)
class LoopResult:
    topology: str
    return_mode: str
    tail_energy_kev: float
    e_cm_kev: float
    density_m3: float
    confinement_s: float
    volume_m3: float
    relative_speed_m_s: float
    sigma_total_m2: float
    sigma_v_m3_s: float
    reaction_rate_m3_s: float
    reactions_per_s: float
    fusion_power_w: float
    tail_inventory_j: float
    useful_feedback_power_w: float
    k_loop_max: float
    required_n_tau_m3_s: float
    required_confinement_s: float
    energy_ledger: EnergyLedger


def centre_of_mass_energy_kev(tail_energy_kev: float, topology: Topology) -> float:
    if topology == "beam_target":
        return 0.5 * tail_energy_kev
    if topology == "head_on_equal_beams":
        return 2.0 * tail_energy_kev
    raise ValueError(f"Unsupported topology: {topology}")


def pair_count_factor(topology: Topology) -> float:
    """Factor in R/V = factor * n^2 * sigma*v for the idealized case.

    beam_target treats n as both the dilute tail normalization and an
    undepleted target density when forming an illustrative volume rate; its
    per-tail loop coefficient is independent of tail fraction.  The equal-beam
    case uses the identical-particle 1/2 factor, which is already optimistic
    because every encounter is assigned the head-on relative energy.
    """
    return 1.0 if topology == "beam_target" else 0.5


def relative_speed_m_s(e_cm_kev: float) -> float:
    reduced_mass = 0.5 * DEUTERON_MASS_KG
    return math.sqrt(2.0 * e_cm_kev * KEV_TO_J / reduced_mass)


def branch_quantities(e_cm_kev: float, return_mode: ReturnMode):
    values = []
    for branch in DD_BRANCHES:
        sigma = branch.cross_section_m2(e_cm_kev)
        q_mev = branch.q_total_mev if return_mode == "absolute_upper" else branch.q_charged_mev
        values.append((branch, sigma, q_mev * MEV_TO_J))
    return values


def calculate_loop_upper_bound(
    tail_energy_kev: float,
    density_m3: float,
    confinement_s: float,
    volume_m3: float,
    topology: Topology = "head_on_equal_beams",
    return_mode: ReturnMode = "absolute_upper",
    feedback_efficiency: float = 1.0,
) -> LoopResult:
    """Calculate an optimistic one-cycle feedback-energy upper bound."""
    if tail_energy_kev <= 0.0:
        raise ValueError("tail_energy_kev must be positive")
    if density_m3 < 0.0 or confinement_s < 0.0 or volume_m3 <= 0.0:
        raise ValueError("density/confinement must be nonnegative and volume positive")
    if not 0.0 <= feedback_efficiency <= 1.0:
        raise ValueError("feedback_efficiency must be between 0 and 1")

    e_cm = centre_of_mass_energy_kev(tail_energy_kev, topology)
    speed = relative_speed_m_s(e_cm)
    branches = branch_quantities(e_cm, return_mode)
    sigma_total = sum(sigma for _, sigma, _ in branches)
    sigma_v_total = sigma_total * speed
    factor = pair_count_factor(topology)

    rate_density = factor * density_m3**2 * sigma_v_total
    reactions_per_s = rate_density * volume_m3
    total_fusion_power = sum(
        factor * density_m3**2 * sigma * speed * volume_m3 * branch.q_total_mev * MEV_TO_J
        for branch, sigma, _ in branches
    )
    if return_mode == "absolute_upper":
        # Reuse the identical arithmetic path so eta=1 closes exactly.
        useful_feedback_power = feedback_efficiency * total_fusion_power
    else:
        useful_feedback_power = feedback_efficiency * sum(
            factor * density_m3**2 * sigma * speed * volume_m3 * q_return_j
            for _, sigma, q_return_j in branches
        )

    tail_inventory = density_m3 * volume_m3 * tail_energy_kev * KEV_TO_J
    useful_feedback_energy = useful_feedback_power * confinement_s
    nuclear_release = total_fusion_power * confinement_s
    unreturned = nuclear_release - useful_feedback_energy
    if abs(unreturned) < 1.0e-14 * max(abs(nuclear_release), 1.0e-300):
        unreturned = 0.0
    residual = nuclear_release - useful_feedback_energy - unreturned
    ledger = EnergyLedger(
        tail_inventory_j=tail_inventory,
        nuclear_release_j=nuclear_release,
        useful_feedback_j=useful_feedback_energy,
        unreturned_product_energy_j=unreturned,
        numerical_residual_j=residual,
    )
    k_loop = useful_feedback_energy / tail_inventory if tail_inventory > 0.0 else 0.0

    weighted_q_return = sum(sigma * q for _, sigma, q in branches)
    gain_per_n_tau = factor * speed * weighted_q_return / (tail_energy_kev * KEV_TO_J)
    gain_per_n_tau *= feedback_efficiency
    required_n_tau = math.inf if gain_per_n_tau <= 0.0 else 1.0 / gain_per_n_tau
    required_confinement = math.inf if density_m3 <= 0.0 else required_n_tau / density_m3

    return LoopResult(
        topology=topology,
        return_mode=return_mode,
        tail_energy_kev=tail_energy_kev,
        e_cm_kev=e_cm,
        density_m3=density_m3,
        confinement_s=confinement_s,
        volume_m3=volume_m3,
        relative_speed_m_s=speed,
        sigma_total_m2=sigma_total,
        sigma_v_m3_s=sigma_v_total,
        reaction_rate_m3_s=rate_density,
        reactions_per_s=reactions_per_s,
        fusion_power_w=total_fusion_power,
        tail_inventory_j=tail_inventory,
        useful_feedback_power_w=useful_feedback_power,
        k_loop_max=k_loop,
        required_n_tau_m3_s=required_n_tau,
        required_confinement_s=required_confinement,
        energy_ledger=ledger,
    )


def logarithmic_grid(low: float, high: float, count: int) -> list[float]:
    if low <= 0.0 or high <= low or count < 2:
        raise ValueError("invalid logarithmic grid")
    step = math.log(high / low) / (count - 1)
    return [low * math.exp(i * step) for i in range(count)]


def scan_strict_bound(
    density_m3: float,
    confinement_s: float,
    volume_m3: float,
    energy_min_kev: float = 0.2,
    energy_max_kev: float = 100.0,
    grid_points: int = 2001,
) -> tuple[LoopResult, list[LoopResult]]:
    results: list[LoopResult] = []
    for topology in ("beam_target", "head_on_equal_beams"):
        for energy in logarithmic_grid(energy_min_kev, energy_max_kev, grid_points):
            results.append(
                calculate_loop_upper_bound(
                    energy,
                    density_m3,
                    confinement_s,
                    volume_m3,
                    topology=topology,
                    return_mode="absolute_upper",
                    feedback_efficiency=1.0,
                )
            )
    return max(results, key=lambda item: item.k_loop_max), results


def result_to_dict(result: LoopResult) -> dict:
    data = asdict(result)
    return data


def write_outputs(output_dir: Path, summary: dict, rows: Iterable[LoopResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "alpha1_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)

    fieldnames = [
        "topology", "return_mode", "tail_energy_kev", "e_cm_kev", "density_m3",
        "confinement_s", "volume_m3", "sigma_total_m2", "sigma_v_m3_s",
        "reactions_per_s", "fusion_power_w", "tail_inventory_j",
        "useful_feedback_power_w", "k_loop_max", "required_n_tau_m3_s",
        "required_confinement_s",
    ]
    with (output_dir / "alpha1_table.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            raw = result_to_dict(row)
            writer.writerow({key: raw[key] for key in fieldnames})


def run_default_study(output_dir: Path) -> dict:
    density = 1.62e16
    confinement = 20.0e-6
    radius = 3.259053e-3
    length = 2.998427
    volume = math.pi * radius**2 * length
    energies = (0.2, 1.0, 3.0, 10.0, 30.0, 100.0)

    strict_best, _ = scan_strict_bound(density, confinement, volume)
    table_rows = []
    for topology in ("beam_target", "head_on_equal_beams"):
        for energy in energies:
            table_rows.append(
                calculate_loop_upper_bound(
                    energy, density, confinement, volume, topology=topology,
                    return_mode="absolute_upper", feedback_efficiency=1.0,
                )
            )

    time_sensitivity = []
    for tau in (121e-9, 20e-6, 1e-3, 1.0):
        time_sensitivity.append(
            calculate_loop_upper_bound(
                100.0, density, tau, volume,
                topology="head_on_equal_beams", return_mode="absolute_upper",
                feedback_efficiency=1.0,
            )
        )

    density_sensitivity = []
    for n in (1.62e16, 3.0e16, 1.0e18, 1.0e20):
        density_sensitivity.append(
            calculate_loop_upper_bound(
                100.0, n, confinement, volume,
                topology="head_on_equal_beams", return_mode="absolute_upper",
                feedback_efficiency=1.0,
            )
        )

    charged_only = calculate_loop_upper_bound(
        100.0, density, confinement, volume,
        topology="head_on_equal_beams", return_mode="charged_products_only",
        feedback_efficiency=1.0,
    )

    summary = {
        "module": "2.0-alpha-1",
        "model_status": "strict optimistic upper bound; not a self-consistent plasma simulation",
        "default_parameters": {
            "density_m3": density,
            "confinement_s": confinement,
            "radius_m": radius,
            "length_m": length,
            "volume_m3": volume,
            "energy_range_kev": [0.2, 100.0],
        },
        "strict_best": result_to_dict(strict_best),
        "charged_products_only_at_best_energy": result_to_dict(charged_only),
        "time_sensitivity": [result_to_dict(item) for item in time_sensitivity],
        "density_sensitivity": [result_to_dict(item) for item in density_sensitivity],
        "decision": (
            "NO_GO_SELF_SUSTAINING_DD_AT_CURRENT_PARAMETERS"
            if strict_best.k_loop_max < 1.0 else "CONTINUE_REFINEMENT"
        ),
    }
    write_outputs(output_dir, summary, table_rows)
    return summary


def _safe_json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json_value(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("alpha1_results"))
    args = parser.parse_args()
    summary = _safe_json_value(run_default_study(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
