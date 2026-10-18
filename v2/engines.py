"""
engines.py — Real engine presets and experimental propulsion tech for the v2
takeoff+landing mission.

REAL ENGINES (published specs; sources noted per engine):
    raptor3   — SpaceX Raptor 3, FFSC methalox
    merlin1d  — SpaceX Merlin 1D, gas-generator kerolox
    be4       — Blue Origin BE-4, ORSC methalox
    rutherford— Rocket Lab Rutherford, electric-pump-fed kerolox

EXPERIMENTAL TECH (projected — NOT flight-proven on an operational rocket;
each entry cites the real program it is based on):
    standard  — conventional bell nozzle (baseline, real)
    aerospike — altitude-compensating spike nozzle
    rdre      — rotating-detonation combustor
    rdre_spike— RDRE combustor + aerospike nozzle (conceptual combo)

A "loadout" = one engine + one tech. The mission env sizes a single-engine
hopper around the loadout and the trained policy is conditioned on the
loadout's key parameters, so one policy flies every combination.
"""

from __future__ import annotations

import math

G0 = 9.81

# ---------------------------------------------------------------------------
# Real engines. Fields:
#   thrust_sl  sea-level thrust, N
#   isp_sl/vac specific impulse, s
#   throttle_min  minimum stable throttle, fraction of full
#   gimbal_deg    gimbal range, degrees (estimated where not published)
#   mass       engine dry mass, kg
#   pc_bar     chamber pressure, bar (drives the HUD readout)
# ---------------------------------------------------------------------------
ENGINES = {
    "raptor3": {
        "name": "Raptor 3",
        "maker": "SpaceX",
        "cycle": "Full-flow staged combustion, LOX/CH4",
        "thrust_sl": 2.75e6,
        "isp_sl": 327,
        "isp_vac": 350,
        "throttle_min": 0.40,
        "gimbal_deg": 8,  # estimate; not published
        "mass": 1525,
        "pc_bar": 350,
        "real": True,
        "sources": "Wikipedia 'SpaceX Raptor' (thrust 280 tf SL, Isp 327s SL / 350s vac, "
        "throttle 40-100%, chamber pressure 350 bar, mass 1,525 kg)",
    },
    "merlin1d": {
        "name": "Merlin 1D",
        "maker": "SpaceX",
        "cycle": "Gas generator, LOX/RP-1",
        "thrust_sl": 845e3,
        "isp_sl": 282,
        "isp_vac": 311,
        "throttle_min": 0.57,
        "gimbal_deg": 5,  # estimate; not published
        "mass": 470,
        "pc_bar": 97,
        "real": True,
        "sources": "Wikipedia 'SpaceX Merlin' (845 kN SL, Isp 282s SL / 311s vac, "
        "throttle 57-100% SL, chamber pressure 9.7 MPa, mass 470 kg)",
    },
    "be4": {
        "name": "BE-4",
        "maker": "Blue Origin",
        "cycle": "Oxygen-rich staged combustion, LOX/CH4",
        "thrust_sl": 2.46e6,
        "isp_sl": 340,
        "isp_vac": 340,
        "throttle_min": 0.40,
        "gimbal_deg": 5,  # published: +/-5 deg
        "mass": 5400,
        "pc_bar": 140,
        "real": True,
        "sources": "Wikipedia 'BE-4' (2,460 kN SL, Isp 340 s, throttle 40-100%, "
        "chamber pressure 140 bar, gimbal +/-5 deg, mass 5,400 kg)",
    },
    "rutherford": {
        "name": "Rutherford",
        "maker": "Rocket Lab",
        "cycle": "Electric-pump-fed, LOX/RP-1",
        "thrust_sl": 25e3,
        "isp_sl": 311,
        "isp_vac": 343,
        "throttle_min": 0.50,  # estimate; not published
        "gimbal_deg": 5,  # estimate; not published
        "mass": 35,
        "pc_bar": 100,  # estimate; not published
        "real": True,
        "sources": "Wikipedia 'Rutherford (rocket engine)' (24-25 kN SL, Isp 311s SL / "
        "343s vac, mass 35 kg); throttle/gimbal/chamber pressure estimated",
    },
}

# ---------------------------------------------------------------------------
# Experimental propulsion tech. Each modifies a base engine. All numbers are
# PROJECTED (engineering estimates based on the cited programs), not measured
# flight performance.
# ---------------------------------------------------------------------------
TECHS = {
    "standard": {
        "name": "Standard bell nozzle",
        "projected": False,
        "blurb": "Conventional regeneratively-cooled bell nozzle. Flight-proven baseline.",
    },
    "aerospike": {
        "name": "Aerospike nozzle",
        "projected": True,
        "blurb": "Altitude-compensating spike nozzle: near-vacuum Isp at all altitudes.",
        "basis": "LEAP 71 XRA-2E5 (200 kN methalox aerospike, hot-fire campaign 2026; "
        "20 kN aerospike hot-fired Dec 2025) and Pangea Aerospace MERLIn (Spain, "
        "EUR 11.6M grant). No aerospike engine has flown in space as of 2026.",
        # projected effects
        "isp_altitude_compensation": True,  # Isp_eff = isp_vac at every altitude
        "mass_mult": 1.08,  # spike cooling hardware
        "thrust_mult": 0.98,  # small cosine losses
    },
    "rdre": {
        "name": "Rotating detonation combustor",
        "projected": True,
        "blurb": "Pressure-gain detonation combustor: +15% efficiency (vendor claim).",
        "basis": "Venus Aerospace: world's first high-thrust RDRE flight test "
        "May 2025 (~2,000 lbf, 7 s flight); claims ~15% more efficient than "
        "conventional engines; $91M Series B Jul 2026. NASA ground demo 2022; "
        "JAXA space firing 2021.",
        # projected effects
        "isp_mult": 1.15,
        "thrust_mult": 1.10,
        "throttle_min_override": 0.60,  # detonation stability limit (projected)
        "mass_mult": 1.12,
        "det_freq_khz": 2.4,  # projected nominal detonation frequency (HUD)
    },
    "rdre_spike": {
        "name": "RDRE + aerospike (concept)",
        "projected": True,
        "blurb": "Conceptual combo: detonation combustor with altitude-compensating spike.",
        "basis": "Conceptual combination of the two tracks above; research concepts "
        "exist but no integrated demonstrator has hot-fired as of 2026.",
        "isp_altitude_compensation": True,
        "isp_mult": 1.15,
        "thrust_mult": 1.08,  # 1.10 * 0.98
        "throttle_min_override": 0.60,
        "mass_mult": 1.20,  # 1.08 * 1.12
        "det_freq_khz": 2.4,
    },
}

# ---------------------------------------------------------------------------
# Weather presets. friction mu applies to the pad surface at touchdown.
# ---------------------------------------------------------------------------
WEATHER = {
    "clear": {"name": "Clear", "mu": 0.70, "gust_add": 0.0, "isp_mult": 1.00,
              "fog": 0.0, "precip": None},
    "rain": {"name": "Rain", "mu": 0.40, "gust_add": 2.0, "isp_mult": 0.98,
             "fog": 0.35, "precip": "rain"},
    "snow": {"name": "Snow", "mu": 0.25, "gust_add": 3.0, "isp_mult": 0.97,
             "fog": 0.55, "precip": "snow"},
    "ice": {"name": "Ice", "mu": 0.05, "gust_add": 0.5, "isp_mult": 1.00,
            "fog": 0.10, "precip": None},
}
WEATHER_IDS = ["clear", "rain", "snow", "ice"]

# ---------------------------------------------------------------------------
# Loadout resolution
# ---------------------------------------------------------------------------
LIFTOFF_TW = 1.35   # vehicle wet mass sized for this liftoff thrust/weight
FULL_BURN_TIME = 75.0  # s of full throttle per full tank (mission-sized)


def resolve_loadout(engine_id: str, tech_id: str) -> dict:
    """Combine an engine preset with a tech modifier into a concrete loadout."""
    if engine_id not in ENGINES:
        raise ValueError(f"unknown engine {engine_id!r}")
    if tech_id not in TECHS:
        raise ValueError(f"unknown tech {tech_id!r}")
    eng = ENGINES[engine_id]
    tech = TECHS[tech_id]

    thrust_sl = eng["thrust_sl"] * tech.get("thrust_mult", 1.0)
    isp_sl = eng["isp_sl"] * tech.get("isp_mult", 1.0)
    isp_vac = eng["isp_vac"] * tech.get("isp_mult", 1.0)
    throttle_min = max(
        eng["throttle_min"], tech.get("throttle_min_override", 0.0)
    )
    engine_mass = eng["mass"] * tech.get("mass_mult", 1.0)
    gimbal_rad = math.radians(eng["gimbal_deg"])

    # size the single-engine hopper around the loadout
    wet_mass = thrust_sl / (LIFTOFF_TW * G0)
    mdot_full = thrust_sl / (isp_sl * G0)  # kg/s at full throttle, SL
    fuel_mass = mdot_full * FULL_BURN_TIME
    dry_mass = wet_mass - fuel_mass
    # sanity: dry mass must comfortably exceed the engine itself
    if dry_mass < engine_mass * 2.0:
        dry_mass = engine_mass * 2.0
        wet_mass = dry_mass + fuel_mass

    return {
        "engine_id": engine_id,
        "tech_id": tech_id,
        "engine_name": eng["name"],
        "engine_maker": eng["maker"],
        "tech_name": tech["name"],
        "projected": tech["projected"],
        "thrust_sl": thrust_sl,
        "isp_sl": isp_sl,
        "isp_vac": isp_vac,
        "altitude_compensation": bool(tech.get("isp_altitude_compensation", False)),
        "throttle_min": throttle_min,
        "gimbal_rad": gimbal_rad,
        "engine_mass": engine_mass,
        "pc_bar": eng["pc_bar"],
        "det_freq_khz": tech.get("det_freq_khz"),
        "dry_mass": dry_mass,
        "fuel_mass": fuel_mass,
        "wet_mass": wet_mass,
        "mdot_full": mdot_full,
        "tw_liftoff": thrust_sl / (wet_mass * G0),
        "real": eng["real"],
    }


def list_loadouts():
    return [(e, t) for e in ENGINES for t in TECHS]


if __name__ == "__main__":
    for e, t in list_loadouts():
        lo = resolve_loadout(e, t)
        print(
            f"{lo['engine_name']:12s} + {lo['tech_name'][:28]:28s} "
            f"T/W={lo['tw_liftoff']:.2f} wet={lo['wet_mass']/1000:7.1f}t "
            f"fuel={lo['fuel_mass']/1000:6.1f}t mdot={lo['mdot_full']:7.1f}kg/s "
            f"thr_min={lo['throttle_min']:.2f}"
            f"{' [PROJECTED]' if lo['projected'] else ''}"
        )
