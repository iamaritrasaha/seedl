"""Example: Simulate open quantum system decoherence under Kraus noise channels."""

from __future__ import annotations

import seedl as sd


def main():
    print("Simulating Quantum Decoherence under Depolarizing Noise Channel...")
    viz_depol = sd.quantum_noise(
        channel="depolarizing",
        param=0.4,
        steps=10,
        title="Depolarizing Channel Simulation",
    )
    path_depol = viz_depol.save("quantum_depolarizing.html")
    print(f"Depolarizing simulation saved to: {path_depol}")

    print("Simulating Quantum Dephasing (Phase Damping)...")
    viz_deph = sd.quantum_noise(
        channel="dephasing",
        param=0.5,
        steps=10,
        title="Phase Damping Channel Simulation",
    )
    path_deph = viz_deph.save("quantum_dephasing.html")
    print(f"Phase damping simulation saved to: {path_deph}")


if __name__ == "__main__":
    main()
