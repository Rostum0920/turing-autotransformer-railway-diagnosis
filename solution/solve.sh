#!/bin/bash
set -euo pipefail

# The deliverable is a reusable program, so the oracle installs the reference solver at the required path ...
mkdir -p /root/results/development
cp /solution/solver.py /root/results/solver.py
chmod +x /root/results/solver.py

# ... and exercises it on the development campaign. The solver reads every campaign file (campaign.json, each line_<ID>.json and
# readings_<ID>.npz), builds the network model, searches the hypotheses and writes the diagnosis.
python3 -B /root/results/solver.py --input-dir /root/data --output-dir /root/results/development

# Structural check of what was computed from the inputs, then comparison with the development key.
python3 - <<'PY'
import json
produced = json.load(open('/root/results/development/diagnosis.json'))['lines']
expected = json.load(open('/root/data/campaign.json'))['lines']
assert sorted(produced) == sorted(expected), 'solver output does not cover the campaign lines'
NUMERIC = ('contact_wire_increases', 'feeder_increases', 'rail_increases', 'rail_leakage_increases', 'voltage_sensor_offsets', 'current_meter_gains')
TOL = {'contact_wire_increases': 0.2, 'feeder_increases': 0.25, 'rail_increases': 0.4, 'rail_leakage_increases': 1.0, 'voltage_sensor_offsets': 20.0, 'current_meter_gains': 4.0}
for lid, e in produced.items():
    assert set(e) == set(NUMERIC) | {'autotransformers_out', 'bad_measurements'}, lid
# The program is developed by running it on L1 and L2 and checking where it disagrees with the key.
key = json.load(open('/root/data/development_key.json'))['lines']
for lid, k in key.items():
    e = produced[lid]
    for s in NUMERIC:
        assert set(e[s]) == set(k[s]) and all(round(abs(e[s][i] - k[s][i]), 9) <= TOL[s] for i in k[s]), (lid, s, e[s], k[s])
    assert e['autotransformers_out'] == k['autotransformers_out'], (lid, 'autotransformers out of service')
    assert e['bad_measurements'] == k['bad_measurements'], (lid, 'bad measurements')
print('development campaign matches the development key:', {lid: {s: e[s] for s in NUMERIC + ('autotransformers_out',)} for lid, e in produced.items()})
PY
