# Philippines Generator Map

Builds an interactive map of operating generators in the Philippines from the GEM power facilities dataset in `capacity/data/`.

## Features

- Economist-inspired visual treatment with warm land tone, muted water, and red boundary accents
- Circle markers sized by `Capacity (MW)`
- Marker colors by generator `Type`
- Interactive hover and popup details:
  - Plant / project
  - Unit / phase
  - Capacity
  - Start year
  - Type
  - Owner and operator
  - Location fields
  - GEM Wiki URL (when available)
- Generator-type toggle checkboxes built into the color legend
- In-map legend for generator type colors and marker size scale

## Run

```bash
source .venv/bin/activate
python capacity/map/src/build_philippines_generator_map.py
```

## Output

- `capacity/map/outputs/philippines_generator_map.html`

Open the output HTML file in your browser.
