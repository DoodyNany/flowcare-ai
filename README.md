# Flow Care AI

Five medical AIs, built and trained from scratch. They run inside one web page, with no server and no outside AI service.

| AI | What it does | Model | Data |
|---|---|---|---|
| Bed forecast | Predicts how full each department (ED, IM, Surgery, OB/GYN, ICU) will be 8 hours ahead | Gradient-boosted decision trees | Simulated hospital, 2023–2026 |
| Hajj & Ramadan planner | Predicts ED visits 1–7 days ahead using the Islamic (Umm al-Qura) calendar | Gradient-boosted decision trees | Simulated hospital |
| Patient risk (warn-only) | Chance of deterioration in the next 24 h, with the reasons | Logistic regression | Simulated ward patients |
| Chest X-ray warning (warn-only) | Sorts the reading list so likely pneumonia is read first; heat map of where it looked | 5-layer CNN, 128×128 (JAX) | PneumoniaMNIST: real, de-identified, public |
| Chest CT nodule warning (warn-only) | Flags lung nodules suspicious for cancer for a radiologist; 3D heat map | 3D CNN, 28×28×28 (JAX) | NoduleMNIST3D (LIDC-IDRI): real, de-identified, public |

## Just use it (no install)

Double-click `web/flowcare_ai.html`. It opens in Chrome or Edge and works with **no internet**: the AI models, data and logo are all inside that one file.

## Rebuild and retrain everything on your PC (Windows)

1. Install **Python 3.12** from python.org and tick **"Add python.exe to PATH"** during setup.
2. Optional, to retrain the image AIs: download `pneumoniamnist_128.npz` and `nodulemnist3d.npz` from zenodo.org/records/10519652 and put them in `xray_data/`.
3. Double-click **`run_all.bat`**. It installs the packages, simulates the hospital, trains all the AIs, rebuilds the dashboard and opens it.

To edit the code, use **VS Code** (free) with its Python extension.

| Folder | What's inside |
|---|---|
| `sim/` | The hospital simulation (change bed numbers, Hajj effect, etc.) |
| `models/` | Training code for the four AIs |
| `export/` | Trained models (JSON) and test results |
| `web/` | Dashboard design (`template.html`), in-browser AI code (`models.js`), builder |
| `brand/` | Logo (SVG + PNG) |

## Put it online with a public link (free, optional)

- **GitHub Pages:** create a repository, upload `web/flowcare_ai.html` renamed to `index.html`, then go to Settings → Pages → Deploy from branch.
- **Netlify Drop:** drag the `web` folder onto app.netlify.com/drop.

## Results (tested on data each model never saw)

**Bed forecast, 8 h ahead** (trained on 2023–2025, tested on all of 2026, simulated)

| Dept | AI error | "Same as now" guess | Warned 2 h+ before filling up | False alarms |
|---|---|---|---|---|
| ED | ±5.5 | ±19.8 | 93% | 4% |
| Internal Medicine | ±2.6 | ±5.2 | 87% | 15% |
| Surgery | ±2.1 | ±3.0 | 55% | 28% |
| OB/GYN | ±2.0 | ±3.8 | 77% | 19% |
| ICU | ±1.1 | ±1.4 | 56% | 17% |

**ED planner, 7 days ahead, Hajj 2026:** ±16 visits/day with the Islamic calendar vs ±72 without it.

**Patient risk** (15,000 simulated patients, same alert rate as NEWS2 ≥ 5): AUC 0.83 vs 0.81 for NEWS2; catches 66% of patients who deteriorate vs 60% for NEWS2.

**Chest X-ray** (624 real test X-rays): flags 97.9% of pneumonia cases (8 missed), 82.1% of normal images correctly not flagged, AUC 0.974.

## Limits (say these to the judges)

- The forecast and risk results come from **simulated** data. They show the method works, not how it would perform in a real hospital.
- The X-ray model uses small (64×64) images of **children's** chest X-rays. It is a research prototype, not cleared by the SFDA, and never diagnoses: a physician decides every case.

## Audit log

The **Log** tab records every click, setting change, AI result and human decision, with the real time (Riyadh), the simulated hospital time and the user's name. It is saved in the browser, so it survives a refresh, and it can be downloaded as a CSV file or copied.
