# Progress Summary — Market & Commodities Intelligence Dashboard

Written before a context compaction, so a fresh session can pick this up
cold. Delete this file once the in-progress work below is finished and
committed — it's a working note, not a permanent project doc (that's
`docs/roadmap.md`).

## Project location

`C:\Users\Nakul Grover\OneDrive\Desktop\market_dashboard` — a git repo,
**no remote configured yet** (user said yes to creating one and pushing,
but that hadn't happened before this interruption — see "Outstanding" below).

## What this project is

A probabilistic forecasting dashboard for 23 NSE stocks/indices/gold-silver
ETF proxies: `P(forward N-day return >= k*volatility*sqrt(N))`, N in
{5, 20} days, via a tuned LightGBM model, served through a FastAPI backend
and a React frontend. Built as a mentorship-style, step-by-step project
with the user (who wanted to learn/defend every methodology choice), with
heavy emphasis on honest reporting of negative results and real bugs found
along the way. Full history and every finding is in `docs/roadmap.md` —
that file is the authoritative methodology log, always read it fresh
rather than trusting this summary's characterization of old phases.

## Fully complete and committed (Phases 0–11, 15 commits, `master` branch)

All of this is committed, tested, and working. Not in progress — background only:

1. **Phase 0**: scaffold, target definition, FastAPI health check.
2. **Phase 1**: yfinance ingestion for the 23-instrument NSE universe
   (`backend/app/data/universe.py`, `ingest.py`).
3. **Phase 2a/2b/2c**: cleaning (`clean.py`), technical indicators
   (`features/technical.py`), cross-asset features (`features/build.py`,
   `cross_asset.py`).
4. **Phase 3**: volatility-scaled labels (`features/labels.py`) —
   **just modified, see "In-progress work" below**.
5. **Phase 4a/b/c/d**: pooled dataset (`models/dataset.py` — **just
   modified**), purged walk-forward CV (`models/cv.py`), logistic
   regression baseline (`models/baseline.py`), a diagnostic detour
   (`models/diagnostics.py` — **just modified**) that revised the label
   threshold from k=0.5 (no signal) to k=1.5 (real signal, ROC-AUC
   ~0.61-0.64).
6. **Phase 5**: tuned LightGBM (`models/gbm.py`) — first model to beat
   naive baseline on Brier score, not just ranking.
7. **Phase 6**: calibration (`models/calibration.py`) — genuine **negative
   result**: Platt/isotonic don't help; raw LightGBM probabilities are
   already reasonably calibrated. Documented as-is, not forced positive.
8. **Phase 7**: SHAP explainability (`models/explain.py`) — found
   `realized_vol_20d` dominates importance ~4x, revealing the model's edge
   is partly volatility-regime detection, not pure direction.
9. **Phase 8**: regime-stratified backtest + cost-aware decision-value
   backtest (`models/backtest.py`) — confirmed the SHAP finding
   quantitatively; edge is real but concentrated in the most confident
   ~5-10% of predictions.
10. **Phase 10a/b**: FastAPI serving layer (`app/api/`, `app/services/model_registry.py`,
    `app/models/persist.py`) + React/TypeScript/Vite frontend (`frontend/`)
    with a dashboard (instrument list + horizon toggle) and detail page
    (hero probability + SHAP chart). Verified end-to-end in a real browser
    against the real running backend.
11. **Phase 11**: CI (`.github/workflows/ci.yml`, verified locally — no
    remote to actually run it on yet), `backend/Dockerfile` (untested, no
    local Docker), `docs/deployment.md`. Found and fixed a real
    portability bug: `paths.py` computed the data directory relative to
    its own file location, which breaks silently in a container — fixed
    with a `MARKET_DATA_ROOT` env var. Same pattern applied to CORS
    (`MARKET_CORS_ORIGINS`). Rewrote the stale top-level `README.md` into
    a real front door (results table, embedded plots, Mermaid diagram).

Test count as of the last full clean run: **133 passing** (before the
in-progress edits below; expect this number to have grown once the
in-progress work's own tests are added).

## In-progress work: adding a genuine downside ("bearish") model

### Why (don't skip this — it's a real methodology point, not just a feature request)

After Phase 11, the user asked to see the dashboard, then said they didn't
like that the single probability number doesn't show "bullish or bearish."

**The problem**: every existing model is upside-only —
`P(forward return >= +threshold)`. A low number does NOT mean "bearish" —
it could mean "calm/neutral" just as easily, since there has never been a
downside model. Relabeling the existing number as bullish/bearish would
have been methodologically dishonest (exactly the kind of shortcut this
whole project has spent 11 phases avoiding).

**Presented two options** via AskUserQuestion:
1. Build a genuine downside model (`P(forward return <= -threshold)`),
   same threshold magnitude, reusing all existing infra, showing both
   probabilities + a transparent derived "lean" on the dashboard.
2. Just relabel the UI without new modeling.

**User chose option 1** (build the real downside model). This is real,
substantial new work — not a small tweak — spanning backend labels →
dataset → CV/tuning → persisted models → API → frontend.

### Design decided (not all implemented yet — see checklist below)

- **Labels** (`backend/app/features/labels.py`): `build_labels_for_instrument`
  now computes both `label_{n}d` (upside, unchanged) and a new
  `label_down_{n}d` (downside: `forward_return <= -threshold`) off the
  *same* symmetric volatility-scaled threshold. A row can satisfy at most
  one of the two (verified by a new test). **DONE, tested, 13/13 tests
  pass** (`test_label_down_is_one_iff_forward_return_meets_negative_threshold`,
  `test_label_and_label_down_are_never_both_true`, plus updated
  `test_build_labels_basic_shape` / `test_label_is_na_when_forward_return_undefined`).
  **NOT YET regenerated on disk** — the persisted label parquet files in
  `data/labels/*.parquet` still only have the old schema; need to re-run
  `python -m app.features.labels` to add the `label_down_{n}d` /
  columns before any downside model can actually be trained.

- **Dataset assembly** (`backend/app/models/dataset.py`): added
  `_label_column_name(horizon, direction)` helper and a `direction: str =
  "up"` parameter threaded through `assemble_panel(...)` and
  `load_modeling_dataset(...)`. `direction="down"` picks
  `label_down_{horizon}d` as the panel's `label` column; `label_end_date`
  is direction-independent (same column either way). **DONE but NOT YET
  TESTED** — I was in the middle of reading `tests/test_dataset.py` to add
  direction-aware tests when interrupted (need tests verifying
  `direction="down"` picks the right column, defaults still work, and
  probably a test that direction="down" and direction="up" panels have
  different `label` values from the same input data).

- **`backend/app/models/diagnostics.py`**: only a docstring fix so far
  (stale "k=0.5" reference corrected to match the k=1.5 reality) — did
  NOT add a `direction` param here, deliberately, since this diagnostic
  tool is about historical k-selection exploration, not the new feature;
  leaving it upside-only (default) is fine.

- **`backend/tests/test_labels.py`**: updated and passing (13/13).

### NOT YET DONE — exact next steps, in order

1. **Finish `test_dataset.py`**: add tests for the `direction` parameter
   (e.g. `test_direction_down_uses_label_down_column`,
   `test_direction_up_is_default`, maybe `test_invalid_direction_raises`
   for `_label_column_name`'s ValueError branch). Run
   `pytest tests/test_dataset.py tests/test_labels.py tests/test_diagnostics.py -v`
   to confirm nothing broke, then the **full suite**
   (`pytest -q` from `backend/`) before moving on.

2. **Regenerate labels on disk**: `python -m app.models.labels`... actually
   the module is `python -m app.features.labels` — re-run it (from
   `backend/`, with the venv) to add `label_down_{5,20}d` columns to every
   instrument's persisted label parquet. This is required before any
   downside training can happen. Sanity-check the resulting downside
   positive rates the same way Phase 4d did for upside (they might differ
   meaningfully — equities have known negative skew / vol clustering on
   drops, so don't be surprised if down-rates run somewhat higher than
   up-rates at the same k — that's a legitimate finding to note if it
   shows up, not a bug).

3. **Retrain for both directions**: `backend/app/models/gbm.py`'s
   `tune_lgbm_hyperparameters` / `build_lgbm_pipeline` don't need any
   changes — they just consume whatever panel `load_modeling_dataset(...,
   direction=...)` produces. Need to decide: re-run the full walk-forward
   evaluation (Phase 5-style) for the downside target too, at least once,
   to get real Brier/AUC numbers for the downside model before trusting
   it — don't skip straight to persisting an unevaluated model. Consider
   whether calibration (Phase 6) and SHAP (Phase 7) should be re-checked
   for downside too, or whether that's over-scoping for what the user
   actually asked for (probably: at minimum re-run tuning + a quick
   walk-forward check; full calibration/backtest re-verification for
   downside is a judgment call to make with the user if it looks like it
   might change the "safe to ship" conclusion).

4. **Persist 4 models total**: update `backend/app/models/persist.py` to
   loop over `direction in ("up", "down")` x `horizon in (5, 20)`, naming
   artifacts distinctly (e.g. `lightgbm_up_5d.joblib`,
   `lightgbm_down_5d.joblib`, etc. — up_5d/up_20d replace the current
   `lightgbm_5d.joblib`/`lightgbm_20d.joblib` naming, so either rename
   consistently or keep old names for "up" and add new ones for "down" —
   pick one convention and update `model_registry.py` to match).

5. **`backend/app/services/model_registry.py`**: currently keys
   `self._artifacts` by `horizon` only. Needs to key by `(direction,
   horizon)` tuple (or two parallel dicts). `predict_and_explain(...)`
   needs a `direction` parameter. Consider adding a convenience method
   that returns both directions' predictions+explanations in one call
   (the frontend will want both at once for the "lean" display) —
   probably cleaner than making the frontend issue two separate requests.

6. **API layer** (`backend/app/api/schemas.py`, `routes.py`): decide and
   implement the contract. Leaning towards: `Prediction`/`PredictionDetail`
   schemas gain `up_probability` + `down_probability` (and
   `up_top_features` + `down_top_features` for the detail endpoint),
   computed by one registry call per direction, combined into one
   response — so the frontend gets the full picture in a single request
   per ticker/horizon, matching how it'll actually be displayed (both
   numbers + a derived lean, never just one). Update
   `tests/test_api.py` and `tests/test_model_registry.py` accordingly —
   both currently assume single-direction responses and will need
   fixtures for two persisted model artifacts (up and down) per test.

7. **Frontend** (`frontend/src/`):
   - `api/types.ts` / `api/client.ts`: mirror the new dual-direction API shape.
   - New `DirectionBar` component (or extend `ProbabilityBar`): a
     **diverging** bar per the dataviz skill's method — direction/polarity
     (bullish vs bearish) is a diverging color job (blue vs red, the same
     pair already used in `ShapChart`), NOT the sequential blue ramp
     `ProbabilityBar` currently uses for plain magnitude. Concretely:
     center-zero horizontal bar, blue extending right for up-probability,
     red extending left for down-probability, both numeric labels shown
     (never color-alone). This is the actual fix for the user's complaint.
   - `pages/Dashboard.tsx`: replace the single `ProbabilityBar` column with
     the new diverging component; probably also a derived text lean
     ("Bullish"/"Bearish"/"Neutral" — pick sensible thresholds, e.g.
     neutral if both probabilities are within some small delta of each
     other) shown alongside, still carefully NOT framed as a BUY/SELL
     recommendation — keep the existing disclaimer copy, possibly
     strengthen it given the new bullish/bearish language is closer to
     that line than before.
   - `pages/InstrumentDetail.tsx`: show both hero probabilities (Up X% /
     Down Y%) and both SHAP charts (or a toggle) — needs a layout decision,
     not deeply designed yet.
   - Re-verify end-to-end in the Browser tool against the real backend
     once both models are trained and served (same verification rigor as
     Phase 10 — check actual SVG/DOM values, not just screenshots, since a
     browser-tool viewport glitch caused false-alarm debugging last time).

8. **Documentation**: add a new dated entry to `docs/roadmap.md`'s
   methodology log and findings sections once this is done and evaluated
   — including the real up-vs-down rate/AUC comparison numbers, and the
   design rationale (diverging color for direction vs sequential for
   magnitude) consistent with how every other phase was documented. Update
   `README.md`'s headline results table if the downside model's numbers
   are worth featuring.

9. **Commit** the finished feature with a clear message once it's tested
   end-to-end — do NOT commit half-finished (the current mid-edit state
   was intentionally left uncommitted when the user asked to save progress
   instead; finish the checklist above first, or if committing
   incrementally, be explicit in commit messages about what's WIP vs done).

## Outstanding / not yet decided

- **GitHub remote**: user said yes to "create a repo and push," but this
  was interrupted before happening (right when the user pivoted to the
  bullish/bearish feedback). Still needs: confirm repo name/visibility
  with the user, `gh repo create` or equivalent, push. Revisit after the
  downside-model feature is done, unless the user wants it done first —
  ask if unclear.
- Whether to re-run full Phase 6 (calibration) and Phase 7 (SHAP) style
  deep-dives for the downside model, or just get it trained/evaluated at
  baseline rigor (tuning + walk-forward metrics) and ship — leaning toward
  the latter unless the numbers look surprising, but flag this choice to
  the user rather than deciding unilaterally if it comes up.

## Environment gotchas (don't rediscover these)

- **MSYS Python shadowing**: `python` on PATH via Git Bash resolves to an
  MSYS2/MinGW build that can't install standard PyPI wheels. Always use
  `py -3.11` via the **PowerShell** tool for venv/pip work on this
  machine, not Bash. The existing `backend/.venv` is already correctly
  built with the real Windows Python — just use
  `.venv\Scripts\python.exe` from PowerShell for everything.
- **FastAPI/Pydantic v2 + `Literal[int,...]` query params**: silently
  422's on every request (string not coerced to int Literal). Use a plain
  `int` annotation + manual membership check instead (see
  `backend/app/api/routes.py`'s `_validate_horizon`).
- **Browser tool viewport can drop to 0x0** mid-session (seen once,
  caused black screenshots and looked like a real rendering bug in
  Recharts — it wasn't; verified by reading actual SVG `width` attributes
  via `javascript_tool`, which matched the real API values exactly). If
  screenshots look wrong, check `read_page`'s reported `Viewport: WxH`
  before assuming an app bug; `resize_window` with explicit width/height
  fixes it, then reset to `preset: "desktop"` when done.
- **Browser preview tool's `.claude/launch.json`** must live in the
  **scratch workspace** directory (not the project directory) — see the
  file at
  `C:\Users\Nakul Grover\AppData\Roaming\Claude\scratch-workspaces\6394c6bd-1543-44d1-832a-534ad52f142e\305864a0-88a3-4ded-afe6-186204cd6537\scratch-2026-09-08-23b6a7\.claude\launch.json`
  (already correctly configured to run the frontend via
  `node .../vite/bin/vite.js <absolute-frontend-path> --host 127.0.0.1`,
  bypassing `npm.cmd`'s Windows path-with-spaces bug). A duplicate
  project-level `.claude/launch.json` was created by mistake earlier and
  deleted — don't recreate it there, it's unused by the tool.
- **Docker is not installed** in this environment — the Dockerfile is
  written to standard conventions but has never actually been
  `docker build`-tested. Say so if asked about it; don't claim it works.
- User's memory files (in this session's memory dir) already track the
  MSYS gotcha and the stepwise-mentorship feedback preference — no need
  to re-save those, just be aware they exist and follow them (explain
  trade-offs before big decisions, implement + test each step, wait for
  confirmation before moving on — exactly the pattern the user is
  currently exercising by asking for the downside model to be discussed
  before being unilaterally added as a relabel).

## Working style reminder for whoever picks this up

This user explicitly wants: discuss trade-offs before major decisions,
implement one step at a time, test against real data (not just mocks)
before declaring something done, and report negative/surprising results
honestly rather than smoothing them over. The whole project's credibility
(and the reason the README/roadmap read well) comes from actually doing
this consistently — don't cut that corner for the downside-model feature
just because it's late in the project.
