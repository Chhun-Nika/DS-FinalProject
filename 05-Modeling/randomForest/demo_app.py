from __future__ import annotations

import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "00-dataSource" / "dataSourceV2" / "feature_engineered_data.csv"
TARGET_COLUMN = "Stress_Level(1-10)"
PLATFORM_OPTIONS = [
    "facebook",
    "instagram",
    "linkedin",
    "tiktok",
    "x (twitter)",
    "youtube",
]
MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "random_state": 42,
}


class RandomForestStressDemo:
    def __init__(self) -> None:
        self.dataset = pd.read_csv(DATASET_PATH)
        self.feature_columns = [col for col in self.dataset.columns if col != TARGET_COLUMN]
        self.metrics = self._evaluate_model()
        self.model = self._train_demo_model()

    def _evaluate_model(self) -> dict[str, float]:
        x = self.dataset[self.feature_columns]
        y = self.dataset[TARGET_COLUMN]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
        )

        model = RandomForestRegressor(**MODEL_PARAMS)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        return {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
            "rows": int(len(self.dataset)),
            "features": int(len(self.feature_columns)),
        }

    def _train_demo_model(self) -> RandomForestRegressor:
        model = RandomForestRegressor(**MODEL_PARAMS)
        model.fit(self.dataset[self.feature_columns], self.dataset[TARGET_COLUMN])
        return model

    def validate_form(self, form: dict[str, str]) -> tuple[dict[str, float | str], list[str]]:
        parsed: dict[str, float | str] = {}
        errors: list[str] = []

        def parse_float(field: str, label: str, min_value: float, max_value: float | None = None) -> None:
            raw_value = form.get(field, "").strip()
            if not raw_value:
                errors.append(f"{label} is required.")
                return

            try:
                value = float(raw_value)
            except ValueError:
                errors.append(f"{label} must be a number.")
                return

            if value < min_value:
                errors.append(f"{label} must be at least {min_value}.")
                return
            if max_value is not None and value > max_value:
                errors.append(f"{label} must be at most {max_value}.")
                return

            parsed[field] = value

        parse_float("Daily_Screen_Time(hrs)", "Daily screen time", 0, 24)
        parse_float("Sleep_Quality(1-10)", "Sleep quality", 1, 10)
        parse_float("Days_Without_Social_Media", "Days without social media", 0)
        parse_float("Exercise_Frequency(week)", "Exercise frequency", 0, 7)
        parse_float("Happiness_Index(1-10)", "Happiness index", 1, 10)

        platform = form.get("Social_Media_Platform", "facebook").strip().lower()
        if platform not in PLATFORM_OPTIONS:
            errors.append("Social media platform must be one of the supported options.")
        else:
            parsed["Social_Media_Platform"] = platform

        return parsed, errors

    def build_feature_frame(self, values: dict[str, float | str]) -> pd.DataFrame:
        screen_time = float(values["Daily_Screen_Time(hrs)"])
        sleep_quality = float(values["Sleep_Quality(1-10)"])
        detox_days = float(values["Days_Without_Social_Media"])
        exercise = float(values["Exercise_Frequency(week)"])
        happiness = float(values["Happiness_Index(1-10)"])
        platform = str(values["Social_Media_Platform"])

        row = {
            "Daily_Screen_Time(hrs)": screen_time,
            "Sleep_Quality(1-10)": sleep_quality,
            "Days_Without_Social_Media": detox_days,
            "Exercise_Frequency(week)": exercise,
            "Happiness_Index(1-10)": happiness,
            "screen_sleep_ratio": screen_time / (sleep_quality + 1e-5),
            "detox_effect": detox_days * sleep_quality,
            "lifestyle_balance": exercise * sleep_quality,
            "screen_exercise_ratio": screen_time / (exercise + 1),
            "Social_Media_Platform_instagram": int(platform == "instagram"),
            "Social_Media_Platform_linkedin": int(platform == "linkedin"),
            "Social_Media_Platform_tiktok": int(platform == "tiktok"),
            "Social_Media_Platform_x (twitter)": int(platform == "x (twitter)"),
            "Social_Media_Platform_youtube": int(platform == "youtube"),
        }

        return pd.DataFrame([[row[col] for col in self.feature_columns]], columns=self.feature_columns)

    def predict(self, values: dict[str, float | str]) -> dict[str, float | str]:
        feature_frame = self.build_feature_frame(values)
        prediction = float(self.model.predict(feature_frame)[0])
        clipped_prediction = max(1.0, min(10.0, prediction))

        if clipped_prediction < 4:
            band = "Low"
        elif clipped_prediction < 7:
            band = "Moderate"
        else:
            band = "High"

        derived = feature_frame.iloc[0][
            [
                "screen_sleep_ratio",
                "detox_effect",
                "lifestyle_balance",
                "screen_exercise_ratio",
            ]
        ].to_dict()

        return {
            "prediction": clipped_prediction,
            "band": band,
            "derived": derived,
        }


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def render_page(
    service: RandomForestStressDemo,
    form_values: dict[str, str],
    prediction: dict[str, float | str] | None = None,
    errors: list[str] | None = None,
) -> str:
    errors = errors or []

    platform_options = "\n".join(
        (
            f'<option value="{esc(platform)}"'
            f'{" selected" if form_values.get("Social_Media_Platform", "facebook") == platform else ""}>'
            f"{esc(platform.title())}</option>"
        )
        for platform in PLATFORM_OPTIONS
    )

    error_html = ""
    if errors:
        items = "".join(f"<li>{esc(error)}</li>" for error in errors)
        error_html = f"""
        <section class="panel error-panel">
          <h2>Input Check</h2>
          <ul>{items}</ul>
        </section>
        """

    prediction_html = ""
    if prediction:
        derived = prediction["derived"]
        prediction_html = f"""
        <section class="panel result-panel">
          <div class="result-header">
            <div>
              <p class="eyebrow">Random Forest Prediction</p>
              <h2>{float(prediction["prediction"]):.2f} / 10</h2>
            </div>
            <span class="band">{esc(prediction["band"])}</span>
          </div>
          <p class="result-copy">
            This demo rebuilds the same engineered features used in training, then predicts the expected stress score.
          </p>
          <div class="derived-grid">
            <article>
              <h3>screen_sleep_ratio</h3>
              <p>{float(derived["screen_sleep_ratio"]):.3f}</p>
            </article>
            <article>
              <h3>detox_effect</h3>
              <p>{float(derived["detox_effect"]):.3f}</p>
            </article>
            <article>
              <h3>lifestyle_balance</h3>
              <p>{float(derived["lifestyle_balance"]):.3f}</p>
            </article>
            <article>
              <h3>screen_exercise_ratio</h3>
              <p>{float(derived["screen_exercise_ratio"]):.3f}</p>
            </article>
          </div>
        </section>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stress Prediction Demo</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --paper: rgba(255, 252, 246, 0.92);
      --ink: #173229;
      --muted: #5a6c64;
      --line: rgba(23, 50, 41, 0.14);
      --accent: #1f6b52;
      --accent-soft: rgba(31, 107, 82, 0.12);
      --warning: #9e3d2e;
      --warning-soft: rgba(158, 61, 46, 0.12);
      --shadow: 0 24px 60px rgba(23, 50, 41, 0.12);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(31, 107, 82, 0.16), transparent 32%),
        radial-gradient(circle at top right, rgba(184, 124, 62, 0.18), transparent 25%),
        linear-gradient(180deg, #f8f4ed 0%, var(--bg) 100%);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
    }}

    .shell {{
      width: min(1120px, calc(100% - 32px));
      margin: 32px auto;
      display: grid;
      gap: 20px;
    }}

    .hero, .panel {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }}

    .hero {{
      padding: 28px;
      display: grid;
      gap: 18px;
    }}

    .eyebrow {{
      margin: 0;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 0.78rem;
      font-weight: 700;
    }}

    h1, h2, h3 {{
      margin: 0;
      font-family: "Iowan Old Style", Georgia, serif;
      font-weight: 700;
    }}

    h1 {{
      font-size: clamp(2rem, 5vw, 3.4rem);
      line-height: 1.04;
      max-width: 12ch;
    }}

    .hero-copy {{
      max-width: 64ch;
      color: var(--muted);
      line-height: 1.6;
      margin: 0;
    }}

    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 14px;
    }}

    .metric {{
      padding: 16px 18px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(255,255,255,0.52));
      border: 1px solid var(--line);
    }}

    .metric span {{
      display: block;
      font-size: 0.78rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 8px;
    }}

    .metric strong {{
      font-size: 1.35rem;
    }}

    .content {{
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(300px, 0.9fr);
      gap: 20px;
    }}

    .panel {{
      padding: 24px;
    }}

    form {{
      display: grid;
      gap: 16px;
    }}

    .field-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
    }}

    label {{
      display: grid;
      gap: 8px;
      font-size: 0.95rem;
      font-weight: 600;
    }}

    .hint {{
      font-size: 0.82rem;
      color: var(--muted);
      font-weight: 500;
    }}

    input, select, button {{
      width: 100%;
      border-radius: 16px;
      border: 1px solid rgba(23, 50, 41, 0.18);
      padding: 14px 15px;
      font: inherit;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.92);
    }}

    input:focus, select:focus {{
      outline: 2px solid rgba(31, 107, 82, 0.25);
      border-color: var(--accent);
    }}

    button {{
      cursor: pointer;
      border: none;
      background: linear-gradient(135deg, #1f6b52, #154635);
      color: #fff;
      font-weight: 700;
      letter-spacing: 0.02em;
      transition: transform 140ms ease, box-shadow 140ms ease;
      box-shadow: 0 18px 30px rgba(21, 70, 53, 0.22);
    }}

    button:hover {{
      transform: translateY(-1px);
    }}

    .notes {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.6;
    }}

    .error-panel {{
      border-color: rgba(158, 61, 46, 0.25);
      background: linear-gradient(180deg, rgba(158, 61, 46, 0.08), rgba(255,252,246,0.92));
    }}

    .error-panel ul {{
      margin: 12px 0 0;
      padding-left: 18px;
      color: var(--warning);
      line-height: 1.6;
    }}

    .result-panel {{
      background:
        radial-gradient(circle at top right, rgba(31, 107, 82, 0.13), transparent 35%),
        linear-gradient(180deg, rgba(255,255,255,0.84), rgba(255,252,246,0.96));
    }}

    .result-header {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 12px;
      margin-bottom: 12px;
    }}

    .result-header h2 {{
      font-size: clamp(2rem, 4vw, 3rem);
    }}

    .band {{
      padding: 10px 14px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-size: 0.8rem;
    }}

    .result-copy {{
      color: var(--muted);
      line-height: 1.6;
      margin-bottom: 18px;
    }}

    .derived-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
    }}

    .derived-grid article {{
      padding: 14px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid var(--line);
    }}

    .derived-grid h3 {{
      font-size: 0.95rem;
      margin-bottom: 8px;
    }}

    .derived-grid p {{
      margin: 0;
      font-size: 1.12rem;
      font-weight: 700;
    }}

    @media (max-width: 860px) {{
      .content {{
        grid-template-columns: 1fr;
      }}

      .hero, .panel {{
        border-radius: 20px;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div>
        <p class="eyebrow">Localhost Demo</p>
        <h1>Random Forest Stress Prediction</h1>
      </div>
      <p class="hero-copy">
        This demo uses the current V2 preprocessing and random forest setup from the project. It rebuilds the engineered
        features from a few user inputs, then estimates the stress score on a 1 to 10 scale.
      </p>
      <div class="metrics">
        <div class="metric">
          <span>MAE</span>
          <strong>{service.metrics["mae"]:.3f}</strong>
        </div>
        <div class="metric">
          <span>RMSE</span>
          <strong>{service.metrics["rmse"]:.3f}</strong>
        </div>
        <div class="metric">
          <span>R²</span>
          <strong>{service.metrics["r2"]:.3f}</strong>
        </div>
        <div class="metric">
          <span>Training Rows</span>
          <strong>{service.metrics["rows"]}</strong>
        </div>
      </div>
    </section>

    {error_html}

    <section class="content">
      <section class="panel">
        <p class="eyebrow">Input Form</p>
        <h2>Enter the lifestyle values</h2>
        <form method="post" action="/predict">
          <div class="field-grid">
            <label>
              Daily Screen Time (hrs)
              <input name="Daily_Screen_Time(hrs)" type="number" min="0" max="24" step="0.1" value="{esc(form_values.get("Daily_Screen_Time(hrs)", "6.0"))}">
              <span class="hint">Range used in cleaning: 0 to 24 hours</span>
            </label>
            <label>
              Sleep Quality (1-10)
              <input name="Sleep_Quality(1-10)" type="number" min="1" max="10" step="0.1" value="{esc(form_values.get("Sleep_Quality(1-10)", "6.5"))}">
              <span class="hint">Higher usually means better sleep quality</span>
            </label>
            <label>
              Days Without Social Media
              <input name="Days_Without_Social_Media" type="number" min="0" step="1" value="{esc(form_values.get("Days_Without_Social_Media", "3"))}">
              <span class="hint">Digital detox days</span>
            </label>
            <label>
              Exercise Frequency (week)
              <input name="Exercise_Frequency(week)" type="number" min="0" max="7" step="1" value="{esc(form_values.get("Exercise_Frequency(week)", "3"))}">
              <span class="hint">Days of exercise in a week</span>
            </label>
            <label>
              Happiness Index (1-10)
              <input name="Happiness_Index(1-10)" type="number" min="1" max="10" step="0.1" value="{esc(form_values.get("Happiness_Index(1-10)", "7.5"))}">
              <span class="hint">Self-reported happiness score</span>
            </label>
            <label>
              Social Media Platform
              <select name="Social_Media_Platform">
                {platform_options}
              </select>
              <span class="hint">This is encoded into platform indicator columns</span>
            </label>
          </div>
          <button type="submit">Predict Stress Level</button>
        </form>
      </section>

      <section class="panel">
        <p class="eyebrow">How It Works</p>
        <h2>Engineered features behind the demo</h2>
        <ul class="notes">
          <li><strong>screen_sleep_ratio</strong> compares screen time with sleep quality.</li>
          <li><strong>detox_effect</strong> combines social media break days and sleep quality.</li>
          <li><strong>lifestyle_balance</strong> combines exercise frequency and sleep quality.</li>
          <li><strong>screen_exercise_ratio</strong> compares screen time with activity level.</li>
          <li>The model also uses one-hot encoded social platform columns, with Facebook as the baseline category.</li>
        </ul>
      </section>
    </section>

    {prediction_html}
  </main>
</body>
</html>
"""


class DemoRequestHandler(BaseHTTPRequestHandler):
    service: RandomForestStressDemo

    def do_HEAD(self) -> None:
        if self.path != "/":
            self.send_error(404)
            return

        self._send_html(
            render_page(
                self.service,
                form_values={"Social_Media_Platform": "facebook"},
            ),
            send_body=False,
        )

    def do_GET(self) -> None:
        self._send_html(
            render_page(
                self.service,
                form_values={"Social_Media_Platform": "facebook"},
            )
        )

    def do_POST(self) -> None:
        if self.path != "/predict":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        parsed = {key: values[0] for key, values in parse_qs(body).items()}

        normalized = {
            "Daily_Screen_Time(hrs)": parsed.get("Daily_Screen_Time(hrs)", ""),
            "Sleep_Quality(1-10)": parsed.get("Sleep_Quality(1-10)", ""),
            "Days_Without_Social_Media": parsed.get("Days_Without_Social_Media", ""),
            "Exercise_Frequency(week)": parsed.get("Exercise_Frequency(week)", ""),
            "Happiness_Index(1-10)": parsed.get("Happiness_Index(1-10)", ""),
            "Social_Media_Platform": parsed.get("Social_Media_Platform", "facebook"),
        }

        values, errors = self.service.validate_form(normalized)
        if errors:
            self._send_html(render_page(self.service, normalized, errors=errors), status=400)
            return

        prediction = self.service.predict(values)
        self._send_html(render_page(self.service, normalized, prediction=prediction))

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_html(self, page: str, status: int = 200, send_body: bool = True) -> None:
        payload = page.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if send_body:
            self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the random forest stress demo locally.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind. Default: 8000")
    args = parser.parse_args()

    service = RandomForestStressDemo()
    DemoRequestHandler.service = service

    server = ThreadingHTTPServer((args.host, args.port), DemoRequestHandler)
    print(f"Stress demo running on http://{args.host}:{args.port}")
    print(f"Dataset: {DATASET_PATH}")
    print(
        "Baseline metrics "
        f"(MAE={service.metrics['mae']:.3f}, RMSE={service.metrics['rmse']:.3f}, R2={service.metrics['r2']:.3f})"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
