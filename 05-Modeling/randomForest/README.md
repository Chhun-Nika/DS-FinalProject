# Random Forest Demo

This folder now includes a small localhost demo for the stress prediction model.

## Files

- `random_forest.ipynb`: notebook version of the current model experiment
- `demo_app.py`: local web server that retrains the random forest and serves a prediction form
- `run_demo.sh`: launcher script that uses the existing `~/sklearn-env` interpreter by default

## Run

From the project root:

```bash
05-Modeling/randomForest/run_demo.sh
```

Or choose a custom host and port:

```bash
05-Modeling/randomForest/run_demo.sh --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Demo Inputs

The form takes the raw user-facing values:

- `Daily_Screen_Time(hrs)`
- `Sleep_Quality(1-10)`
- `Days_Without_Social_Media`
- `Exercise_Frequency(week)`
- `Happiness_Index(1-10)`
- `Social_Media_Platform`

The app rebuilds the engineered features used by the model:

- `screen_sleep_ratio`
- `detox_effect`
- `lifestyle_balance`
- `screen_exercise_ratio`

It also one-hot encodes the social media platform columns before running the prediction.
