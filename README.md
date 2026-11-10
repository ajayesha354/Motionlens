# IMU Activity Lab

Recognise movement from a wearable motion sensor (IMU) with a small 1D CNN, and estimate mechanical loading.
Upload a CSV to the website and the model runs **in your browser**. Nothing is uploaded.

- Website: `web/index.html` (single file, also copied to `docs/index.html` for GitHub Pages)
- Data: [MotionSense](https://github.com/mmalekzadeh/motion-sense) (MIT). 24 people, iPhone in the front trouser pocket, 50 Hz, six activities: walking, jogging, upstairs, downstairs, sitting, standing.
- Model: 33k-parameter 1D CNN, trained in JAX, re-implemented in plain JavaScript for the browser.

## Results (honest version)

Everything is split **by person**: the people in the test set are never seen in training.

| Model | Held-out test (4 people) | 6-fold CV (every fold tests 4 unseen people) |
|---|---|---|
| CNN, accelerometer + gyroscope | 88.7% (macro F1 0.854) | 94.3% ± 3.7 |
| CNN, accelerometer only | 88.2% (macro F1 0.850) | 93.4% ± 3.1 |
| Random forest on hand-made features, 9 ch | 91.0% | 93.1% ± 2.2 |
| Random forest, 6 ch | 90.5% | 91.0% ± 2.3 |

How to read this:
- The CNN is about 1 to 2 points ahead in cross-validation (same folds as the baseline), but that is inside the person-to-person spread. It is **not** a clear win on accuracy.
- Accuracy depends strongly on the person. One test person (no. 19) walks in a way the model reads as stair-descent (walking recall 1%). Without them the test score is above 95%.
- Gravity-aligned input channels (acceleration along and across gravity) raised tuning accuracy from 93.2% to 95.4%. The choice was made by cross-validation on the 20 training people, not on the test people.
- Smoothing over time (Viterbi) did not help on synthetic test streams, so it is off by default.
- Confidence scores are only roughly calibrated (calibration error about 0.06 after temperature scaling).

## Joint loading: what it is and is not

The loading numbers are **mechanical proxies from acceleration at the sensor**: accumulated acceleration change (g/s, similar to the sports-science PlayerLoad idea), peak acceleration, dominant rhythm, and a 0 to 100 index (percentile among about 13,000 reference windows from the training people).
They are **not joint forces or moments**. No public IMU dataset with measured joint loads was available. To estimate real joint loads, fit a regression from these windows to your own labels (motion capture plus force plates, OpenSim inverse dynamics) and replace `window_load` in `imuhar/signal.py` and in the page.

## Limits

- Trained on a phone in the front trouser pocket. Other placements or activities (squats, cycling, driving) will be misclassified or flagged as low confidence.
- Sitting vs standing relies on gravity direction, so mounting orientation matters.
- Six activities, 24 adults. Not validated for children, older adults or clinical gait.
- The cross-validation figures use 12 training epochs and one seed per fold.

## What I learned

- **Split by person, not by window.** Neighbouring windows from the same person are near-duplicates, so a random split would likely inflate the score. Holding out whole people shows how the model behaves on someone new.
- **An average hides the people it fails on.** Most test people scored above 95%, but one scored 1% on walking. Reporting results per person told me more than one accuracy number.
- **A simple baseline keeps you honest.** A random forest on hand-made features came within 1 to 2 points of the CNN. The CNN is a reasonable choice here, but it is not a big win, and I say so in the results.
- **The input representation mattered more than the network.** Adding gravity-aligned channels (motion along and across gravity, which do not change when the phone rotates in the pocket) raised cross-validated accuracy from 93.2% to 95.4%. Making the network bigger was not the lever.
- **Check for overfitting early.** My first CNN peaked on validation after a few epochs and then got worse. Fewer epochs, more dropout and stronger rotation augmentation helped.
- **Not every extra helps.** Smoothing predictions over time looked sensible but did not improve my test streams, so I turned it off by default.
- **Test the deployed version, not just the trained one.** The website re-implements the pipeline in JavaScript, so I wrote tests that compare it with the Python version on the same file, including unit and sampling-rate handling.
- **Name a proxy as a proxy.** The loading index comes from acceleration at the sensor. It is not a joint force, and the page and README say so.

## What I would improve

1. **More varied data.** More people, other sensor positions (thigh, waist, wrist) and more activities, including squats. Then test on a second public dataset to see if it works beyond MotionSense.
2. **Handle unusual people.** Fine-tune on a minute of a new person's data, or adapt the model at test time, to fix cases like the walker the model got wrong.
3. **Real joint load.** Train a regression on my own motion-capture and force-plate data (for example from OpenSim inverse dynamics) and report the error against measured values. This is the step that would connect the project to my thesis work.
4. **Know when it does not know.** Better calibrated confidence and a check for data unlike the training set, so the page can say "I am not sure" more reliably.
5. **Stronger evaluation.** Several random seeds, confidence intervals, and ablations (which channels, which augmentations) instead of one run per fold.
6. **Other models.** Compare against a temporal convolutional network or a small transformer, and a smaller distilled model for phones.
7. **Realistic streams.** The demo is stitched from separate trials, and the stream test is synthetic. I would evaluate on long continuous recordings with real transitions.
8. **Engineering.** Test in real browsers and on phones, and move the model to a background thread so large files do not freeze the page.

## Reproduce

```bash
pip install -r requirements.txt
python -c "from imuhar import data; r=data.download('data_raw'); data.build('data_raw','cache/ms')"
python -m imuhar.train          # trains both models (about 2 min on CPU), writes models/ and results/metrics.json
python -m imuhar.cv             # 6-fold subject-wise CV (about 15 min)
python -m imuhar.export_web     # builds web/index.html (needs results/baseline_rf.json from the baseline run)
python -m pytest                # 15 tests, including JavaScript vs Python parity (needs Node)
```
`results/` holds the saved metrics, the tuning runs (`tune.json`) and the baseline.
`export_web.py` expects the raw MotionSense folder at `data_raw/motion-sense-master/data/A_DeviceMotion_data` for the demo session and reference statistics.

## CSV format for the website

A header row with a time column (seconds, ms, µs or ns, auto-detected) or a known sampling rate, plus `ax ay az` (g or m/s², total acceleration including gravity) and optionally `gx gy gz` (rad/s or deg/s). Column names are auto-detected and can be remapped. MotionSense files (`userAcceleration` + `gravity`) work directly.

## Layout

`imuhar/` signal processing, features, JAX model, training, CV, web export. `web/template.html` is the page source. `tests/` unit and parity tests. `samples/demo_session.csv` is a stitched demo from a held-out person.
