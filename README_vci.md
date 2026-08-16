## Summary table

| Feature | Files changed |
|---|---|
| Confidence-aware pruning | `arguments/__init__.py`, `scene/gaussian_model.py`, `train.py` |
| Residual-guided densification | `arguments/__init__.py`, `scene/gaussian_model.py`, `utils/graphics_utils.py`, `train.py` |
| Trajectory smoothness loss | `arguments/__init__.py`, `scene/gaussian_model.py`, `train.py` |

For an example on running on VCI data, see `vci/batch.sh`.

Typically:
* Run `prepare_data.py` to reorganize data from VCI dataset structure to the one 4DGS expect (and apply scaling and masking if needed).
* Run `multiviewtria.sh` to run the COLMAP triangulation pipeline using VCI's calibrated camera data.
* `train.py`, `render.py` and `metrics.py` like in the `README.md`.
  Use the respective config files from `arguments/multipleview/vci/*` to set the improvements hyperparameters.
  `vci/train.sh`, `vci/render.sh`, `vci/metrics.sh` are just batch scripts to train multiple ablations consecutively (example in `vci/batch.sh`).
