# GNM Webcam Puppet

Two independent demos that drive Google's [GNM Head](https://github.com/google/GNM)
parametric face model from a live webcam: track a face, solve it into GNM
identity / expression / pose parameters, render the result in real time.

| Directory | What it is |
| --- | --- |
| [`webcam_puppet/`](webcam_puppet/README.md) | Python desktop app — MediaPipe tracking, NumPy solve, on-screen render |
| [`web_puppet/`](web_puppet/README.md) | Browser rebuild — same idea, runs client-side on the GPU, no Python at runtime |

Neither directory modifies or vendors the GNM package itself; both depend on
it via `pip install gnm-shape` (or `pip install -e ./gnm/shape` from a GNM
checkout). See each subproject's own README for setup and status.

## License

Apache 2.0 — see [LICENSE](LICENSE).
