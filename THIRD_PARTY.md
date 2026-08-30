# Third-party code and data

RECAL is released under the MIT License (see `LICENSE`).  This file records the
upstream work it builds on.

## MAGIC (code)

<https://github.com/FDUDSDE/MAGIC> — *MAGIC: Detecting Advanced Persistent
Threats via Masked Graph Representation Learning* (USENIX Security 2024).
Licensed MIT.  The following files are ported from or adapted from it; each one
names its upstream counterpart in its module docstring:

| file | upstream | relation |
|---|---|---|
| `src/model/gat.py` | `model/gat.py` | edge-type-conditioned GAT re-implemented on `torch_geometric.MessagePassing` (upstream is DGL) |
| `src/model/loss_func.py` | `model/loss_func.py` | scaled cosine error, copied |
| `src/model/autoencoder.py` | `model/autoencoder.py` | model structure; per-relation decoder heads and the masking schedule are ours |
| `preprocess/parse_cdm.py` | `utils/trace_parser.py` | CDM JSON parser; deduplication key, timestamps and entity-name extraction changed |
| `src/detect.py` | `model/eval.py` | stage-1 k-NN anomaly scorer |
| `src/metrics.py` | `model/eval.py` | the `skip_benign` test-set convention |

MAGIC's license text, reproduced as required:

```
MIT License

Copyright (c) 2023 Jimmyokok

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## THREATRACE (evaluation protocol and labels)

<https://github.com/threaTrace-detector/threaTrace> — *THREATRACE: Detecting and
Tracing Host-Based Threats in Node Level Through Provenance Graph Learning*
(IEEE TIFS vol. 17, 2022).  Two things come from it:

* the node-level counting protocol used for every number we report
  (§VI-C-1, p. 3982), implemented in `src/metrics.py`;
* the ground-truth malicious-UUID lists, `groundtruth/{cadets,theia,trace}.txt`
  in that repository, which this repository expects at `data/label/{ds}.txt`.

Neither the labels nor any DARPA data are redistributed here; see `README.md`
for where to fetch them.

## DARPA Transparent Computing Engagement 3 (data)

<https://github.com/darpa-i2o/Transparent-Computing> — the CDM JSON audit
volumes.  Released by DARPA for public research use; not redistributed here.
