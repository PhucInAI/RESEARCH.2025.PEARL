# PEARL project page

Project website for **PEARL: Photometric-Error and Attenuation-Regularized Learning for Underwater 3D Gaussian Splatting**.

Static site, no build step: `index.html` + `static/`.

```bash
python3 -m http.server 8000   # preview at http://localhost:8000
```

Deploy: push to a GitHub repo and enable **Settings → Pages → Deploy from branch → `main` / root**.

Before publication, update in `index.html`:
- the Paper and Code buttons (`is-soon` placeholders) with real links,
- the BibTeX `journal` field.
