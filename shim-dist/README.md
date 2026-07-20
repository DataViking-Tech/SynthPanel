# synthpanel → althing

**synthpanel has been renamed to [althing](https://pypi.org/project/althing/).**

This package is a transitional shim: installing it simply installs
`althing`, which provides:

- the `althing` CLI (plus a deprecated `synthpanel` alias)
- `import althing` (plus deprecated `synthpanel` / `synth_panel` import shims)
- `ALTHING_*` environment variables (legacy `SYNTHPANEL_*` still honored)

Update your dependencies to `althing` directly:

```bash
pip install althing
```

Site: <https://althing.dev> · Repo: <https://github.com/DataViking-Tech/Althing>

This shim will stop receiving updates one major release after the rename.
