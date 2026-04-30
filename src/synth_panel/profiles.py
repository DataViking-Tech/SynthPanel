"""Profile system for named YAML parameter sets.

A profile provides default values for CLI flags (model, temperature, etc.)
that can be loaded by name or path. CLI flags override profile values.

Resolution order: CLI flags > --config/--profile > built-in defaults.

Search order for ``--profile NAME``:
  1. Bundled profiles (package data: ``src/synth_panel/packs/profiles/``)
  2. ``./profiles/NAME.yaml`` (working directory)
  3. ``~/.synthpanel/profiles/NAME.yaml`` (user home)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Profile:
    """A named set of default parameter values."""

    name: str
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    synthesis_model: str | None = None
    synthesis_temperature: float | None = None
    prompt_template: str | None = None
    models: str | None = None  # multi-model spec, e.g. "haiku:0.5,gemini:0.5"
    source_path: str | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict (omitting None values and source_path)."""
        d: dict[str, Any] = {"name": self.name}
        if self.model is not None:
            d["model"] = self.model
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.top_p is not None:
            d["top_p"] = self.top_p
        if self.synthesis_model is not None:
            d["synthesis_model"] = self.synthesis_model
        if self.synthesis_temperature is not None:
            d["synthesis_temperature"] = self.synthesis_temperature
        if self.prompt_template is not None:
            d["prompt_template"] = self.prompt_template
        if self.models is not None:
            d["models"] = self.models
        return d

    def config_hash(self) -> str:
        """SHA256 hash of the profile for reproducibility tracking."""
        serialized = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _bundled_profiles_dir() -> Path:
    """Return the path to bundled profiles in package data."""
    return Path(__file__).parent / "packs" / "profiles"


def _cwd_profiles_dir() -> Path:
    """Return the profiles directory relative to cwd."""
    return Path.cwd() / "profiles"


def _user_profiles_dir() -> Path:
    """Return the user-level profiles directory."""
    return Path.home() / ".synthpanel" / "profiles"


def load_profile_by_name(name: str) -> Profile:
    """Load a profile by name from the search path.

    Search order:
      1. Bundled profiles (package data)
      2. ./profiles/NAME.yaml
      3. ~/.synthpanel/profiles/NAME.yaml

    Raises FileNotFoundError if no matching profile is found.
    """
    # Normalize: strip .yaml extension if provided
    if name.endswith(".yaml") or name.endswith(".yml"):
        name = Path(name).stem

    candidates = [
        _bundled_profiles_dir() / f"{name}.yaml",
        _cwd_profiles_dir() / f"{name}.yaml",
        _user_profiles_dir() / f"{name}.yaml",
    ]
    for path in candidates:
        if path.exists():
            return _load_profile_from_path(path)

    searched = ", ".join(str(p.parent) for p in candidates)
    raise FileNotFoundError(f"Profile '{name}' not found. Searched: {searched}")


def load_profile_from_path(path: str) -> Profile:
    """Load a profile from an arbitrary file path (--config)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return _load_profile_from_path(p)


def _load_profile_from_path(path: Path, _chain: tuple[Path, ...] = ()) -> Profile:
    """Parse a YAML file into a Profile, with inheritance via ``extends:``."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Profile must be a YAML mapping, got {type(data).__name__}")

    extends_spec = data.pop("extends", None)
    parent: Profile | None = None
    if extends_spec is not None:
        parent_path = _resolve_parent_path(str(extends_spec), path)
        if parent_path in _chain:
            cycle = " -> ".join(str(p) for p in (*_chain, parent_path))
            raise ValueError(f"Profile inheritance cycle detected: {cycle}")
        parent = _load_profile_from_path(parent_path, (*_chain, path))

    own = Profile(
        name=data.get("name", path.stem),
        model=data.get("model"),
        temperature=_opt_float(data.get("temperature")),
        top_p=_opt_float(data.get("top_p")),
        synthesis_model=data.get("synthesis_model"),
        synthesis_temperature=_opt_float(data.get("synthesis_temperature")),
        prompt_template=data.get("prompt_template"),
        models=data.get("models"),
        source_path=str(path),
    )
    if parent is None:
        return own
    # Child fields win; None means "inherit from parent".
    return Profile(
        name=own.name,
        model=own.model if own.model is not None else parent.model,
        temperature=own.temperature if own.temperature is not None else parent.temperature,
        top_p=own.top_p if own.top_p is not None else parent.top_p,
        synthesis_model=own.synthesis_model if own.synthesis_model is not None else parent.synthesis_model,
        synthesis_temperature=own.synthesis_temperature if own.synthesis_temperature is not None else parent.synthesis_temperature,
        prompt_template=own.prompt_template if own.prompt_template is not None else parent.prompt_template,
        models=own.models if own.models is not None else parent.models,
        source_path=own.source_path,
    )


def _resolve_parent_path(extends: str, child_path: Path) -> Path:
    """Resolve an ``extends:`` spec to an absolute path.

    Accepts:
    - Relative paths starting with ``./`` or ``../``
    - Absolute paths
    - Profile names (with or without ``.yaml`` extension), searched via the
      same search path as ``load_profile_by_name``, with the child's directory
      searched first.
    """
    if extends.startswith("./") or extends.startswith("../") or Path(extends).is_absolute():
        parent_path = (child_path.parent / extends).resolve()
        if not parent_path.exists():
            raise FileNotFoundError(
                f"Extended profile not found: {extends} (referenced from {child_path})"
            )
        return parent_path

    # Name-based resolution
    name = extends
    if name.endswith(".yaml") or name.endswith(".yml"):
        name = Path(name).stem

    candidates = [
        child_path.parent / f"{name}.yaml",  # same directory as child first
        _bundled_profiles_dir() / f"{name}.yaml",
        _cwd_profiles_dir() / f"{name}.yaml",
        _user_profiles_dir() / f"{name}.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p

    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Extended profile '{name}' not found. Searched: {searched}")


def _opt_float(val: Any) -> float | None:
    """Convert a value to float, treating None as None."""
    if val is None:
        return None
    return float(val)


def apply_profile_to_args(profile: Profile, args: Any) -> dict[str, Any]:
    """Apply profile defaults to argparse args, without overriding explicit CLI flags.

    Returns a dict of which profile fields were applied (for /config display).
    """
    applied: dict[str, Any] = {}

    # model: only if --model not given AND --models not given
    if profile.model and not getattr(args, "model", None) and not getattr(args, "models", None):
        args.model = profile.model
        applied["model"] = profile.model

    # temperature
    if profile.temperature is not None and getattr(args, "temperature", None) is None:
        args.temperature = profile.temperature
        applied["temperature"] = profile.temperature

    # top_p
    if profile.top_p is not None and getattr(args, "top_p", None) is None:
        args.top_p = profile.top_p
        applied["top_p"] = profile.top_p

    # synthesis_model
    if profile.synthesis_model and not getattr(args, "synthesis_model", None):
        args.synthesis_model = profile.synthesis_model
        applied["synthesis_model"] = profile.synthesis_model

    # synthesis_temperature
    if profile.synthesis_temperature is not None and getattr(args, "synthesis_temperature", None) is None:
        args.synthesis_temperature = profile.synthesis_temperature
        applied["synthesis_temperature"] = profile.synthesis_temperature

    # prompt_template
    if profile.prompt_template and not getattr(args, "prompt_template", None):
        args.prompt_template = profile.prompt_template
        applied["prompt_template"] = profile.prompt_template

    # models (multi-model spec): only if --models not given AND --model not given
    if profile.models and not getattr(args, "models", None) and not getattr(args, "model", None):
        args.models = profile.models
        applied["models"] = profile.models

    return applied


def list_available_profiles() -> list[dict[str, str]]:
    """List all discoverable profiles across all search paths."""
    seen: set[str] = set()
    results: list[dict[str, str]] = []
    for label, directory in [
        ("bundled", _bundled_profiles_dir()),
        ("local", _cwd_profiles_dir()),
        ("user", _user_profiles_dir()),
    ]:
        if not directory.exists():
            continue
        for f in sorted(directory.glob("*.yaml")):
            name = f.stem
            if name not in seen:
                seen.add(name)
                results.append({"name": name, "source": label, "path": str(f)})
    return results
