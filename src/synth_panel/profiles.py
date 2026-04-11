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


def _load_profile_from_path(path: Path) -> Profile:
    """Parse a YAML file into a Profile."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Profile must be a YAML mapping, got {type(data).__name__}")
    return Profile(
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
