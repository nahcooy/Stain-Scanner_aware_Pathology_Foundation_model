#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2] / "ssl"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import *  # noqa: F401,F403

