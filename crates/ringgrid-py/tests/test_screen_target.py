import json
import struct

import pytest

from ringgrid import ScreenTargetArtifacts, TargetLayout


def test_screen_target_returns_realized_spec_at_native_size():
    target = TargetLayout.default_hex()
    options = {
        "width_px": 825,
        "height_px": 1200,
        "pixel_pitch_mm": 25.4 / 150,
        "marker_diameter_mm": 7.168,
    }
    result = target.render_fitted_screen_target(options)
    assert isinstance(result, ScreenTargetArtifacts)
    assert struct.unpack(">II", result.png_bytes[16:24]) == (825, 1200)
    assert result.png_bytes[24:26] == bytes((8, 0))
    spec = json.loads(result.json_text)
    assert spec["schema"] == "ringgrid.target.v6"
    assert spec["lattice"]["rows"] == 25
    assert spec["lattice"]["long_row_cols"] == 15
    assert result.metadata["marker_count"] == 363
    assert result.metadata["screen"]["margin_px"] == 8
    options["width_px"] = 1
    with pytest.raises(ValueError, match="margin"):
        target.render_fitted_screen_target(options)
