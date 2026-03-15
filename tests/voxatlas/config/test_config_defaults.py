from voxatlas.config.config import expand_defaults


def test_expand_defaults():

    user_cfg = {
        "features": ["acoustic.pitch.f0"]
    }

    final = expand_defaults(user_cfg)

    assert final["features"] == ["acoustic.pitch.f0"]

    assert "pipeline" in final
    assert final["pipeline"]["cache"] is True