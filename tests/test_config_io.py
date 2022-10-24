from ms2ml.config import Config


def test_config_toml_export_works(tmpdir):
    """Tests that the config can be exported to a toml file."""
    config = Config()
    config.to_toml(tmpdir / "test.toml")

    assert (tmpdir / "test.toml").exists()

    config2 = Config.from_toml(tmpdir / "test.toml")
    assert config == config2
