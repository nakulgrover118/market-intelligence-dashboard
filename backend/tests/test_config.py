from app.core.config import Settings


def test_cors_origin_list_defaults_to_vite_dev_server():
    settings = Settings()
    assert settings.cors_origin_list == ["http://localhost:5173", "http://127.0.0.1:5173"]


def test_cors_origin_list_parses_comma_separated_env_value():
    settings = Settings(cors_origins="https://example.com, https://www.example.com ,")
    assert settings.cors_origin_list == ["https://example.com", "https://www.example.com"]


def test_cors_origin_list_handles_single_origin():
    settings = Settings(cors_origins="https://example.com")
    assert settings.cors_origin_list == ["https://example.com"]
