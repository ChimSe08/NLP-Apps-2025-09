def test_import_main():
    import src.main as m
    assert hasattr(m, "run_demo")
