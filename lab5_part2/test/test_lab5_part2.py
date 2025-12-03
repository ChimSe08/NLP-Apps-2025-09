# test_lab5_part2.py
# Pytest skeleton for lab5_part2 module

def test_import():
    import src.lab5_part2 as lab
    assert hasattr(lab, "run_all")
