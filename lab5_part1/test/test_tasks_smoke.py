def test_run_each_task_smoke():
    from src.tensors_intro import run_task1_tensors
    from src.autograd_intro import run_task2_autograd
    from src.nn_intro import run_task3_nn

    # Smoke tests: chỉ cần chạy không lỗi (có thể in ra nhiều)
    run_task1_tensors()
    run_task2_autograd()
    run_task3_nn()
