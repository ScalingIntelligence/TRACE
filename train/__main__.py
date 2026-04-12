from .train_grpo import main
from .dist_utils import dist_cleanup

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        dist_cleanup()
