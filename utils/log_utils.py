class LogUtils:
    @staticmethod
    def log_info(*msg):
        print(f"[INFO] {msg}")

    @staticmethod
    def log_debug(*msg):
        print(f"[DEBUG] {msg}")

    @staticmethod
    def log_error(*msg):
        print(f"[ERROR] {msg}")
