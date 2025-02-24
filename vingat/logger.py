from datetime import datetime
import pytz


def get_current_time():
    japan_tz = pytz.timezone('Asia/Tokyo')
    return datetime.now(japan_tz).strftime("%Y%m%d-%H%M%S")


def get_run_name():
    return f"run-{get_current_time()}"


def log_metrics(metrics: dict, prefix: str = "", suffix: str = ""):
    txt = ""
    for key, value in metrics.items():
        txt += f"{key}: {value:.4f}, "
    print(f"{prefix} {txt} {suffix}")
