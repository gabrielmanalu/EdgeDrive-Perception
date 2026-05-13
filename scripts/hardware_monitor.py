import subprocess
import re
import sys
import time

# --- Regex patterns ---
RAM_RE = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")
CPU_RE = re.compile(r"\bCPU\s+\[([^\]]+)\]")
GPU_RE = re.compile(r"\bGR3D_FREQ\s+(\d+)%(?:@(?:\[(\d+)\]|(\d+)))?")
TEMP_CPU_RE = re.compile(r"\bcpu@([0-9.]+)C")
TEMP_GPU_RE = re.compile(r"\bgpu@([0-9.]+)C")
PWR_IN_RE = re.compile(r"\bVDD_IN\s+(\d+)mW")
PWR_CV_RE = re.compile(r"\bVDD_CPU_GPU_CV\s+(\d+)mW")
PWR_SOC_RE = re.compile(r"\bVDD_SOC\s+(\d+)mW")


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def fmt_gb(mb):
    return f"{mb / 1024:.1f} GB"


def parse_cpu_avg(cpu_text):
    # Example: "32%@1728,24%@1728,19%@1728,21%@1728,25%@1728,27%@1728"
    usages = []
    for core in cpu_text.split(","):
        core = core.strip()
        m = re.search(r"(\d+)%@", core)
        if m:
            usages.append(int(m.group(1)))
    if not usages:
        return "N/A"
    return f"{sum(usages) / len(usages):.0f}%"


def parse_line(line):
    data = {
        "ram": "N/A",
        "cpu_avg": "N/A",
        "gpu": "N/A",
        "cpu_temp": "N/A",
        "gpu_temp": "N/A",
        "pwr_in": "N/A",
        "pwr_cv": "N/A",
        "pwr_soc": "N/A",
    }

    ram_match = RAM_RE.search(line)
    if ram_match:
        used_mb = int(ram_match.group(1))
        total_mb = int(ram_match.group(2))
        data["ram"] = f"{fmt_gb(used_mb)} / {fmt_gb(total_mb)}"

    cpu_match = CPU_RE.search(line)
    if cpu_match:
        data["cpu_avg"] = parse_cpu_avg(cpu_match.group(1))

    gpu_match = GPU_RE.search(line)
    if gpu_match:
        gpu_usage = gpu_match.group(1)

        # clock may not exist
        gpu_clock = gpu_match.group(2) or gpu_match.group(3)

        if gpu_clock:
            data["gpu"] = f"{gpu_usage}% @ {gpu_clock} MHz"
        else:
            data["gpu"] = f"{gpu_usage}%"

    temp_cpu = TEMP_CPU_RE.search(line)
    if temp_cpu:
        data["cpu_temp"] = f"{temp_cpu.group(1)} °C"

    temp_gpu = TEMP_GPU_RE.search(line)
    if temp_gpu:
        data["gpu_temp"] = f"{temp_gpu.group(1)} °C"

    pwr_in = PWR_IN_RE.search(line)
    if pwr_in:
        data["pwr_in"] = f"{int(pwr_in.group(1)) / 1000:.2f} W"

    pwr_cv = PWR_CV_RE.search(line)
    if pwr_cv:
        data["pwr_cv"] = f"{int(pwr_cv.group(1)) / 1000:.2f} W"

    pwr_soc = PWR_SOC_RE.search(line)
    if pwr_soc:
        data["pwr_soc"] = f"{int(pwr_soc.group(1)) / 1000:.2f} W"

    return data


def render_ui(data):
    return f"""
==========================================
        EdgeDrive Hardware Monitor
==========================================
 🧠 CPU Avg Load :  {data['cpu_avg']}
 🌡️  CPU Temp     :  {data['cpu_temp']}
------------------------------------------
 🎮 GPU Load     :  {data['gpu']}
 🌡️  GPU Temp     :  {data['gpu_temp']}
------------------------------------------
 🔋 Total Power  :  {data['pwr_in']} (VDD_IN)
 ⚡ CPU/GPU Pwr  :  {data['pwr_cv']} (CV)
 ⚡ SOC Power    :  {data['pwr_soc']} (SOC)
------------------------------------------
 💾 RAM Usage    :  {data['ram']}
==========================================
""".rstrip()


def main():
    print("Starting EdgeDrive Hardware Monitor...")

    # Use a 1-second interval. You can change to 100 or 200 for faster updates.
    cmd = ["tegrastats", "--interval", "1000"]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        while True:
            line = process.stdout.readline()

            if not line:
                break


            line = line.strip()
            # Uncomment this for debugging exact raw output:
            # print("RAW:", line)

            data = parse_line(line)

            clear_screen()
            print(render_ui(data))
            sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nStopping Hardware Monitor...")
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()