import subprocess

interface_number = '7'  # Use your interface number
cmd = [r'C:\Program Files\Wireshark\tshark.exe', '-i', interface_number, '-T', 'json']

print("Starting tshark subprocess...")
try:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("Subprocess started. Reading from tshark...")
except Exception as e:
    print(f"Failed to start tshark subprocess: {e}")
    exit(1)

max_lines = 1000  # Change this to 20 if you prefer
line_count = 0

with open("cool_output.json", "w") as output_file:
    print("Output file 'cool_output.json' created.")
    try:
        for line in proc.stdout:
            print(f"[DEBUG] Line {line_count+1} read from tshark: {line.strip()[:80]}")
            output_file.write(line)
            line_count += 1
            if line_count >= max_lines:
                print(f"[DEBUG] Reached {max_lines} lines. Exiting loop.")
                break
    except KeyboardInterrupt:
        print("[DEBUG] Capture interrupted by user.")

proc.terminate()
print("Stopped live capture and closed file.")
print("[DEBUG] Script finished.")
