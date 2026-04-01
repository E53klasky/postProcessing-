import numpy as np
import matplotlib.pyplot as plt
import adios2
import ReaderClass

# -------------------------------------------------------
# File definitions
# -------------------------------------------------------

files = {
    "129-257":   "div_curl_re7k/div_curl_129_257_re7k.bp",
    "257-513":   "div_curl_re7k/div_curl_257_513_re7k.bp/",
    "513-1025":  "div_curl_re7k/div_curl_513_1025_re7k.bp/",
    "1025-2049": "div_curl_re7k/div_curl_1025_2049_re7k.bp",
    "2049-4097": "div_curl_re7k/div_curl_2049_4097_re7k.bp/",
}

VAR    = "Div"
READIO = "reader1"

# -------------------------------------------------------
# Read data
# -------------------------------------------------------

all_means = {}

for res, bp in files.items():

    print(f"Reading {res}: {bp}")

    r = ReaderClass.Reader(READIO, bp)

    means = []

    while True:

        status = r.begin_step()

        if status != adios2.bindings.StepStatus.OK:
            break

        r.set_read_vars([VAR])

        data = r.read_step(VAR)

        if data is not None:
            means.append(np.max(np.abs(data.squeeze()))) 

        r.end_step()

    all_means[res] = np.array(means)
    print(f"  -> {len(means)} steps, values: {means}")

# -------------------------------------------------------
# Plot
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

for res, means in all_means.items():
    ax.plot(range(len(means)), means, 'o-', label=res, markersize=4)

ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlabel("Snapshot index")
ax.set_ylabel(f"max of {VAR}")
ax.set_title(f"Max {VAR} across snapshots — all grid levels (RE 7k)")
ax.legend(fontsize=9)
fig.tight_layout()

outfile = f"../RESULTS/timehist_{VAR}_re7k.png"
plt.savefig(outfile, dpi=200)
plt.close()
print("Saved", outfile)
