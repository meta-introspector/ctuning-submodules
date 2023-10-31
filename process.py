import os
import subprocess

files = [os.path.join(dirpath, filename)
    for (dirpath, dirs, files) in os.walk('.')
    for filename in (dirs + files)]
for f  in files:
    if f.endswith(".md"):
        out = f + ".org"
        if not os.path.exists(out):
            dd = subprocess.run([
                "pandoc", "-i", f,  "-o", out
            ], capture_output=True)
            if dd.stderr or dd.stdout:
                print(f, dd.stderr,dd.stdout)
            else:
                print(out)
