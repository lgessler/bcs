import html
with open('encow/encow16a01_1.sgml', 'r') as fin, open('encow/encow_sent.txt', 'w') as fout:
    sent = []
    for line in fin:
        line = line.strip()
        if len(line) == 0:
            continue
        if line[0] == '<' and line[-1] == '>':
            if line[0:4] == "</s>":
                fout.write(html.unescape(" ".join(sent) + "\n").replace("&bquo;", '"').replace("&equo;", '"'))
                sent = []
        else:
            sent.append(line.split("\t")[0])

import subprocess
subprocess.run(["split", "-l", "100000", "--suffix-length=3", "-d", "encow/encow_sent.txt", "encow/encow_sent.txt."])
print('OK')
