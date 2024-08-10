def pick(path, outpath,  m, n, p, q):
    with open(path, 'r') as f:
        for num, line in enumerate(f):
            if num == m:
                l = line
            if num == n:
                b = line
            if num == q:
                z = line
            if num == p:
                with open(outpath, 'w') as file:
                    file.write(str(l))
                    file.write(str(b))
                    file.write(str(z))
                    file.write(str(line))
