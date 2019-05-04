def decomment(fobj):
    for line in fobj:
        line = line.split("#")[0]  # preserve before '#'
        line = line.strip()  # line might contain whitespace(s). remove them
        if line == '':
            continue
        yield line
