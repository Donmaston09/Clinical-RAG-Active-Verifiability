import time

def check_nice_alignment(synthesis, nice_guidelines):
    start = time.time()
    matches = []

    for g in nice_guidelines:
        if g["keyword"].lower() in synthesis.lower():
            matches.append(g["id"])

    latency = time.time() - start
    return matches, latency
