import json

RESULTS_PATH = "evaluation/results.jsonl"


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main():
    crts_values = []
    sf_values = []
    crr_values = []
    ar_values = []
    ga_values = []

    stance_accs = []
    ga_accs = []

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            crts = obj.get("crts", {})
            crts_values.append(crts.get("crts"))
            sf_values.append(crts.get("sf"))
            crr_values.append(crts.get("crr"))
            ar_values.append(crts.get("ar"))
            ga_values.append(crts.get("ga"))

            stance_accs.append(obj.get("stance_accuracy_agentA"))
            ga_accs.append(obj.get("guideline_alignment_accuracy_agentB"))

    print("Number of cases:", len(crts_values))
    print("Mean CRTS:", mean(crts_values))
    print("Mean SF:", mean(sf_values))
    print("Mean CRR:", mean(crr_values))
    print("Mean AR:", mean(ar_values))
    print("Mean GA:", mean(ga_values))
    print("Mean stance accuracy (Agent A):", mean(stance_accs))
    print("Mean guideline alignment accuracy (Agent B):", mean(ga_accs))


if __name__ == "__main__":
    main()