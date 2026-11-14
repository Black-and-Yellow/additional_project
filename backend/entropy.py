"""Discrete empirical entropy in nats, with explicit clustering semantics."""
import math
import re


def canonical(text):
    return ' '.join(re.findall(r'\w+', text.lower()))


def cluster_answers(answers, equivalent=None):
    if not answers or any(not s.strip() for s in answers):
        raise ValueError('At least one non-empty answer is required.')
    equivalent = equivalent or (lambda a, b: canonical(a) == canonical(b))
    groups = []
    for i, answer in enumerate(answers):
        # Complete-link avoids joining contradictory endpoints through a bridge.
        group = next((g for g in groups if all(equivalent(answer, answers[j]) for j in g['members'])), None)
        if group is None:
            groups.append({'label': answer, 'members': [i]})
        else:
            group['members'].append(i)
    n = len(answers)
    entropy = -sum((len(g['members']) / n) * math.log(len(g['members']) / n) for g in groups)
    return {'clusters': groups, 'entropy_nats': max(0.0, entropy), 'max_entropy_nats': math.log(n), 'sample_count': n}

