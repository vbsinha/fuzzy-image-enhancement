import scipy.stats

def medium(x):
    assert x >=0 and x <= 1
    return 0.25 * scipy.stats.norm(0.4,0.1).pdf(x)


def small(x):
    assert x >=0 and x <= 1
    return 0.5 * scipy.stats.norm(0,0.2).pdf(x)


def large(x):
    assert x >=0 and x <= 1
    if x < 0.8:
        return 0.5 * scipy.stats.norm(0.8,0.2).pdf(x)
    else:
        return 1