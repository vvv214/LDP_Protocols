import math


def blanket(coeff, binom):
    return math.sqrt(coeff / binom)


def blanket_naive(eps0, delta, n, d):
    coeff = 14 * math.log(2 / delta)
    binom = (n - 1) / (math.exp(eps0) + d - 1)
    return blanket(coeff, binom)


def blanket_naive_olh(eps0, delta, n):
    d = round(math.exp(eps0 / 2) + 1)
    coeff = 4 * 14 * math.log(2 / delta)
    binom = (n - 1) / (math.exp(eps0) + d - 1)
    return blanket(coeff, binom)


def blanket_naive_fn(eps0, delta, n, d, nr):
    coeff = 14 * math.log(2 / delta)
    binom = (n - 1) / (math.exp(eps0) + d - 1) + nr / d
    return blanket(coeff, binom)


def blanket_naive_fn_alone(delta, d, nr):
    coeff = 14 * math.log(2 / delta)
    binom = nr / d
    return blanket(coeff, binom)


def blanket_naive_reverse(eps, delta, n, d):
    ee0 = eps ** 2 * (n - 1) / (14 * math.log(2 / delta)) - d + 1
    if ee0 > 1:
        return math.log(ee0)
    else:
        return -1


def blanket_naive_reverse_olh(eps, delta, n):
    ee0 = ((eps / 2) ** 2 * (n - 1) / (14 * math.log(2 / delta)) - 1) ** 2
    if ee0 > 1:
        return math.log(ee0)
    else:
        return -1


def blanket_naive_reverse_fn(eps, delta, n, d, nr):
    tmp = (n - 1) / (14 * math.log(2 / delta) / eps ** 2 - nr / d)
    if tmp - (d - 1) < 1:
        return -1
    else:
        return math.log(tmp - (d - 1))


def blanket_naive_reverse_fn_olh(eps, delta, n, nr):
    tmp = eps ** 2 * (n - 1 + nr) / (4 * 14 * math.log(2 / delta)) - 1
    if tmp < 1:
        return -1
    else:
        return math.log(tmp) * 2