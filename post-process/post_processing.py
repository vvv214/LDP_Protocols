from scipy.stats import norm
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from validate_scripts.l1regls import l1regls
from cvxopt import normal
import cvxpy as cp

domain = 0
epsilon = 0.0
ee = 0.0
n = 0

similarities = None
Y = []
X = []
REAL_DIST = []
NOISY_DIST = []

# old file that do all the things

def generate():
    if args.dist == 'uniform':
        value = np.random.randint(domain)
    elif args.dist == 'zipfs':
        value = int(np.random.zipf(args.zipfs_s)) - 1
        while value < 0 or value >= domain:
            value = int(np.random.zipf(args.zipfs_s)) - 1
    elif args.dist == 'normal':
        # density at 0 will be greater because any value in (-1,1) is rounded to 0
        value = int(np.random.normal(args.normal_mean, args.normal_std))
        while value < 0 or value >= domain:
            value = int(np.random.normal(args.normal_mean, args.normal_std))
    elif args.dist == 'exponential':
        # we start encoding from 0
        value = int(np.random.geometric(args.expo_scale)) - 1
        while value < 0 or value >= domain:
            value = int(np.random.geometric(args.expo_scale)) - 1
    return value


def generate_dist():
    global X, REAL_DIST
    for i in range(n):
        X[i] = generate()
        REAL_DIST[X[i]] += 1


def generate_auxiliary():
    global X, ESTIMATE_DIST, REAL_DIST, NOISY_DIST, n, epsilon, domain, ee, method_num
    domain = args.domain
    epsilon = args.epsilon
    ee = math.exp(epsilon)
    n = args.n_user
    X = np.zeros(n, dtype=np.int)
    REAL_DIST = np.zeros(domain, dtype=np.int)
    NOISY_DIST = np.zeros(domain, dtype=np.int)


def aggregate():
    global NOISY_DIST
    p = 1.0 / 2
    q = 1.0 / (math.exp(epsilon) + 1)

    # step 1: initial count
    TEMP_DIST = np.copy(REAL_DIST)
    ESTIMATE_DIST = np.random.binomial(TEMP_DIST, p)

    # step 2: noise
    TEMP_DIST = np.copy(REAL_DIST)
    TEMP_DIST = n - TEMP_DIST
    ESTIMATE_DIST += np.random.binomial(TEMP_DIST, q)
    NOISY_DIST = np.copy(ESTIMATE_DIST)

    # step 3: normalize
    a = 1.0 / (p - q)
    b = n * q / (p - q)
    ESTIMATE_DIST = a * ESTIMATE_DIST - b

    return ESTIMATE_DIST


def error_metric(ESTIMATE_DIST):
    abs_error = 0.0
    for x in range(domain):
        abs_error += np.abs(REAL_DIST[x] - ESTIMATE_DIST[x]) ** 2
    return abs_error / domain


def cut_below_threshold(RAW_ESTIMATE_DIST):
    estimates = np.copy(RAW_ESTIMATE_DIST)

    p = 1.0 / 2
    q = 1.0 / (math.exp(epsilon) + 1)

    var = q * (1.0 - q) / (p - q) ** 2
    threshold = 1.0 * norm.ppf(1 - 0.05 / domain, 0, 1) * math.sqrt(n * var)

    for i in range(len(estimates)):
        if estimates[i] < threshold:
            estimates[i] = 0
    return estimates


def divide_below_threshold(RAW_ESTIMATE_DIST):
    estimates = np.copy(RAW_ESTIMATE_DIST)

    p = 1.0 / 2
    q = 1.0 / (math.exp(epsilon) + 1)

    var = q * (1.0 - q) / (p - q) ** 2
    threshold = 1.0 * norm.ppf(1 - 0.05 / domain, 0, 1) * math.sqrt(n * var)

    below_count = 0
    for i in range(len(estimates)):
        if estimates[i] < threshold:
            estimates[i] = 0
            below_count += 1

    residue = (n - sum(estimates)) * 1.0 / below_count
    if residue > 0:
        for i in range(len(estimates)):
            if estimates[i] == 0:
                estimates[i] = residue

    return estimates


def generate_reports():
    global Y
    p = 1.0 / 2
    q = 1.0 / (math.exp(epsilon) + 1)

    Y = np.zeros((n, domain), dtype=np.int)
    samples_zero = np.random.random_sample((n, domain))
    samples_one = np.random.random_sample(n)

    for i in range(n):
        y = np.zeros(domain, dtype=np.int8)

        for k in range(domain):
            if samples_zero[i][k] < q:
                y[k] = 1

        y[X[i]] = 1
        if samples_one[i] < p:
            y[X[i]] = 0

        Y[i] = y


def calculate_similarities(fo):
    generate_reports()

    if fo == 'oue':
        p = 1.0 / 2
        q = 1.0 / (math.exp(epsilon) + 1)

        similarities = np.full((n, domain), q)

        for j in range(n):
            for i in range(domain):
                if Y[j][i] == 1:
                    similarities[j][i] = p
    return similarities


def mle(RAW_ESTIMATE_DIST, mode):
    global similarities
    estimates = np.copy(RAW_ESTIMATE_DIST)
    estimates /= n
    estimates = np.array([1.0 / domain] * domain)

    if mode == 'non_neg':
        estimates[estimates < 0] = 0
    elif mode == 'non_neg_normal':
        estimates[estimates < 0] = 0
        estimates /= sum(estimates)
    elif mode == 'normal':
        estimates /= sum(estimates)

    similarities = calculate_similarities('oue')

    estimates_new = np.copy(estimates)
    posterior_probability = np.zeros((n, domain))
    diff = 1
    round_count = 0
    while diff > 1e-6 and round_count < 50:
        diff = 0
        # E step
        for j in range(n):
            denom = 0.0
            for i in range(domain):
                denom += similarities[j][i] * estimates[i]

            for i in range(domain):
                posterior_probability[j][i] = estimates[i] * similarities[j][i] / denom

        # M step
        for i in range(domain):
            estimates_new[i] = sum(posterior_probability[:, i]) / n

        for i in range(domain):
            diff = max(diff, abs(estimates_new[i] - estimates[i]))
        round_count += 1
        # print(round_count, diff)

        estimates = np.copy(estimates_new)
        if mode == 'non_neg':
            estimates[estimates < 0] = 0
        elif mode == 'non_neg_normal':
            estimates[estimates < 0] = 0
            estimates /= sum(estimates)
        elif mode == 'normal':
            estimates /= sum(estimates)

    return estimates * n


def least_square():
    p = 1.0 / 2
    q = 1.0 / (math.exp(epsilon) + 1)

    A = np.zeros((domain, domain))
    for i in range(domain):
        for j in range(domain):
            if i == j:
                A[i][j] = p
            else:
                A[i][j] = q

    b = np.copy(NOISY_DIST) / n

    # m, n = 50, 200
    # A, b = normal(m, n), normal(m, 1)
    # x = l1regls(A, b)

    # p = program(minimize(norm2(Ax - b)), [equals(sum(x), 1)], geq(x, 0)])

    x = cp.Variable(domain)
    objective = cp.Minimize(cp.sum_squares(A * x - b))
    constraints = [0 <= x, x <= 1, sum(x) == 1]
    prob = cp.Problem(objective, constraints)

    # The optimal objective value is returned by `prob.solve()`.
    result = prob.solve()
    # The optimal value for x is stored in `x.value`.
    # print(x.value)
    return x.value * n


def error_emp():
    errors = []

    RAW_ESTIMATE_DIST = aggregate()
    CLIP_ESTIMATE_DIST = cut_below_threshold(RAW_ESTIMATE_DIST)
    # DIVIDE_ESTIMATE_DIST = divide_below_threshold(RAW_ESTIMATE_DIST)
    # MLE_RAW_ESTIMATE_DIST = mle(RAW_ESTIMATE_DIST, 'raw')
    # MLE_NNEG_ESTIMATE_DIST = mle(RAW_ESTIMATE_DIST, 'non_neg')
    # MLE_NOR_ESTIMATE_DIST = mle(RAW_ESTIMATE_DIST, 'normal')
    # MLE_NNEG_NOR_ESTIMATE_DIST = mle(RAW_ESTIMATE_DIST, 'non_neg_normal')
    LST_SQ_ESTIMATE_DIST = least_square()

    errors.append(error_metric(RAW_ESTIMATE_DIST))
    errors.append(error_metric(CLIP_ESTIMATE_DIST))
    # errors.append(error_metric(DIVIDE_ESTIMATE_DIST))
    # errors.append(error_metric(MLE_RAW_ESTIMATE_DIST))
    # errors.append(error_metric(MLE_NNEG_ESTIMATE_DIST))
    # errors.append(error_metric(MLE_NOR_ESTIMATE_DIST))
    # errors.append(error_metric(MLE_NNEG_NOR_ESTIMATE_DIST))
    errors.append(error_metric(LST_SQ_ESTIMATE_DIST))

    if args.exp_round == 1:
        draw_round = 100
        RAW_ESTIMATE_DIST = []
        CLIP_ESTIMATE_DIST = []
        MLE_RAW_ESTIMATE_DIST = []
        MLE_NNEG_ESTIMATE_DIST = []
        MLE_NOR_ESTIMATE_DIST = []
        MLE_NNEG_NOR_ESTIMATE_DIST = []
        LST_SQ_ESTIMATE_DIST = []

        for _ in range(draw_round):
            temp_raw = aggregate()
            # RAW_ESTIMATE_DIST.append(temp_raw)
            CLIP_ESTIMATE_DIST.append(cut_below_threshold(temp_raw))
            # MLE_RAW_ESTIMATE_DIST.append(mle(temp_raw, 'raw'))
            # MLE_NNEG_ESTIMATE_DIST.append(mle(temp_raw, 'non_neg'))
            # MLE_NOR_ESTIMATE_DIST.append(mle(temp_raw, 'normal'))
            # MLE_NNEG_NOR_ESTIMATE_DIST.append(mle(temp_raw, 'non_neg_normal'))
            LST_SQ_ESTIMATE_DIST.append(least_square())

        draw_comparison(np.array([
            REAL_DIST,
            # np.mean(RAW_ESTIMATE_DIST, axis=0),
            np.mean(CLIP_ESTIMATE_DIST, axis=0),
            # np.mean(MLE_RAW_ESTIMATE_DIST, axis=0),
            # np.mean(MLE_NNEG_ESTIMATE_DIST, axis=0),
            # np.mean(MLE_NOR_ESTIMATE_DIST, axis=0),
            # np.mean(MLE_NNEG_NOR_ESTIMATE_DIST, axis=0)
            np.mean(LST_SQ_ESTIMATE_DIST, axis=0)
            ]))
    return errors


def draw_comparison(datas):
    xticks = int(domain / 20)
    print(datas)
    for i in range(len(datas)):
        data = datas[i]
        x_data = []
        y_data = []
        y_error = []

        for d in range(domain):
            if not d % xticks == 0:
                continue

            x_data.append(d)

            y_data.append(data[d])
            y_error.append(0)

        (_, caps, _) = plt.errorbar(x_data, y_data, y_error, label=i,
                                    fillstyle='none',
                                    markeredgewidth=2.0)

        for cap in caps:
            cap.set_markeredgewidth(2)

    plt.legend(fontsize=15, ncol=3, loc="best")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.yscale("log")
    plt.savefig('post_processing_%s_e%d.pdf' % (args.dist, epsilon * 10))
    plt.cla()


def main():
    method_num = 3
    results = np.zeros((args.exp_round, method_num))
    for i in range(args.exp_round):
        results[i] = error_emp()

    returned_results = []
    for i in range(method_num):
        print('%1.1e' % np.mean(results[:, i]), end=' ')
        returned_results.append(np.mean(results[:, i]))
    print(end='\n\t')
    for i in range(method_num):
        print('%1.1e' % np.std(results[:, i]), end=' ')
    print()
    return returned_results


def dispatcher():
    global g, ee, epsilon

    generate_auxiliary()
    generate_dist()

    results = []
    epss = np.round(np.linspace(0.2, 4.0, 20), 1)
    # epss = np.round(np.linspace(4, 4.4, 2), 1)
    # epss = np.round(np.linspace(1, 1.4, 2), 1)
    # epss = np.round(np.linspace(0.4, 0.8, 3), 1)

    for e in epss:
        ee = math.exp(e)
        epsilon = e
        args.epsilon = float(e)
        print('%.1f' % epsilon, end='\t')
        result = main()
        results.append(result)

    if args.exp_round > 1:
        draw(results, epss, [
            'raw',
            'divide',
            # 'mle_raw',
            # 'mle_nneg',
            # 'mle_norm',
            # 'mle_nneg_norm'
            'lst_sq'
        ])


def draw(results, epss, column_names):
    df = pd.DataFrame(results, index=epss, columns=column_names)
    df['eps'] = df.index
    df = df.melt('eps', var_name='post-process', value_name='var')

    sns.set_context("notebook", font_scale=0.7, rc={"lines.linewidth": 0.5})
    sns_plot = sns.factorplot(x='eps', y='var', hue='post-process', data=df, marker=',')
    sns_plot.fig.get_axes()[0].set_yscale('log')
    sns_plot.savefig('post_processing_%s.pdf' % (args.dist))


parser = argparse.ArgumentParser(description='Comparisor of different schemes.')
parser.add_argument('--domain', type=int, default=102400,
                    help='specify the domain of the representation of domain')
parser.add_argument('--n_user', type=int, default=100000,
                    help='specify the number of data point, default 10000')
parser.add_argument('--exp_round', type=int, default=1,
                    help='specify the n_userations for the experiments, default 10')
parser.add_argument('--epsilon', type=float, default=1,
                    help='specify the differential privacy parameter, epsilon')

parser.add_argument('--dist', type=str, default='zipfs',
                    help='specify the distribution')
parser.add_argument('--normal_mean', type=float, default=50,
                    help='specify the mean of the normal distribution (if used)')
parser.add_argument('--normal_std', type=float, default=50,
                    help='specify the std of the normal distribution (if used)')
parser.add_argument('--expo_scale', type=float, default=0.75,
                    help='specify the parameter of the exponential distribution (if used)')
parser.add_argument('--zipfs_s', type=float, default=1.05,
                    help='specify the s parameter of the zipfs distribution(if used)')
args = parser.parse_args()
dispatcher()
