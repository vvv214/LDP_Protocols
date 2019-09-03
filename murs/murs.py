import abc
import math
import numpy as np

import primitive
import analysis


class MURS():

    use_olh = True
    var_value = 0

    def init_local_eps(self, eps_0, delta_c):
        if self.d - 2 > 3 * math.exp(eps_0):
            self.d = round(math.exp(eps_0) + 1)
        self.eps_c = analysis.blanket_naive_fn(eps_0, self.args.delta, self.n, self.d, self.fn_n)
        self.eps_auxi = self.eps_user = self.eps_0 = eps_0
        self.delta_c = delta_c

    def init_central_eps(self, eps_c, delta_c):
        eps_0_grr = analysis.blanket_naive_reverse_fn(eps_c, delta_c, self.n, self.d, self.fn_n)
        eps_0_olh = analysis.blanket_naive_reverse_fn_olh(eps_c, delta_c, self.n, self.fn_n)

        var_grr = var_olh = 1e9
        if eps_0_grr > 0:
            self.eps_0 = eps_0_grr
            var_grr = self.var()
        if eps_0_olh > 0:
            if round(math.exp(eps_0_olh / 2) + 1) < self.d:
                self.eps_0 = eps_0_olh
                self.d = round(math.exp(eps_0_olh / 2) + 1)
                var_olh = self.var()

        if (var_grr == 1e9 and var_olh == 1e9) or max(eps_0_olh, eps_0_grr) < eps_c:

            self.eps_0 = eps_c
            if self.d - 2 > 3 * math.exp(eps_c):
                self.d = round(math.exp(eps_c) + 1)
                self.use_olh = True
            else:
                self.d = self.users.domain
                self.use_olh = False

        if var_grr < var_olh:
            eps_0 = eps_0_grr
            self.d = self.users.domain
            self.use_olh = False
        else:
            eps_0 = eps_0_olh
            self.d = round(math.exp(eps_0 / 2) + 1)
            self.use_olh = True

        self.eps_auxi = self.eps_user = self.eps_0 = eps_0
        self.delta_c = delta_c

    def init_search_eps(self, eps_c, delta_c):
        ns = np.linspace(1000, 2 * self.n, 1000)
        var_list = np.empty_like(ns)
        for fn_i, fn_n in enumerate(ns):
            self.fn_n = fn_n
            self.d = self.users.domain
            self.init_central_eps(eps_c, delta_c)
            var_list[fn_i] = self.var()
        opt_n = ns[np.argmin(var_list)]
        return opt_n

    def est(self):
        p = math.exp(self.eps_0) / (math.exp(self.eps_0) + self.d - 1)
        q = 1 / (math.exp(self.eps_0) + self.d - 1)

        fn_n = self.args.fn_n
        dist = self.users.REAL_DIST
        fake_raw = np.random.randint(0, self.users.domain, fn_n)
        for i in range(fn_n):
            dist[fake_raw[i]] += 1
        if self.use_olh:
            res = primitive.olh(dist, p, q)
        else:
            res = primitive.grr(dist, p, q, self.users.domain)
        res -= fn_n / self.users.domain
        return res

    def var(self):
        p = math.exp(self.eps_0) / (math.exp(self.eps_0) + self.d - 1)
        if self.d == self.users.domain:
            q = 1 / (math.exp(self.eps_0) + self.d - 1)
        else:
            q = 1 / self.d
        p2 = q + (p - q) / self.d
        return (self.n * q * (1 - q) + self.n / self.d * (p * (1 - p) - q * (1 - q)) + self.fn_n * p2 * (1 - p2)) / self.n ** 2 / (p - q) ** 2
