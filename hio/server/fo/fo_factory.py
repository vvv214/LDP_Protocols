import math

from server.fo.rr import RR
from server.fo.ue import UE
from server.fo.lh import LH


class FOFactory(object):
    @staticmethod
    def create_fo(name, args, eps=0, domain_size=0):
        if name == 'rr':
            return RR(args)
        elif name == 'ue':
            return UE(args)
        elif name == 'lh':
            return LH(args)
        elif name == 'adap':
            if domain_size > math.exp(eps) * 3 + 2:
                fo = LH(args)
                fo.init_e(eps, domain_size)
            else:
                fo = RR(args)
                fo.init_e(eps, domain_size)
            return fo
