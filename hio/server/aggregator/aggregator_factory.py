from server.aggregator.ce import CE
from server.aggregator.sr import SR


class AggregatorFactory(object):
    @staticmethod
    def create_aggregator(name, args):
        if name == 'ce':
            return CE(args)
        elif name == 'sr':
            return SR(args)
