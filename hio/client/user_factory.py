from client.uci_user import UCIUser


class UserFactory(object):
    @staticmethod
    def create_user(name, args):
        return UCIUser(args)