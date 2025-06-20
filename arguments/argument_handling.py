from copy import deepcopy

from argparse import Namespace


class Argument:
    def __init__(
        self,
        name,
        long_name=None,
        **kwargs,
    ):
        if long_name is None:
            self.long_name = name
            self.short_name = None
        else:
            self.short_name = name
            self.long_name = long_name
        self.kwargs = kwargs

        self._prefix = ""

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, v):
        self._prefix = v

    @property
    def full_short_name(self):
        return self._prefix + self.short_name

    @property
    def full_long_name(self):
        return self._prefix + self.long_name

    def __call__(self, parser):
        name_list = []

        if self.short_name:
            name_list.append("-" + self.full_short_name)

        name_list.append("--" + self.full_long_name)

        parser.add_argument(*name_list, **self.kwargs)


class ArgumentHandler:
    def __init__(self, prefix, arguments):
        self.arguments = deepcopy(arguments)

        for arg in self.arguments:
            arg.prefix = prefix

        self.prefix = prefix
        self.full_name_to_argument = { arg.full_long_name: arg for arg in self.arguments}

    def __call__(self, parser):
        for argument in self.arguments:
            argument(parser)

    def get_args(self, args):
        args = vars(args)

        relevant_args = {}

        for arg, val in args.items():
            if arg in self.full_name_to_argument:
                relevant_args[self.full_name_to_argument[arg].long_name] = val

        return Namespace(**relevant_args)