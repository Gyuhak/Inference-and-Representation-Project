# base.py

import utils


class Base:
    def __init__(self):
        self.fig_idx = 0
        self.base_const = 0.1 ** 20

    def init_figure(self, figsize=(10, 10)):
        self.fig_idx += 1
        return utils.init_figure_with_idx(self.fig_idx, figsize)


# End of script
