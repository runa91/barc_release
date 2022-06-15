
# Modified from:
#   https://github.com/anibali/pytorch-stacked-hourglass 
#   https://github.com/bearpaw/pytorch-pose

import numpy as np

__all__ = ['Logger']


class Logger:
    """Log training metrics to a file."""
    def __init__(self, fpath, resume=False):
        if resume:   ############################################################################
            # Read header names and previously logged values.
            with open(fpath, 'r') as f:
                header_line = f.readline()
                self.names = header_line.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []
                for numbers in f:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(float(numbers[i]))

            self.file = open(fpath, 'a')
            self.header_written = True
        else:
            self.file = open(fpath, 'w')
            self.header_written = False

    def _write_line(self, field_values):
        self.file.write('\t'.join(field_values) + '\n')
        self.file.flush()

    def set_names(self, names):
        """Set field names and write log header line."""
        assert not self.header_written, 'Log header has already been written'
        self.names = names
        self.numbers = {name: [] for name in self.names}
        self._write_line(self.names)
        self.header_written = True

    def append(self, numbers):
        """Append values to the log."""
        assert self.header_written, 'Log header has not been written yet (use `set_names`)'
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.numbers[self.names[index]].append(num)
        self._write_line(['{0:.6f}'.format(num) for num in numbers])

    def plot(self, ax, names=None):
        """Plot logged metrics on a set of Matplotlib axes."""
        names = self.names if names == None else names
        for name in names:
            values = self.numbers[name]
            ax.plot(np.arange(len(values)), np.asarray(values))
        ax.grid(True)
        ax.legend(names, loc='best')

    def plot_to_file(self, fpath, names=None, dpi=150):
        """Plot logged metrics and save the resulting figure to a file."""
        import matplotlib.pyplot as plt
        fig = plt.figure(dpi=dpi)
        ax = fig.subplots()
        self.plot(ax, names)
        fig.savefig(fpath)
        plt.close(fig)
        del ax, fig

    def close(self):
        self.file.close()
