import numpy as np

from scipy import signal

if __name__ == "__main__":
    import sys; sys.path.append("..")

    from base_time_series_model import BaseTimeSeriesModel
else:
    from .base_time_series_model import BaseTimeSeriesModel


class MovingAverageTimeSeriesModel(BaseTimeSeriesModel):
    MODES = {
        "e": "exponential",
        "g": "gaussian",
        "l": "linear",
        "exp": "exponential",
        "gauss": "gaussian",
        "lin": "linear",
        "exponential": "exponential",
        "gaussian": "gaussian",
        "linear": "linear",
    }
    def __init__(self, n=1, base=0.5, sigma=None, alignment="end", mode="exponential"):
        assert alignment in ("e", "end", "l", "last", "c", "center"), f"Invalid value for alignment: {alignment}"
        if mode in self.MODES:
            mode = self.MODES[mode]
        assert mode in self.MODES.values()

        if alignment in ("e", "end", "l", "last"):
            alignment = "end"
        elif alignment in ("c", "center"):
            alignment = "center"

            assert n % 2 == 1, "With center alignment the kernel size has to be odd."
        else:
            raise ValueError("Invalid alignment: " + alignment)

        self.n = n
        self.mode = mode
        self.base = base
        self.sigma = sigma
        self.alignment = alignment

    def __call__(self, x_list, seq_ids=None, clip=None):
        if isinstance(x_list, tuple):
            assert len(x_list) == 1, "Moving average does not support multi inputs"

            x_list = x_list[0]

        x = np.stack(x_list)

        if self.alignment == "end":
            if self.mode == "exponential":
                kernel = np.array([[self.base ** i] for i in range(self.n)][::-1])
            elif self.mode == "linear":
                kernel = np.array([[self.n * 1 / self.n * (i + 1)] for i in range(self.n)])
            elif self.mode == "gaussian":
                kernel = np.array([[v] for v in signal.windows.gaussian(2 * self.n, self.sigma, sym=False)[:self.n]])
        elif self.alignment == "center":
            if self.mode == "exponential":
                half_kernel = [[self.base ** i] for i in range(self.n // 2 + 1)][::-1]
                kernel = np.array(half_kernel + half_kernel[:-1][::-1])
            elif self.mode == "linear":
                m = self.n // 2 + 1
                half_kernel = [[i + 1] for i in range(m)]
                kernel = np.array(half_kernel + half_kernel[:-1][::-1])
            elif self.mode == "gaussian":
                kernel = np.array([[v] for v in signal.windows.gaussian(self.n, self.sigma, sym=True)])


        #print("Kernel:", kernel)
        # Potentially expand kernel dimensions for higher dimension input tensors
        while len(kernel.shape) < len(x.shape):
            kernel = kernel[..., None]

        new_rows = []

        for row_id, row in enumerate(x):
            if self.alignment == "end":
                wdw_size = min(self.n, row_id + 1)
                kernel_real = kernel[-wdw_size:]
            elif self.alignment == "center":
                wdw_size = min(self.n, row_id + self.n // 2)
                from_ = self.n // 2 - (self.n // 2 - max(self.n // 2 - row_id, 0))
                to_ = self.n // 2 + 1 + (self.n // 2 - max(row_id + 1 + self.n // 2 - len(x), 0))
                kernel_real = kernel[
                    from_:to_
                ]

            # Normalize kernel
            kernel_real = kernel_real / kernel_real.sum()

            #print("Real:", kernel_real)
            #print(kernel_real * x[max(0, row_id - wdw_size):row_id + 1])

            #print(kernel_real.shape)
            #print(x[max(0, row_id - wdw_size):row_id + 1].shape)
            if self.alignment == "end":
                row_new = np.sum(kernel_real * x[max(0, row_id - wdw_size + 1):row_id + 1], axis=0)
            elif self.alignment == "center":
                row_new = np.sum(kernel_real * x[max(0, row_id - self.n // 2):row_id + 1 + self.n // 2], axis=0)

            new_rows.append(row_new)

        #print("Old x:", x)
        #print("New x:", np.stack(new_rows))

        return np.stack(new_rows)


if __name__ == "__main__":
    x = np.array([
        [1, 2, 3, 4, 5, 1,],
        [2, 2, 5, 3, 8, 1,],
        [1, 3, 3, 4, 9, 1,],
        [1, 3, 3, 4, 9, 2,],
        [1, 3, 3, 4, 9, 1,],
        [1, 3, 3, 4, 9, 1,],
        [1, 3, 3, 4, 9, 1,],
        [1, 3, 3, 4, 9, 1,],
    ])

    mdl = MovingAverageTimeSeriesModel(3, alignment="c", mode="linear", sigma=1)

    out = mdl(x)

    print(out)
