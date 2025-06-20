import numpy as np
from scipy.stats import norm

if __name__ == "__main__":
    from utils import ensure_numpy
else:
    from .utils import ensure_numpy


@ensure_numpy
def nonlinear_color_transform(x, sigma_max=1, n_funcs=5):
    min_ = -3
    max_ = 3

    # Transform x to fit the function
    x = x.astype("float32")
    x /= 255
    x *= max_ - min_
    x += min_

    def get_transformed_vals(x, my, sigma):
        y = norm.cdf(x, loc=my, scale=sigma)

        min_v = norm.cdf(min_, loc=my, scale=sigma)
        max_v = norm.cdf(max_, loc=my, scale=sigma)

        # Scale everyting between min and max
        y = (y - min_v) * (1 / (max_v - min_v))

        return y

    vals = np.zeros_like(x)

    for _ in range(n_funcs):
        vals += get_transformed_vals(x, np.random.randint(-3, 3), np.random.random(1) * sigma_max)

    vals /= n_funcs
    # Vals are now between 0 and 1, transform back into image color space
    vals = np.clip(vals, 0, 1)
    vals *= 255

    return vals
